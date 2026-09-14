"""Cancer / no-cancer on a full DBT exam: step 1's decision head, at last measurable.

    python -m imaging.examclf --data-dir data/preprocessed_data/dbt_exams --folds 5

What this is, and why it could not exist before
-------------------------------------------------
Every model this project trained on DBT so far was scored on a corpus where 100 % of
patients carried a cancer -- annotation-driven preprocessing never wrote a negative,
so a specificity, a PPV or a patient-level ROC-AUC were not measurable at all (plan.md,
"Cible chiffrée"). ``TransformData.preprocess_dbt_exams`` is what changed that: it
labels a series from the per-view status table, so a series with no box is a negative
rather than a skip, and both classes share one geometry. This module is the first
thing trained on the result: 870 exams, 272 patients, 610 normal / 153 benign / 107
cancer series (56 cancer patients).

Multiple-instance learning, and why max rather than mean
----------------------------------------------------------
The label is exam-level -- cancer or not -- but the signal, when it exists, occupies a
handful of slices out of 24-114: most slices of a cancer exam show nothing, unlike a
corpus already cropped to the lesion, where every slice could safely be averaged.
Averaging a mostly-empty exam down to one score would wash out exactly the
minority of slices that matter, and is the same failure mode ``sliceclf``'s docstring
already names: "for detection ... one suspicious slice is enough". So a bag's score is
the **max** over its slices, at both training and evaluation. Training samples a
random subset of ``--bag-size`` slices per exam per step rather than the whole exam
(869 exams x up to 114 slices would not fit a batch); across enough steps the model
sees every slice of every training exam multiple times. The gradient of a max only
flows through the arg-max slice of each bag, which is the point: for a positive bag,
whichever sampled slice currently looks most suspicious is pushed further that way; for
a negative bag, every sampled slice is pushed down, because a negative bag truly holds
no positive instance to spare.

The encoder is ``sliceclf.SliceClassifier``, reused rather than re-derived: its
GroupNorm stack, not BatchNorm (see plan.md §4.1 on why BatchNorm collapses here), and
its avg+max pooled head already separate "how much of the slice looks abnormal" from
"is there one abnormal spot" -- exactly the two things a single suspicious region needs.
224 is chosen for the bank's ``image_size`` because it is ``32 * 7``: the five stride-2
pooling stages divide it with no rounding.

Why cross-validation, again
----------------------------
89 cancer patients exist in the whole BCS-DBT collection (plan.md); this corpus holds
56 of them, and none are held back for tuning here. A single 15 % test split would
leave roughly 8-9 cancer patients, an interval wide enough to be compatible with
chance. Pooling out-of-fold predictions scores every patient exactly once, with a
model that never saw them -- coverage, not independence, since the folds share a
corpus and a hyper-parameter choice. No checkpoint or epoch is selected on held-out
patients: the epoch budget is fixed in advance and the last epoch is scored -- the
project has already published one metric that skipped this discipline and paid for it
in an optimistic number (plan.md, "Écarts doc <-> code").

Patient level, not exam level
-------------------------------
A patient can contribute up to four views (exams). The score reported is per
**patient** -- the level plan.md's target is stated at -- taking the max over that
patient's exam scores (one suspicious view is enough to flag the patient), and the
label is 1 if any of the patient's exams is a cancer.

Two levers on top of that, both off by default so the published measurements stay
reproducible (plan.md, "Prochaines pistes pour l'étape 1")
------------------------------------------------------------------------------------
``--warm-start-epochs N`` trains the encoder on the denser slice-level question
first -- "is a lesion visible on this slice?", from the boxes the bank already
carries -- on the fold's **training** exams only, then hands those weights to the MIL
phase (:func:`warm_start_encoder`). ``--top-k K`` averages the K most suspicious
slices of a bag instead of betting it on the single arg-max (:func:`bag_logit`).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:  # allow both "python -m imaging.examclf" and direct execution
    from .dataset import _gamma_jitter, default_patient_key, kfold_by_patient
    from .exambank import ExamBank, build_exam_bank, exam_id_for_path
    from .metrics import bootstrap_auc, operating_point, roc_auc
    from .sliceclf import SliceClassifier
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset import _gamma_jitter, default_patient_key, kfold_by_patient
    from exambank import ExamBank, build_exam_bank, exam_id_for_path
    from metrics import bootstrap_auc, operating_point, roc_auc
    from sliceclf import SliceClassifier

import config  # noqa: E402 - repo root, importable under both invocations
from logging_setup import setup_logging

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = os.path.join(config.MODELS_DIR, "examclf")
DEFAULT_BANK_DIR = os.path.join(config.CURATED_DATA_DIR, "exam_bank")


class ExamBagDataset(Dataset):
    """One item = one exam: ``k`` slices sampled from its bank rows, plus its label.

    Sampled without replacement when the exam holds at least ``k`` slices, with
    replacement otherwise (the shortest exam here has 24). Rows are read in sorted
    order for a monotonic memmap access pattern; the bag itself is order-invariant
    since it is aggregated by max.
    """

    def __init__(self, bank, exam_ids, k=16, seed=0, augment=False):
        self.bank = bank
        self.exam_ids = list(exam_ids)
        self.k = k
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.exam_ids)

    def __getitem__(self, i):
        exam = self.exam_ids[i]
        rows = self.bank.rows_for(exam)
        replace = len(rows) < self.k
        picked = np.sort(self.rng.choice(rows, size=self.k, replace=replace))
        imgs = torch.from_numpy(np.asarray(self.bank.volumes[picked], dtype=np.float32))
        imgs = imgs[:, None]  # (k, 1, S, S)

        if self.augment:
            for j in range(imgs.shape[0]):
                img = imgs[j]
                if self.rng.random() < 0.5:
                    img = torch.flip(img, dims=[2])
                if self.rng.random() < 0.5:
                    img = torch.flip(img, dims=[1])
                imgs[j] = _gamma_jitter(img, self.rng)

        label = torch.tensor([self.bank.exam_label[exam]], dtype=torch.float32)
        return imgs, label

    def pos_weight(self):
        """``n_negative / n_positive`` over the exams this dataset draws bags from."""
        labels = np.array([self.bank.exam_label[e] for e in self.exam_ids])
        n_pos = float((labels > 0).sum())
        return float((labels.size - n_pos) / n_pos) if n_pos else 1.0


def bag_logit(logits, k=1, relative=False):
    """Collapse a bag's per-slice logits ``(B, K)`` into one logit per bag ``(B,)``.

    ``k=1`` is the plain max this head started with, and stays the default so earlier
    measurements reproduce. A larger ``k`` averages the ``k`` highest-scoring slices
    instead, which is the standard MIL generalisation of max (``k=1``) and mean
    (``k=len``): a pure max lets one over-confident slice decide a whole exam, and
    with 68 slices per exam and 107 positive bags there are plenty of chances for one
    to be wrong. ``k`` is clamped to the bag's own size, so a short exam is not padded
    with slices it does not have.

    ``relative`` subtracts the bag's own median before that, which is not a tweak but
    the fix for a measured design error (plan.md §4.9). The quantity that separates a
    lesion-bearing slice from its neighbours is *relative to the exam it belongs to*:
    the 99th percentile of a slice ranks painted slices at 0.736 AUC **within** an
    exam and 0.532 **pooled across** exams -- chance. Absolute per-slice scores are
    therefore not comparable between patients, yet comparing them across patients is
    exactly what a patient-level ROC asks of them. Subtracting the bag's median asks
    instead "how far does this exam's most suspicious slice stand out from the rest of
    *this* exam", which is the form the signal actually takes. The median, not the
    mean: a bag is mostly unremarkable slices, and the few that are not should move
    the reference point as little as possible. ``torch.median`` on an even-length bag
    returns the lower of the two middle values rather than their average -- a detail
    worth knowing since it shifts every score this head emits, and one a test pins.
    """
    if relative:
        logits = logits - logits.median(dim=1, keepdim=True).values
    k = max(1, min(k, logits.shape[1]))
    return logits.topk(k, dim=1).values.mean(dim=1)


class WarmStartSliceDataset(Dataset):
    """One item = one slice, labelled by whether an annotated box paints it.

    This is a **different target** from the exam label the MIL head optimises: it says
    "a lesion is visible on this slice", benign or cancer alike, and it only exists for
    the annotated series of the corpus (1 401 painted slices out of 59 529). It
    initialises the encoder; it scores nothing. See :func:`warm_start_encoder`.

    Negatives are resampled every epoch rather than all taken: 2.35 % of the bank's
    slices are painted, so an epoch over every negative would cost four MIL epochs for
    a phase whose only job is to teach the encoder what a lesion looks like. Across
    epochs the model still sees many distinct negatives.
    """

    def __init__(self, bank, exam_ids, neg_per_pos=8, seed=0, augment=True, exclude_band=0):
        painted_rows, negative_rows = [], []
        for exam in exam_ids:
            rows = bank.rows_for(exam)                      # already in slice order
            painted = bank.has_lesion[rows]
            painted_rows.append(rows[painted])
            if exclude_band and painted.any():
                # Distance in slices to the nearest painted slice of this exam. The
                # annotation marks one central slice per lesion and preprocessing
                # paints +-2 around it, but a tomosynthesis mass stays visible well
                # past that: the slices just outside the painted band show the same
                # lesion while being labelled "no lesion". Training on them asks the
                # encoder to separate two nearly identical images by their label,
                # which is how a loss lands on the constant-predictor floor. Dropping
                # a band around each lesion turns those from wrong negatives into no
                # example at all -- the honest thing to do with a slice whose label
                # this corpus does not actually know.
                depth = np.arange(len(rows))
                distance = np.abs(depth[:, None] - depth[painted][None, :]).min(axis=1)
                negative_rows.append(rows[~painted & (distance > exclude_band)])
            else:
                negative_rows.append(rows[~painted])

        self.bank = bank
        self.painted_rows = np.concatenate(painted_rows) if painted_rows else np.array([], int)
        self.unpainted_rows = np.concatenate(negative_rows) if negative_rows else np.array([], int)
        self.exclude_band = exclude_band
        self.neg_per_pos = neg_per_pos
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.resample()

    def resample(self):
        """Draw a fresh negative sample. Called once per epoch, by the trainer."""
        n_neg = min(len(self.unpainted_rows), self.neg_per_pos * len(self.painted_rows))
        negatives = self.rng.choice(self.unpainted_rows, size=n_neg, replace=False)
        self.rows = np.sort(np.concatenate([self.painted_rows, negatives]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = int(self.rows[i])
        img = torch.from_numpy(np.asarray(self.bank.volumes[row], dtype=np.float32))[None]
        if self.augment:
            if self.rng.random() < 0.5:
                img = torch.flip(img, dims=[2])
            if self.rng.random() < 0.5:
                img = torch.flip(img, dims=[1])
            img = _gamma_jitter(img, self.rng)
        label = torch.tensor([float(self.bank.has_lesion[row])], dtype=torch.float32)
        return img, label

    def pos_weight(self):
        n_pos = float(len(self.painted_rows))
        return float((len(self.rows) - n_pos) / n_pos) if n_pos else 1.0


def warm_start_encoder(model, bank, train_exams, args, device, fold):
    """Train ``model`` on "is a lesion visible on this slice?" before the MIL phase.

    Why: plan.md §4.7 measured that the MIL loss does not fall in *any* of the five
    folds -- it oscillates at its starting level for all 25 epochs. A from-scratch
    encoder has to discover what a lesion looks like from 107 positive bags whose
    signal is ~5 slices deep out of 68, which is a very thin gradient to learn an
    image representation from. The bank already carries a denser label for that one
    question: ``has_lesion``, painted by the same boxes ``preprocess_dbt_exams``
    reads. Learning it first hands the MIL phase an encoder that already separates
    lesion from parenchyma, and leaves it the question it is actually for -- cancer or
    not, which a painted box does not answer (two thirds of painted slices here belong
    to *benign* lesions).

    **Only the fold's training exams are passed in.** The painted-slice label comes
    from the same annotations as the exam label, so warming up on a held-out patient's
    slices would leak that patient's answer into the encoder that then scores them.
    That is the one property of this function worth a test.
    """
    dataset = WarmStartSliceDataset(bank, train_exams, neg_per_pos=args.warm_start_neg_per_pos,
                                    seed=args.seed + fold, augment=not args.no_augment,
                                    exclude_band=args.warm_start_exclude_band)
    if not len(dataset.painted_rows):
        log.warning(f"  fold {fold}: no painted slice in the training exams, warm start skipped")
        return model

    loader = DataLoader(dataset, batch_size=args.warm_start_batch_size, shuffle=True,
                        num_workers=args.num_workers)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([dataset.pos_weight()], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    for epoch in range(1, args.warm_start_epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for imgs, target in loader:
            imgs, target = imgs.to(device), target.to(device)
            loss = criterion(model(imgs), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item() * imgs.shape[0]
            seen += imgs.shape[0]
        log.info(f"  fold {fold} warm-start {epoch}/{args.warm_start_epochs} | "
                 f"loss {running / max(1, seen):.4f} | "
                 f"{len(dataset.painted_rows)} painted / {len(dataset)} slices")
        dataset.resample()
    return model


@torch.no_grad()
def slice_lesion_auc(model, bank, exam_ids, device, batch_size=64):
    """Held-out AUC of "does a box paint this slice?" -- a **diagnostic**, not a score.

    Read before believing a negative exam-level result. The exam-level AUC alone
    cannot separate two very different failures: an encoder that never learned what a
    lesion looks like, and an encoder that did but whose bag aggregation throws the
    information away. This says which, by asking the encoder the *slice* question on
    held-out slices, where ``has_lesion`` gives a ground truth.

    Nothing is ever selected on it -- no checkpoint, no epoch, no hyper-parameter. It
    is recorded next to the result the way a temperature is recorded next to an
    experiment. ``None`` when the held-out exams carry no painted slice at all (an AUC
    needs both classes).
    """
    model.eval()
    probs, targets = [], []
    for exam in exam_ids:
        rows = bank.rows_for(exam)
        for lo in range(0, len(rows), batch_size):
            chunk = rows[lo:lo + batch_size]
            imgs = torch.from_numpy(
                np.asarray(bank.volumes[chunk], dtype=np.float32))[:, None].to(device)
            probs.append(torch.sigmoid(model(imgs))[:, 0].cpu().numpy())
            targets.append(bank.has_lesion[chunk].astype(np.float64))
    y = np.concatenate(targets)
    if y.min() == y.max():
        return None
    return float(roc_auc(y, np.concatenate(probs)))


@torch.no_grad()
def exam_scores(model, device, bank, exam_ids, batch_size=64, top_k=1, relative=False):
    """Cancer score per exam, aggregated over **all** of that exam's slices.

    Unlike training, evaluation is not sampled: with 24-114 slices per exam this is
    cheap, and a held-out patient's score should not depend on which slices luck
    happened to draw. ``top_k`` and ``relative`` match the training aggregation (see
    :func:`bag_logit`) -- optimising one aggregation and scoring with another would
    make the two disagree about what the model was asked to learn.

    With ``relative``, the returned value is a sigmoid of a *difference* of logits, so
    it is a ranking score rather than a calibrated probability: 0.5 no longer means
    "as likely as not", it means "this exam's top slice equals its own median". The
    threshold to serve is the one P1 asks for -- chosen on validation for a target
    sensitivity -- not 0.5.
    """
    model.eval()
    scores, labels = {}, {}
    for exam in exam_ids:
        rows = bank.rows_for(exam)
        logits = np.zeros(len(rows), dtype=np.float64)
        for lo in range(0, len(rows), batch_size):
            chunk = rows[lo:lo + batch_size]
            imgs = torch.from_numpy(
                np.asarray(bank.volumes[chunk], dtype=np.float32))[:, None].to(device)
            logits[lo:lo + batch_size] = model(imgs)[:, 0].cpu().numpy()
        if relative:
            logits = logits - np.median(logits)
        k = max(1, min(top_k, logits.size))
        scores[exam] = float(1.0 / (1.0 + np.exp(-np.sort(logits)[-k:].mean())))
        labels[exam] = bank.exam_label[exam]
    return scores, labels


def pool_to_patient(exam_score, exam_label, exam_patient):
    """Collapse exam-level scores/labels to one per patient, both by max.

    A patient is a cancer patient if any of their exams is (the same rule
    ``TransformData.preprocess_dbt_with_boxes`` applies to a mixed series' boxes);
    symmetrically, the patient's score is the most suspicious view found -- one
    positive view is enough to flag the patient, not something an average should
    dilute.
    """
    patient_score, patient_label = {}, {}
    for exam, score in exam_score.items():
        patient = exam_patient[exam]
        patient_score[patient] = max(patient_score.get(patient, -1.0), score)
        patient_label[patient] = max(patient_label.get(patient, 0), exam_label[exam])
    return patient_score, patient_label


def train_one_fold(train_exams, heldout_exams, bank, args, device, fold):
    """Train from scratch on ``train_exams``, then score the held-out exams."""
    train_set = ExamBagDataset(bank, train_exams, k=args.bag_size,
                               seed=args.seed + fold, augment=not args.no_augment)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        drop_last=len(train_set) > args.batch_size)

    torch.manual_seed(args.seed + fold)
    model = SliceClassifier(base=args.base_channels, dropout=args.dropout).to(device)
    diagnostics = {}
    if args.warm_start_epochs:
        warm_start_encoder(model, bank, train_exams, args, device, fold)
        diagnostics["slice_auc_after_warm_start"] = slice_lesion_auc(
            model, bank, heldout_exams, device, args.batch_size)
        log.info(f"  fold {fold} diagnostic: held-out slice-lesion AUC after warm start "
                 f"{diagnostics['slice_auc_after_warm_start']}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([train_set.pos_weight()], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for imgs, target in loader:
            b, k = imgs.shape[0], imgs.shape[1]
            imgs = imgs.view(b * k, 1, imgs.shape[3], imgs.shape[4]).to(device)
            target = target.to(device)
            logits = model(imgs).view(b, k)
            loss = criterion(
                bag_logit(logits, args.top_k, args.bag_relative).unsqueeze(1), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item() * b
            seen += b
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            log.info(f"  fold {fold} epoch {epoch:3d}/{args.epochs} | "
                     f"loss {running / max(1, seen):.4f}")

    diagnostics["slice_auc_after_mil"] = slice_lesion_auc(
        model, bank, heldout_exams, device, args.batch_size)
    log.info(f"  fold {fold} diagnostic: held-out slice-lesion AUC after MIL "
             f"{diagnostics['slice_auc_after_mil']}")

    scores, labels = exam_scores(model, device, bank, heldout_exams, args.batch_size,
                                 top_k=args.top_k, relative=args.bag_relative)
    return scores, labels, model, len(train_set), len(heldout_exams), diagnostics


def cross_validate(args):
    """Pool out-of-fold patient predictions, then report what they support."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    log.info(f"Device: {device}")

    all_paths = sorted(glob(os.path.join(args.data_dir, "*.npz")))
    if not all_paths:
        raise FileNotFoundError(f"No .npz files found in {args.data_dir!r}.")
    build_exam_bank(all_paths, args.bank_dir, image_size=args.image_size)
    bank = ExamBank(args.bank_dir)
    log.info(f"Bank: {len(bank.unique_exams())} exams, {len(bank.volumes)} slices, "
             f"{len(set(bank.case_id))} patients")

    folds = list(kfold_by_patient(args.data_dir, n_splits=args.folds, seed=args.seed))

    exam_patient = {}
    pooled_scores, pooled_labels, pooled_fold = {}, {}, {}
    fold_diagnostics = []
    for i, (train_paths, heldout_paths) in enumerate(folds):
        train_exams = [exam_id_for_path(p) for p in train_paths]
        heldout_exams = [exam_id_for_path(p) for p in heldout_paths]
        for path, exam in zip(train_paths + heldout_paths, train_exams + heldout_exams):
            exam_patient[exam] = default_patient_key(path)

        scores, labels, model, n_train, n_held, diagnostics = train_one_fold(
            train_exams, heldout_exams, bank, args, device, i)
        log.info(f"fold {i}: {n_train} training exams (bag size {args.bag_size}), "
                 f"{n_held} held-out exams")
        pooled_scores.update(scores)
        pooled_labels.update(labels)
        pooled_fold.update({e: i for e in scores})
        fold_diagnostics.append(diagnostics)
        if i == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, "fold0.pt"))

    patient_score, patient_label = pool_to_patient(pooled_scores, pooled_labels, exam_patient)
    patients = sorted(patient_score)
    y = np.array([patient_label[p] for p in patients])
    s = np.array([patient_score[p] for p in patients])

    report = {
        "task": "cancer / no-cancer on a full DBT exam (step 1, exam-level decision head)",
        "protocol": {
            "folds": args.folds,
            "selection": "no checkpoint or epoch chosen on held-out patients; "
                         f"{args.epochs} epochs fixed in advance, last epoch scored",
            "aggregation": (f"mean of the top {args.top_k} slice probabilities per exam"
                            if args.top_k > 1 else "max slice probability per exam")
                           + ", then max exam score per patient "
                             "(one suspicious slice/view is enough)",
            "bag_size": args.bag_size, "epochs": args.epochs, "top_k": args.top_k,
            "bag_relative": args.bag_relative,
            "warm_start_epochs": args.warm_start_epochs,
            "warm_start_neg_per_pos": args.warm_start_neg_per_pos,
            "warm_start_exclude_band": args.warm_start_exclude_band,
            "warm_start_target": (
                "slice-level has_lesion (benign or cancer box), fold-training exams "
                "only -- encoder initialisation, never scored"
                if args.warm_start_epochs else None),
            "image_size": args.image_size, "base_channels": args.base_channels,
            "lr": args.lr, "seed": args.seed,
        },
        "corpus": {
            "exams": len(bank.unique_exams()),
            "slices": int(len(bank.volumes)),
            "patients": len(patients),
            "cancer_patients": int((y > 0).sum()),
            "noncancer_patients": int((y == 0).sum()),
        },
        "auc_patient": bootstrap_auc(y, s, n_resamples=args.bootstrap, seed=args.seed),
        "operating_point_0.5": operating_point(y, s, 0.5),
        "always_negative_accuracy": float((y == 0).mean()),
        # Diagnostics, never selection criteria: held-out AUC of the *slice* question
        # ("does a box paint this slice?"), per fold, before and after the MIL phase.
        # They say whether a negative exam-level result is an encoder that learned
        # nothing or an aggregation that discarded what it learned.
        "diagnostics_slice_lesion_auc": fold_diagnostics,
    }
    with open(os.path.join(args.output_dir, "cv_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.output_dir, "cv_predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "label", "score", "fold"])
        for p in patients:
            exams_of_p = [e for e, pat in exam_patient.items() if pat == p and e in pooled_fold]
            fold_of_p = pooled_fold[exams_of_p[0]] if exams_of_p else -1
            writer.writerow([p, patient_label[p], f"{patient_score[p]:.6f}", fold_of_p])

    auc = report["auc_patient"]
    point = report["operating_point_0.5"]
    log.info(f"Patient ROC-AUC {auc['auc']:.3f} [{auc['lo']:.3f}-{auc['hi']:.3f}] "
             f"on {auc['n']} patients ({auc['n_positive']} cancers)")
    log.info(f"At 0.5: sensitivity {point['sensitivity']:.3f}, "
             f"specificity {point['specificity']:.3f}, PPV {point['ppv']:.3f} "
             f"at prevalence {point['prevalence']:.3f}; "
             f"accuracy {point['accuracy']:.3f} against "
             f"{report['always_negative_accuracy']:.3f} for always-negative")
    return report


def _smoke_report(args):
    """Run the loop end to end on random volumes, to check the plumbing, not a model."""
    import tempfile

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.folds * 2):
            label = i % 2
            depth = 6
            mask = np.zeros((depth, 32, 32), dtype=np.uint8)
            if label:
                mask[2, 10:20, 10:20] = 1
            np.savez_compressed(
                os.path.join(tmp, f"P{i:02d}-s0.npz"),
                volume=rng.standard_normal((depth, 32, 32)).astype(np.float16),
                mask=mask, crop_offset=np.zeros(3, dtype=np.int32),
                case_id=np.asarray(f"P{i:02d}"),
                label=np.asarray(label, dtype=np.uint8))
        args.data_dir = tmp
        args.bank_dir = os.path.join(tmp, "bank")
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, "smoke_test")
        args.epochs = min(args.epochs, 2)
        args.warm_start_epochs = min(args.warm_start_epochs, 1)
        args.bag_size = min(args.bag_size, 4)
        args.image_size = 32
        args.bootstrap = min(args.bootstrap, 200)
        return cross_validate(args)


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default=config.DBT_EXAMS_PREPROCESSED_DIR)
    p.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25,
                   help="Fixed in advance: no epoch is chosen on held-out patients.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Number of exams (bags) per step; each contributes --bag-size slices.")
    p.add_argument("--bag-size", type=int, default=16,
                   help="Slices sampled per exam per step, aggregated by max.")
    p.add_argument("--bag-relative", action="store_true",
                   help="Score a bag by how far its top slice stands out from its own "
                        "median, not by an absolute logit: per-slice scores are not "
                        "comparable across exams (plan.md 4.9).")
    p.add_argument("--top-k", type=int, default=1,
                   help="Slices averaged to score a bag: 1 is the max this head "
                        "started with, >1 averages the k most suspicious slices.")
    p.add_argument("--warm-start-epochs", type=int, default=0,
                   help="Epochs of slice-level 'is a lesion visible here' training "
                        "before the MIL phase, on the fold's training exams only. "
                        "0 keeps the from-scratch encoder earlier runs measured.")
    p.add_argument("--warm-start-neg-per-pos", type=int, default=8,
                   help="Unpainted slices sampled per painted one, per warm-start epoch.")
    p.add_argument("--warm-start-exclude-band", type=int, default=0,
                   help="Drop unpainted slices within this many slices of a painted "
                        "one from the negatives: they show the same lesion, which the "
                        "+-2 annotation band does not cover. 0 keeps every negative.")
    p.add_argument("--warm-start-batch-size", type=int, default=32,
                   help="Slices per step during the warm start (one slice, not one bag).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run the whole loop on random volumes, without the dataset.")
    return p


def main(argv=None):
    setup_logging(logfile="examclf.log")
    args = build_arg_parser().parse_args(argv)
    return _smoke_report(args) if args.smoke_test else cross_validate(args)


if __name__ == "__main__":
    main()
