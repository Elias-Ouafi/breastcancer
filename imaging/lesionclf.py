"""Benign vs cancer on a DBT lesion: step 2, read off the image instead of a form.

    python -m imaging.lesionclf --data-dir data/preprocessed_data/dbt --folds 5

What this is, and what it is not
--------------------------------
Step 2 of the product characterises a lesion that has already been found: the
tabular model does it from thirty measurements a cytopathologist produced. This does
the same job from the image, on the corpus that ``preprocess_dbt_with_boxes`` writes
-- volumes cropped around an annotated box, each carrying its ``lesion_class``.

So the crop *presupposes the lesion's location*, which is exactly step 2's premise
and exactly why this is not a detection model. It cannot be read as one, and a
number from it says nothing about the sensitivity or specificity of finding a cancer
in a screening exam. That is the separate exam-level head, which needs full frames
and normal exams (plan.md).

Why cross-validation rather than one test split
-----------------------------------------------
The corpus holds 130 patients, 55 of them with a cancer. A single 15 % test split
leaves about 20 patients and 8 cancers: the bootstrap interval on an AUC that small
covers nearly everything, so the run would produce a number no one could act on.
Pooling out-of-fold predictions evaluates every patient exactly once, with a model
that never saw them. It buys coverage, not independence -- the folds share a corpus,
a preprocessing and one hyper-parameter choice -- so it is reported as what it is.

No checkpoint is selected inside a fold, and no epoch is chosen by watching the
held-out patients: the epoch budget is fixed before the run and the last epoch is
the one scored. Selecting on the held-out fold is the standard way a
cross-validation number ends up optimistic, and this project has already published
one metric that was measured before its split (``AnalyzeData``, plan.md).

The encoder is ``sliceclf.SliceClassifier``, reused rather than re-derived: the same
GroupNorm stack, the same avg+max pooled head, for the reason documented there
(BatchNorm statistics never converge on batches this small). It is trained from
scratch -- an ImageNet encoder was tried and rejected on evidence for this data
(plan.md section 4.1), and in any case the weights cannot be fetched from this
machine.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:  # allow both "python -m imaging.lesionclf" and direct execution
    from .dataset import _gamma_jitter, _random_affine, default_patient_key, kfold_by_patient
    from .metrics import bootstrap_auc, operating_point
    from .sliceclf import SliceClassifier
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dataset import _gamma_jitter, _random_affine, default_patient_key, kfold_by_patient
    from metrics import bootstrap_auc, operating_point
    from sliceclf import SliceClassifier

import config  # noqa: E402 - repo root, importable under both invocations
from logging_setup import setup_logging

log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = os.path.join(config.MODELS_DIR, "lesionclf")


def lesion_slices(path, image_size=224):
    """Every lesion-bearing slice of one ``.npz``, resized, plus its patient and label.

    Only slices the mask touches are kept. The volume is a crop around the box, so
    its other slices show the same region at another depth -- they carry the lesion's
    surroundings but not the lesion, and calling them benign or cancer would label
    tissue by what sits a few millimetres away.

    Returns ``(images, patient, label)`` with ``images`` shaped ``(n, size, size)``
    in float16: at ~7 lesion slices per series the whole corpus fits in memory, and
    paying the decompression once per run rather than once per epoch is what makes
    five folds cheap.
    """
    with np.load(path) as data:
        if "label" not in data.files:
            raise KeyError(
                f"{os.path.basename(path)} carries no `label`: this model needs the "
                "benign/cancer class, which preprocess_dbt_with_boxes stores only "
                "when the boxes CSV says so. Re-run the preprocessing.")
        volume = data["volume"]
        mask = data["mask"]
        label = int(data["label"])
        case_id = str(data["case_id"]) if "case_id" in data.files else None

    keep = np.nonzero(mask.reshape(mask.shape[0], -1).any(axis=1))[0]
    if keep.size == 0:
        return np.zeros((0, image_size, image_size), dtype=np.float16), case_id, label

    slices = torch.from_numpy(volume[keep].astype(np.float32))[:, None]
    slices = F.interpolate(slices, size=(image_size, image_size),
                           mode="bilinear", align_corners=False)
    patient = case_id or default_patient_key(path)
    return slices[:, 0].numpy().astype(np.float16), patient, label


class LesionSliceDataset(Dataset):
    """Lesion slices of the given ``.npz`` paths, as ``(image, class)`` pairs.

    ``augment`` reuses the gamma jitter and random affine of the segmentation
    pipeline. Unlike there, no mask has to follow the transform, but the affine is
    still applied to a lesion sitting mid-frame, so a large rotation would push it
    out: the defaults documented in ``dataset`` keep it inside.
    """

    def __init__(self, paths, image_size=224, augment=False, seed=0, preloaded=None):
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        chunks, patients, labels = [], [], []
        for path in paths:
            images, patient, label = (preloaded[path] if preloaded is not None
                                      else lesion_slices(path, image_size))
            if images.shape[0] == 0:
                continue
            chunks.append(images)
            patients.extend([patient] * images.shape[0])
            labels.extend([label] * images.shape[0])

        self.images = (np.concatenate(chunks) if chunks
                       else np.zeros((0, image_size, image_size), dtype=np.float16))
        self.patients = np.asarray(patients)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i].astype(np.float32))[None]
        if self.augment:
            img = _gamma_jitter(img, self.rng)
            img, _ = _random_affine(img, img.clone(), self.rng)
        return img, torch.tensor([self.labels[i]], dtype=torch.float32)

    def pos_weight(self):
        """``n_benign / n_cancer``, to stop the loss from simply predicting benign."""
        n_pos = float((self.labels > 0).sum())
        return float((self.labels.size - n_pos) / n_pos) if n_pos else 1.0


@torch.no_grad()
def patient_scores(model, device, dataset, batch_size=64):
    """Mean cancer probability per patient, over that patient's lesion slices.

    The mean, not the maximum: this is characterisation, where every view of the same
    lesion is evidence about the same question, and the brightest slice of a benign
    lesion is not a reason to call it a cancer. (For detection the argument runs the
    other way -- one suspicious slice is enough -- which is one more reason the two
    heads stay separate.)
    """
    model.eval()
    if len(dataset) == 0:
        return {}, {}

    probs = np.zeros(len(dataset), dtype=np.float64)
    for lo in range(0, len(dataset), batch_size):
        batch = torch.from_numpy(
            dataset.images[lo:lo + batch_size].astype(np.float32))[:, None].to(device)
        probs[lo:lo + batch_size] = torch.sigmoid(model(batch))[:, 0].cpu().numpy()

    scores, labels = {}, {}
    for patient in np.unique(dataset.patients):
        rows = dataset.patients == patient
        scores[patient] = float(probs[rows].mean())
        labels[patient] = int(dataset.labels[rows][0])
    return scores, labels


def train_one_fold(train_paths, heldout_paths, args, device, preloaded, fold):
    """Train from scratch on ``train_paths``, then score the held-out patients."""
    train_set = LesionSliceDataset(train_paths, args.image_size,
                                   augment=not args.no_augment, seed=args.seed + fold,
                                   preloaded=preloaded)
    heldout_set = LesionSliceDataset(heldout_paths, args.image_size,
                                     preloaded=preloaded)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=len(train_set) > args.batch_size)

    torch.manual_seed(args.seed + fold)
    model = SliceClassifier(base=args.base_channels, dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([train_set.pos_weight()], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for img, target in loader:
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(img), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item() * img.size(0)
            seen += img.size(0)
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            log.info(f"  fold {fold} epoch {epoch:3d}/{args.epochs} | "
                     f"loss {running / max(1, seen):.4f}")

    scores, labels = patient_scores(model, device, heldout_set, args.batch_size)
    return scores, labels, model, len(train_set), len(heldout_set)


def cross_validate(args):
    """Pool out-of-fold patient predictions, then report what they support."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    log.info(f"Device: {device}")

    folds = list(kfold_by_patient(args.data_dir, n_splits=args.folds, seed=args.seed))

    # Decompress once: every fold reads the same volumes, four times as training data
    # and once as held-out.
    preloaded = {}
    for path in sorted({p for train, held in folds for p in train + held}):
        preloaded[path] = lesion_slices(path, args.image_size)
    n_slices = sum(v[0].shape[0] for v in preloaded.values())
    log.info(f"{len(preloaded)} series, {n_slices} lesion slices, "
             f"{len({v[1] for v in preloaded.values()})} patients")

    pooled_scores, pooled_labels, pooled_fold = {}, {}, {}
    for i, (train_paths, heldout_paths) in enumerate(folds):
        scores, labels, model, n_train, n_held = train_one_fold(
            train_paths, heldout_paths, args, device, preloaded, i)
        log.info(f"fold {i}: {n_train} training slices, {n_held} held-out slices, "
                 f"{len(scores)} held-out patients")
        pooled_scores.update(scores)
        pooled_labels.update(labels)
        pooled_fold.update({p: i for p in scores})
        if i == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, "fold0.pt"))

    patients = sorted(pooled_scores)
    y = np.array([pooled_labels[p] for p in patients])
    s = np.array([pooled_scores[p] for p in patients])

    report = {
        "task": "benign vs cancer on an annotated DBT lesion crop (step 2, image side)",
        "not_a_detection_model": (
            "the volume is cropped around the annotated box, so the lesion's location "
            "is given; nothing here measures finding one in a screening exam"),
        "protocol": {
            "folds": args.folds,
            "selection": "no checkpoint or epoch chosen on held-out patients; "
                         f"{args.epochs} epochs fixed in advance, last epoch scored",
            "aggregation": "mean cancer probability over a patient's lesion slices",
            "epochs": args.epochs, "image_size": args.image_size,
            "base_channels": args.base_channels, "lr": args.lr, "seed": args.seed,
        },
        "corpus": {
            "series": len(preloaded), "lesion_slices": int(n_slices),
            "patients": len(patients),
            "cancer_patients": int((y > 0).sum()),
            "benign_patients": int((y == 0).sum()),
        },
        "auc_patient": bootstrap_auc(y, s, n_resamples=args.bootstrap, seed=args.seed),
        "operating_point_0.5": operating_point(y, s, 0.5),
        "always_benign_accuracy": float((y == 0).mean()),
    }
    with open(os.path.join(args.output_dir, "cv_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.output_dir, "cv_predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "label", "score", "fold"])
        for p in patients:
            writer.writerow([p, pooled_labels[p], f"{pooled_scores[p]:.6f}",
                             pooled_fold[p]])

    auc = report["auc_patient"]
    point = report["operating_point_0.5"]
    log.info(f"Patient ROC-AUC {auc['auc']:.3f} [{auc['lo']:.3f}-{auc['hi']:.3f}] "
             f"on {auc['n']} patients ({auc['n_positive']} cancers)")
    log.info(f"At 0.5: sensitivity {point['sensitivity']:.3f}, "
             f"specificity {point['specificity']:.3f}, PPV {point['ppv']:.3f} "
             f"at prevalence {point['prevalence']:.3f}; "
             f"accuracy {point['accuracy']:.3f} against "
             f"{report['always_benign_accuracy']:.3f} for always-benign")
    return report


def _smoke_report(args):
    """Run the loop end to end on random volumes, to check the plumbing, not a model."""
    import tempfile

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.folds * 2):
            label = i % 2
            depth = 4
            mask = np.zeros((depth, 64, 64), dtype=np.uint8)
            mask[1:3, 20:40, 20:40] = 1
            np.savez_compressed(
                os.path.join(tmp, f"P{i:02d}-s0.npz"),
                volume=rng.standard_normal((depth, 64, 64)).astype(np.float16),
                mask=mask, crop_offset=np.zeros(3, dtype=np.int32),
                case_id=np.asarray(f"P{i:02d}"),
                label=np.asarray(label, dtype=np.uint8))
        args.data_dir = tmp
        args.output_dir = os.path.join(DEFAULT_OUTPUT_DIR, "smoke_test")
        args.epochs = min(args.epochs, 2)
        args.bootstrap = min(args.bootstrap, 200)
        return cross_validate(args)


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default=config.DBT_PREPROCESSED_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25,
                   help="Fixed in advance: no epoch is chosen on held-out patients.")
    p.add_argument("--batch-size", type=int, default=32)
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
    setup_logging(logfile="lesionclf.log")
    args = build_arg_parser().parse_args(argv)
    return _smoke_report(args) if args.smoke_test else cross_validate(args)


if __name__ == "__main__":
    main()
