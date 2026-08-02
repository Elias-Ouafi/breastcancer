"""Slice-level lesion classifier: "does this slice contain a lesion?".

    python -m imaging.sliceclf --slice-bank slice_bank_p2 --epochs 20

Why a separate model
--------------------
The segmentation U-Net cannot pick the lesion's slice out of a full volume. This is
now quantified rather than suspected (``imaging.evaluate`` on the 28 held-out
patients): it raises a blob on **99.97 % of lesion-free slices**, ~222 false
components per volume. Its per-pixel confidence therefore carries no usable ranking
signal, which is exactly why taking the arg-max slice landed on a lesion 0/186 times.

Ranking is a different problem from outlining, so it gets its own model and its own
loss. Two things matter:

* **Every slice is a training example.** The segmentation model is trained on lesion
  slices plus a few sampled negatives (``--neg-per-pos``); it never sees the chest
  wall, the edge slices or the diffuse background enhancement that a real volume is
  mostly made of. Here the negatives *are* the task, so all ~31 k slices are used and
  the imbalance (15 % positive) is handled with ``pos_weight`` instead of by throwing
  negatives away.
* **The metric is the one that failed.** Not accuracy or AUC in the abstract but
  top-1: for each patient, is the highest-scoring slice of the whole volume actually
  lesion-bearing? That is the number the app depends on, and the one the checkpoint
  is selected on.

The encoder reuses ``DoubleConv`` (GroupNorm) from ``imaging.unet`` for the reason
documented there: with tiny foreground fractions and small batches, BatchNorm's
running statistics never converge and the model collapses in ``eval()`` mode.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:  # allow both "python -m imaging.sliceclf" and direct execution
    from .dataset import split_npz_by_patient
    from .slicebank import (INDEX_FILE, SliceBankDataset, build_slice_bank,
                            case_ids_for_paths)
    from .unet import DoubleConv
except ImportError:  # pragma: no cover
    from dataset import split_npz_by_patient
    from slicebank import INDEX_FILE, SliceBankDataset, build_slice_bank, case_ids_for_paths
    from unet import DoubleConv


class SliceClassifier(nn.Module):
    """Small GroupNorm CNN mapping a slice to one lesion-presence logit.

    Five downsampling stages take 256x256 to 8x8, then the map is pooled two ways and
    concatenated: average pooling captures how much of the slice looks lesion-like,
    max pooling captures the single most suspicious location. A lesion is small and
    focal, so average pooling alone washes it out; max pooling alone reacts to any
    bright speckle. Both together is what makes the score rank slices sensibly.
    """

    def __init__(self, base=16, dropout=0.3):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(1, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.enc5 = DoubleConv(base * 8, base * 8)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base * 16, base * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base * 4, 1),
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(self.pool(x))
        x = self.enc3(self.pool(x))
        x = self.enc4(self.pool(x))
        x = self.enc5(self.pool(x))
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        return self.head(torch.cat([avg, mx], dim=1))  # (N, 1) logits


class SliceLabelDataset(Dataset):
    """Wraps a :class:`SliceBankDataset` to yield ``(image, lesion_present)``.

    The label is derived from the mask *after* augmentation, not before: a random
    rotation or scale can push a small lesion out of frame, and calling such a slice
    positive would teach the model to fire on an image with nothing in it.
    """

    def __init__(self, bank):
        self.bank = bank

    def __len__(self):
        return len(self.bank)

    def __getitem__(self, i):
        img, msk = self.bank[i]
        return img, (msk.sum() > 0).float().reshape(1)


def _patient_slices(bank_dir, case_ids):
    """Return ``{case_id: (row indices, has_lesion flags)}`` in stored slice order.

    Volume order matters: the app scans a volume and takes an arg-max over it, so
    evaluation has to reproduce exactly that grouping.
    """
    with np.load(os.path.join(bank_dir, INDEX_FILE), allow_pickle=True) as idx:
        all_cases = idx["case_id"]
        has_lesion = idx["has_lesion"]
    wanted = set(case_ids)
    out = {}
    for i, case in enumerate(all_cases):
        if case in wanted:
            out.setdefault(case, [[], []])
            out[case][0].append(i)
            out[case][1].append(bool(has_lesion[i]))
    return {c: (np.array(v[0]), np.array(v[1])) for c, v in out.items()}


@torch.no_grad()
def score_slices(model, device, volumes, rows, batch_size=64):
    """Lesion probability for each bank row in ``rows``."""
    model.eval()
    scores = np.zeros(len(rows), dtype=np.float32)
    for lo in range(0, len(rows), batch_size):
        chunk = rows[lo:lo + batch_size]
        batch = np.asarray(volumes[chunk], dtype=np.float32)[:, None]
        logits = model(torch.from_numpy(batch).to(device))
        scores[lo:lo + batch_size] = torch.sigmoid(logits)[:, 0].cpu().numpy()
    return scores


def evaluate_ranking(model, device, bank_dir, case_ids, topk=3):
    """Top-1 / top-k slice selection and AUC over whole volumes.

    ``top1`` is the metric this model exists for: the fraction of patients whose
    highest-scoring slice really contains a lesion. The old segmentation-confidence
    ranking scored 0.000 here.
    """
    volumes = np.load(os.path.join(bank_dir, "volumes.npy"), mmap_mode="r")
    per_patient = _patient_slices(bank_dir, case_ids)

    hits1, hitsk, aucs, ranks = [], [], [], []
    for _case, (rows, has_lesion) in sorted(per_patient.items()):
        if not has_lesion.any():
            continue  # no lesion to find in this volume
        scores = score_slices(model, device, volumes, rows)
        order = np.argsort(-scores)
        hits1.append(bool(has_lesion[order[0]]))
        hitsk.append(bool(has_lesion[order[:topk]].any()))
        # Rank of the first true positive: 1 means the top slice was right.
        ranks.append(int(np.nonzero(has_lesion[order])[0][0]) + 1)

        pos, neg = scores[has_lesion], scores[~has_lesion]
        if pos.size and neg.size:
            # Mann-Whitney form of the AUC: P(a random lesion slice outranks a
            # random lesion-free one), ties counted as half.
            wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
            aucs.append(float(wins / (pos.size * neg.size)))

    return {
        "n_patients": len(hits1),
        "top1": float(np.mean(hits1)) if hits1 else float("nan"),
        f"top{topk}": float(np.mean(hitsk)) if hitsk else float("nan"),
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
        "auc": float(np.mean(aucs)) if aucs else float("nan"),
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_paths, val_paths, test_paths = split_npz_by_patient(
        args.data_dir, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    print(f"Patients -> train {len(train_paths)}, val {len(val_paths)}, test {len(test_paths)}")

    build_slice_bank(list(train_paths) + list(val_paths) + list(test_paths),
                     args.slice_bank, image_size=args.image_size)

    train_cases = case_ids_for_paths(train_paths)
    val_cases = case_ids_for_paths(val_paths)
    test_cases = case_ids_for_paths(test_paths)

    # positive_only=False and neg_per_pos=None together mean "every slice": the hard
    # negatives are the entire point of this model.
    bank = SliceBankDataset(args.slice_bank, case_ids=train_cases, positive_only=False,
                            neg_per_pos=None, seed=args.seed, augment=not args.no_augment)
    train_ds = SliceLabelDataset(bank)
    n_pos = int(bank.all_has_lesion[bank.index].sum())
    n_neg = len(bank) - n_pos
    print(f"Training slices: {len(bank)} ({n_pos} avec lésion, {n_neg} sans)")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers)

    model = SliceClassifier(base=args.base_channels, dropout=args.dropout).to(device)
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
    print(f"pos_weight = {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "sliceclf_best.pt")
    metrics_path = os.path.join(args.output_dir, "sliceclf_metrics.csv")
    best_top1 = -1.0

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_top1", "val_top3", "val_auc", "val_median_rank"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            running, seen = 0.0, 0
            for img, label in loader:
                img, label = img.to(device), label.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(model(img), label)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                running += loss.item() * img.size(0)
                seen += img.size(0)

            val = evaluate_ranking(model, device, args.slice_bank, val_cases)
            scheduler.step(val["top1"])
            print(f"Epoch {epoch:3d} | loss {running / max(1, seen):.4f} | "
                  f"val top1 {val['top1']:.3f} | top3 {val['top3']:.3f} | "
                  f"AUC {val['auc']:.3f} | rang médian {val['median_rank']:.0f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")
            writer.writerow([epoch, f"{running / max(1, seen):.6f}", f"{val['top1']:.6f}",
                             f"{val['top3']:.6f}", f"{val['auc']:.6f}", f"{val['median_rank']:.1f}"])
            f.flush()

            if val["top1"] > best_top1:
                best_top1 = val["top1"]
                torch.save(model.state_dict(), ckpt_path)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test = evaluate_ranking(model, device, args.slice_bank, test_cases)
    print(f"\nTest | top1 {test['top1']:.3f} | top3 {test['top3']:.3f} | "
          f"AUC {test['auc']:.3f} | rang médian {test['median_rank']:.0f} "
          f"({test['n_patients']} patients)")
    with open(os.path.join(args.output_dir, "sliceclf_test_metrics.json"), "w") as f:
        json.dump({"val_best_top1": best_top1, "test": test,
                   "base_channels": args.base_channels}, f, indent=2)
    print(f"Checkpoint: {ckpt_path}")
    return test


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train a slice-level lesion classifier.")
    p.add_argument("--data-dir", default="preprocessed_data_mri_p2")
    p.add_argument("--slice-bank", default="slice_bank_p2")
    p.add_argument("--output-dir", default="results_sliceclf")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="L2 regularisation. The negatives vastly outnumber the positives, "
                        "which makes memorising individual volumes an easy shortcut.")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42,
                   help="Must match imaging.train's seed so the splits are the same.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
