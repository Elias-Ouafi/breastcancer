"""Train a 2D U-Net to detect/localise breast lesions in preprocessed MRI slices.

Run from the repository root:

    python -m imaging.train --data-dir preprocessed_data --epochs 30

The target masks come from the annotation bounding boxes, so this learns lesion
*localisation* (Dice/IoU against the box), the objective chosen for the first
imaging brick. Metrics are written to ``results/segmentation_metrics.csv`` and the
best checkpoint to ``results/unet_best.pt``.

Use ``--smoke-test`` to validate the full forward/backward/eval loop on random
tensors, without any real data (handy right after installing PyTorch).
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:  # allow both "python -m imaging.train" and "python imaging/train.py"
    from .dataset import MRISliceDataset, split_npz_by_patient
    from .metrics import DiceBCELoss, dice_coeff, iou_score
    from .unet import UNet2D
except ImportError:  # pragma: no cover - fallback for direct script execution
    from dataset import MRISliceDataset, split_npz_by_patient
    from metrics import DiceBCELoss, dice_coeff, iou_score
    from unet import UNet2D


def evaluate(model, loader, device, threshold=0.5):
    """Return mean Dice and IoU over ``loader`` (empty loader -> NaN)."""
    if loader is None:
        return {"dice": float("nan"), "iou": float("nan")}
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for img, msk in loader:
            img, msk = img.to(device), msk.to(device)
            pred = (torch.sigmoid(model(img)) > threshold).float()
            dices.append(dice_coeff(pred, msk).item())
            ious.append(iou_score(pred, msk).item())
    if not dices:
        return {"dice": float("nan"), "iou": float("nan")}
    return {"dice": float(np.mean(dices)), "iou": float(np.mean(ious))}


def _make_loaders(args):
    """Build train/val/test DataLoaders from the preprocessed .npz files."""
    train_paths, val_paths, test_paths = split_npz_by_patient(
        args.data_dir, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(f"Cases -> train: {len(train_paths)}, val: {len(val_paths)}, test: {len(test_paths)}")

    train_ds = MRISliceDataset(train_paths, image_size=args.image_size, positive_only=args.positive_only)
    if not len(train_ds):
        raise RuntimeError("No training slices found. Check the data or --positive-only.")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = None
    if val_paths:
        val_ds = MRISliceDataset(val_paths, image_size=args.image_size, positive_only=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    test_loader = None
    if test_paths:
        test_ds = MRISliceDataset(test_paths, image_size=args.image_size, positive_only=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader, len(train_ds)


def _make_smoke_loaders(args):
    """Random-tensor loaders so the loop can be exercised without real data."""
    n, size = 12, args.image_size
    img = torch.randn(n, 1, size, size)
    msk = (torch.rand(n, 1, size, size) > 0.7).float()
    ds = TensorDataset(img, msk)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    val = DataLoader(ds, batch_size=args.batch_size)
    return loader, val, val, n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.smoke_test:
        train_loader, val_loader, test_loader, n_train = _make_smoke_loaders(args)
    else:
        train_loader, val_loader, test_loader, n_train = _make_loaders(args)

    model = UNet2D(base=args.base_channels).to(device)
    criterion = DiceBCELoss(bce_weight=args.bce_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "segmentation_metrics.csv")
    ckpt_path = os.path.join(args.output_dir, "unet_best.pt")
    best_dice = -1.0

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_dice", "val_iou"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for img, msk in train_loader:
                img, msk = img.to(device), msk.to(device)
                optimizer.zero_grad()
                loss = criterion(model(img), msk)
                loss.backward()
                optimizer.step()
                running += loss.item() * img.size(0)

            train_loss = running / max(1, n_train)
            val_metrics = evaluate(model, val_loader, device, threshold=args.threshold)
            print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
                  f"val Dice {val_metrics['dice']:.4f} | val IoU {val_metrics['iou']:.4f}")
            writer.writerow([epoch, f"{train_loss:.6f}",
                             f"{val_metrics['dice']:.6f}", f"{val_metrics['iou']:.6f}"])
            f.flush()

            if val_loader is not None and val_metrics["dice"] > best_dice:
                best_dice = val_metrics["dice"]
                torch.save(model.state_dict(), ckpt_path)

    # Final evaluation on the held-out test split, using the best checkpoint.
    if test_loader is not None:
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_metrics = evaluate(model, test_loader, device, threshold=args.threshold)
        print(f"Test  | Dice {test_metrics['dice']:.4f} | IoU {test_metrics['iou']:.4f}")
        with open(os.path.join(args.output_dir, "segmentation_test_metrics.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dice", "iou"])
            writer.writerow([f"{test_metrics['dice']:.6f}", f"{test_metrics['iou']:.6f}"])

    print(f"\nMetrics written to {metrics_path}")
    if os.path.exists(ckpt_path):
        print(f"Best checkpoint saved to {ckpt_path}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train a 2D U-Net for MRI lesion localisation.")
    p.add_argument("--data-dir", default="preprocessed_data", help="Folder of preprocessed .npz volumes.")
    p.add_argument("--output-dir", default="results", help="Where to write metrics and checkpoints.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=256, help="Square slice size (must be divisible by 16).")
    p.add_argument("--base-channels", type=int, default=32, help="U-Net width at the first level.")
    p.add_argument("--bce-weight", type=float, default=0.5, help="Weight of BCE vs Dice in the loss.")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for the binary mask.")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows).")
    p.add_argument("--positive-only", action="store_true", default=True,
                   help="Train only on slices containing lesion voxels.")
    p.add_argument("--all-slices", dest="positive_only", action="store_false",
                   help="Train on every slice, including lesion-free ones.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run the loop on random tensors (no real data) to validate the pipeline.")
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
