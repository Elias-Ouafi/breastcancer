"""Imaging pipeline: 2D U-Net lesion localisation on preprocessed MRI volumes.

Consumes the ``.npz`` files (``volume`` + ``mask`` + ``crop_offset``) produced by
``TransformData.save_preprocessed``. Train with ``python -m imaging.train``.
"""

from .dataset import MRISliceDataset, split_npz_by_patient
from .metrics import DiceBCELoss, dice_coeff, iou_score
from .unet import UNet2D

__all__ = [
    "MRISliceDataset",
    "split_npz_by_patient",
    "DiceBCELoss",
    "dice_coeff",
    "iou_score",
    "UNet2D",
]
