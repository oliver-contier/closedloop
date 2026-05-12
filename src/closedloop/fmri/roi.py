"""ROI masking utilities for the targeted fMRI experiment analyses."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.masking import apply_mask


def apply_roi_mask(
    nifti_files: List[str],
    ffa_files: List[str],
) -> Tuple[np.ndarray, Any, Any]:
    """Apply an ROI mask (e.g. union of left and right FFA) to NIfTI inputs.

    Returns
    -------
    betas : np.ndarray of shape (n_samples, n_voxels)
    mask_img_resampled : Nifti1Image
    bg_img : Nifti1Image
    """
    if not ffa_files or len(ffa_files) not in [1, 2]:
        raise ValueError(
            f"Expected 1 or 2 ROI files, but got "
            f"{len(ffa_files) if ffa_files else 0}."
        )
    if not nifti_files:
        raise ValueError("No NIfTI files provided.")

    ref_img = load_img(nifti_files[0])
    bg_img = load_img(nifti_files[-1])

    if len(ffa_files) == 2:
        mask_img = math_img(
            "img1 + img2",
            img1=load_img(ffa_files[0]),
            img2=load_img(ffa_files[1]),
        )
    else:
        mask_img = load_img(ffa_files[0])

    mask_img_resampled = resample_to_img(mask_img, ref_img, interpolation="nearest")
    betas = apply_mask(nifti_files, mask_img_resampled)
    return betas, mask_img_resampled, bg_img
