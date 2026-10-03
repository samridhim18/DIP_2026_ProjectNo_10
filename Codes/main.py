#!/usr/bin/env python3
"""
ECE501 Digital Image Processing Project
Brain MRI Tumor Segmentation and Quantitative Analysis Using Classical Image Processing

Stage 1: Dataset Loading, Volume Inspection, Intensity Normalization & Visualization Pipeline

Author: DIP Project Team
"""

import os
import sys

# Add project root directory to python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.loader import load_nifti_volume, get_volume_info
from src.preprocessing import (
    normalize_min_max,
    normalize_z_score,
    clip_percentiles,
    extract_axial_slice,
    find_best_slice
)
from src.visualization import plot_mri_and_mask
from src.utils import ensure_dir, print_volume_summary


def main():
    print("=" * 70)
    print(" ECE501 DIP Project: Stage 1 - BraTS MRI Data Loader & Visualizer")
    print("=" * 70)

    # 1. Define File Paths
    flair_path = os.path.join(PROJECT_ROOT, "BraTS20_Training_001_flair.nii")
    seg_path = os.path.join(PROJECT_ROOT, "BraTS20_Training_001_seg.nii")
    output_dir = ensure_dir(os.path.join(PROJECT_ROOT, "outputs"))

    # Verify input files exist
    if not os.path.exists(flair_path):
        raise FileNotFoundError(f"FLAIR MRI file missing at: {flair_path}")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation mask file missing at: {seg_path}")

    # 2. Load NIfTI 3D Volumes using nibabel
    print("\n[Step 1] Loading NIfTI volumes using nibabel...")
    flair_vol, flair_nifti = load_nifti_volume(flair_path)
    seg_vol, seg_nifti = load_nifti_volume(seg_path)

    # 3. Inspect 3D Volume Dimensions & Statistics
    print("\n[Step 2] Inspecting 3D volume dimensions and intensity distribution...")
    flair_info = get_volume_info(flair_vol, flair_nifti, flair_path, is_mask=False)
    seg_info = get_volume_info(seg_vol, seg_nifti, seg_path, is_mask=True)

    print_volume_summary("FLAIR MRI", flair_info)
    print_volume_summary("Ground-Truth Mask", seg_info)

    # 4. Normalize MRI Intensities (Classical DIP)
    print("\n[Step 3] Normalizing MRI intensities (Percentile Clipping + Min-Max [0, 1])...")
    clipped_flair = clip_percentiles(flair_vol, p_low=0.5, p_high=99.5)
    flair_norm = normalize_min_max(clipped_flair)
    flair_zscore = normalize_z_score(flair_vol, mask_zero_bg=True)

    print(f" Normalized Min Intensity : {flair_norm.min():.4f}")
    print(f" Normalized Max Intensity : {flair_norm.max():.4f}")
    print(f" Z-score Mean (brain area): {flair_zscore[flair_vol > 0].mean():.4f}")
    print(f" Z-score Std  (brain area): {flair_zscore[flair_vol > 0].std():.4f}")

    # 5. Extract 2D Axial Slices
    print("\n[Step 4] Extracting 2D axial slices...")
    # Find slice with maximum tumor presence
    peak_slice_idx, peak_tumor_voxels = find_best_slice(seg_vol)
    mid_slice_idx = flair_vol.shape[2] // 2

    print(f" Peak Tumor Slice Index  : {peak_slice_idx} (Tumor Voxel Count: {peak_tumor_voxels:,})")
    print(f" Middle Axial Slice Index: {mid_slice_idx}")

    # Extract 2D slices
    flair_slice_peak = extract_axial_slice(flair_norm, peak_slice_idx)
    seg_slice_peak = extract_axial_slice(seg_vol, peak_slice_idx)

    flair_slice_mid = extract_axial_slice(flair_norm, mid_slice_idx)
    seg_slice_mid = extract_axial_slice(seg_vol, mid_slice_idx)

    # 6. Display and Save Visualization Results
    print("\n[Step 5] Generating and saving visualization figures...")
    save_path_peak = os.path.join(output_dir, f"flair_seg_overlay_peak_slice_{peak_slice_idx}.png")
    save_path_mid = os.path.join(output_dir, f"flair_seg_overlay_mid_slice_{mid_slice_idx}.png")

    plot_mri_and_mask(
        mri_slice=flair_slice_peak,
        mask_slice=seg_slice_peak,
        slice_idx=peak_slice_idx,
        title_prefix="BraTS20_Training_001 (Peak Tumor Area)",
        alpha=0.5,
        save_path=save_path_peak
    )

    plot_mri_and_mask(
        mri_slice=flair_slice_mid,
        mask_slice=seg_slice_mid,
        slice_idx=mid_slice_idx,
        title_prefix="BraTS20_Training_001 (Middle Slice)",
        alpha=0.5,
        save_path=save_path_mid
    )

    print("\n" + "=" * 70)
    print(" Stage 1 Pipeline Completed Successfully!")
    print(f" Output visualizations saved in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
