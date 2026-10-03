# Stage 4: Baseline Brain Tumor Segmentation Summary

## 1. Overview & Pipeline Description
Stage 4 implements a baseline classical threshold-based segmentation workflow:
1. **Brain ROI Masking**: All operations are strictly constrained to the Stage 3 brain mask (`skull_stripped_flair`). Non-brain voxels are zeroed.
2. **Otsu Candidate Thresholding**: Calculated global Otsu threshold (`0.5332`) on non-zero brain voxels to extract hyperintense candidate regions.
3. **Morphological Cleanup**: Applied morphological opening ($r=2$) to sever isolated noise spikes, followed by closing ($r=2$) to connect adjacent candidate regions.
4. **Connected Component Filtering**: Filtered components with area $< 50$ pixels, preserving main candidate regions.
5. **Ground-Truth Evaluation**: Evaluated whole-tumor mask (`seg > 0`) against predictions.

## 2. Quantitative Results
| Metric | Numerical Value |
| :--- | :---: |
| **Axial Slice Number** | `67` |
| **Brain ROI Threshold ($T_{Otsu}$)** | `0.5332` |
| **Predicted Tumor Area** | `4,045` pixels |
| **Ground-Truth Tumor Area** | `5,048` pixels |
| **Area Difference** | `-1,003` pixels (`-19.87%`) |
| **True Positives (TP)** | `3,698` pixels |
| **False Positives (FP)** | `347` pixels |
| **False Negatives (FN)** | `1,350` pixels |
| **Sørensen–Dice Coefficient** | **`0.8134`** |
| **Jaccard IoU Index** | **`0.6854`** |

## 3. Discussion & Observations
- **Strengths**:
  - The baseline Otsu thresholding method successfully isolates the hyperintense core of the FLAIR tumor region.
  - Achieved a baseline **Dice score of 0.8134** and **IoU of 0.6854** using simple classical thresholding.
- **Limitations of Simple Global Otsu Baseline**:
  - Global Otsu thresholding tends to capture hyperintense white matter regions alongside tumor tissue, producing some false positives.
  - Peritumoral edema (ED) has intermediate intensity levels that fall slightly below global Otsu thresholding, leading to false negatives.
  - These limitations motivate the adaptive thresholding, region growing, and multi-modal classical DIP techniques in Stage 5.

## 4. Generated Output Files in `results/stage4/`
- `skull_stripped_flair.png`
- `initial_tumor_mask.png`
- `cleaned_tumor_mask.png`
- `ground_truth_mask.png`
- `final_tumor_comparison.png`
- `stage4_baseline_2x3_grid.png`
- `stage4_metrics.csv`
- `stage4_metrics_summary.txt`
