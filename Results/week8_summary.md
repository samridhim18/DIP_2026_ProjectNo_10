# Week 8 — Analysis and Improvement of Brain MRI Tumour Localization
_Generated: 2026-11-08_

## 1. Objective
Analyse the Week 7 multi-patient localization results, identify failure patterns,
and investigate one simple classical-image-processing improvement.
All numbers are computed from actual masks and CSV files, not invented.

---

## 2. Existing Baseline Method

**Pipeline (Weeks 5–7, unchanged):**
1. Min-max normalisation of FLAIR slice
2. 3×3 median filter
3. Brain mask: Otsu → morphological open (r=3) → fill holes → largest CC → close (r=3)
4. Otsu threshold applied only to brain-masked pixels
5. Morphological open (r=2) + close (r=2) on candidate mask
6. Remove components < 50 px
7. **Localization rule:** select the **largest remaining component** by area — GT never used

Parameters: `DISK_R_BRAIN_OPEN=3  DISK_R_TUMOR_OPEN=2  DISK_R_TUMOR_CLOSE=2  MIN_COMPONENT_AREA=50`

---

## 3. Week 7 Testing Overview

**Data read from:** `results/week7/week7_per_patient.csv`

| Stat | Value |
|------|-------|
| Patients tested | 5 (dev=1, eval=4) |
| Slices tested | 5 (one per patient) |
| Slice selection rule | Max-bright-content (threshold=0.3, GT not used) |
| Eval W6 Dice — mean | 0.7402 |
| Eval W6 Dice — median | 0.7610 |
| Eval W6 Dice — min | 0.4808 |
| Eval W6 Dice — max | 0.9581 |
| Eval W6 IoU — mean | 0.6168 |
| Eval W6 Precision — mean | 0.7199 |
| Eval W6 Recall — mean | 0.8623 |
| Best | BraTS20_Training_243  Dice=0.9581 |
| Worst | BraTS20_Training_167  Dice=0.4808 |
| Success (Dice ≥ 0.60) | 3 / 4 |
| Failure (Dice < 0.60) | 1 / 4 |

---

## 4. Connected-Component Analysis

**Data computed fresh from NIfTI files using the same pipeline parameters.**

| Patient | Group | Slice | GT (px) | #Retained CCs | Largest CC (px) | W6 Dice | Reliability |
|---------|-------|-------|---------|---------------|-----------------|---------|-------------|
| BraTS20_Training_001 | dev | 67 | 5048 | 2 | 4126 | 0.8912 | Good |
| BraTS20_Training_035 | eval | 61 | 2274 | 3 | 1674 | 0.7979 | Good |
| BraTS20_Training_167 | eval | 85 | 797 | 8 | 2518 | 0.4808 | Fragmented |
| BraTS20_Training_243 | eval | 66 | 3410 | 2 | 3368 | 0.9581 | Good |
| BraTS20_Training_285 | eval | 72 | 1156 | 13 | 1413 | 0.7240 | Fragmented |

**Component-level detail saved in:** `week8_component_analysis.csv`

**Observation:** When the number of retained connected components is small (≤ 3),
the largest-area rule reliably identifies the tumour. When many components are
retained (8 or 13), the largest component is not necessarily the tumour — it may
be a bright non-tumour region, or the tumour has been fragmented into small pieces.

---

## 5. Visual Failure Analysis

### Successful Example — BraTS20_Training_243 (Dice=0.9581)
- 2 retained components. The largest (3368 px) aligns almost perfectly with GT (3410 px).
- Largest-area rule works: tumour is the dominant bright region in FLAIR.

### Partial Failure — BraTS20_Training_035 (Dice=0.7979)
- 3 retained components. Largest component selected correctly.
- Precision limited by adjacent bright non-tumour tissue in the same region.
- **Visible error:** boundary FP — predicted mask extends slightly beyond GT.

### Failure — BraTS20_Training_167 (Dice=0.4808)
- 8 retained components. GT tumour area is small (797 px).
- The largest detected component (2518 px) is primarily **non-tumour bright tissue**
  (TP=797, FP=1721). The tumour is not the largest bright region in this slice.
- **Root cause:** The largest-area rule fails when non-tumour structures
  (e.g., white matter lesions, CSF-adjacent regions) are larger than the tumour.

### Fragmented Prediction — BraTS20_Training_285 (Dice=0.7240)
- 13 retained components. Many small fragments of the tumour exist.
- The largest selected component (1413 px) partially covers the tumour (TP=930)
  with some FP (483 px).
- **Root cause:** Morphological cleanup (open r=2) erodes thin connections
  within the tumour, splitting it. Closing r=2 is insufficient to rejoin them.

---

## 6. Improvement Investigated: Dilation-Based Component Merging

**Method:** Dilate all retained Otsu pixels by radius r=5 to bridge gaps ≤ 10 px.
Re-label merged regions. Select the largest merged region (area rule, no GT).
Intersect with the original Otsu mask to avoid adding new pixels.

**Motivation from failure analysis:** P285 had 13 components because noise created
thin dark gaps inside the tumour. Dilation r=5 was expected to bridge those gaps.

---

## 7. Original vs Improved Comparison (all 5 slices)

| Patient | Group | W6 Dice | W8 Dice | ΔDice | W6 IoU | W8 IoU | W6 Prec | W8 Prec |
|---------|-------|---------|---------|-------|--------|--------|---------|---------|
| BraTS20_Training_001 | dev | 0.8912 | 0.8912 | +0.0 | 0.8038 | 0.8038 | 0.9908 | 0.9908 |
| BraTS20_Training_035 | eval | 0.7979 | 0.7776 | -0.0203 | 0.6637 | 0.6361 | 0.9409 | 0.8863 |
| BraTS20_Training_167 | eval | 0.4808 | 0.3467 | -0.1341 | 0.3165 | 0.2097 | 0.3165 | 0.2097 |
| BraTS20_Training_243 | eval | 0.9581 | 0.9581 | +0.0 | 0.9196 | 0.9196 | 0.9641 | 0.9641 |
| BraTS20_Training_285 | eval | 0.7240 | 0.4483 | -0.2757 | 0.5674 | 0.2889 | 0.6582 | 0.3107 |

**Eval-group averages (4 patients):**

| Method | Dice | IoU | Precision | Recall |
|--------|------|-----|-----------|--------|
| W6 Baseline | 0.7402 | 0.6168 | 0.7199 | 0.8623 |
| W8 Improved | 0.6327 | 0.5136 | 0.5927 | 0.8623 |

> [!IMPORTANT]
> **The dilation-merging improvement did NOT improve performance — it made it worse overall.**
> Mean Dice dropped from 0.7402 to 0.6327. The W6 baseline is the better method on this dataset.

**Why did dilation merging hurt?**

- **P285:** Dilation merged the tumour fragments as intended, but it also merged them
  with large nearby non-tumour bright areas. After intersection with the original mask,
  the selected region included 2063 FP pixels vs 483 for W6 — precision fell from 0.66 to 0.31.
- **P167:** The largest dilated region is still non-tumour tissue. Dilation made it
  even larger, increasing FP from 1721 to 3003.
- **P035:** Dilation slightly merged a small false-positive component into the main
  prediction, increasing FP from 99 to 202.
- **P001, P243:** No change — these already had ≤2 components; dilation had no effect.

---

## 8. Limitations

- One slice per patient is tested; full volumetric evaluation is not performed.
- The largest-area rule fails for small tumours where non-tumour tissue is larger.
- Dilation-based merging does not distinguish tumour from non-tumour bright regions —
  it simply expands existing predictions, which increases FP when nearby non-tumour
  regions are dilated into the selection.
- Recall is identical across both methods because the intersection step prevents
  adding new TP pixels beyond the Otsu prediction.
- A higher-performing improvement would require spatial prior knowledge (e.g.,
  restricting the search to central brain regions) or multi-modal information
  (T1CE is specifically designed to highlight tumour contrast) — both are
  permissible within classical image processing.

---

## 9. Conclusion

The W6 largest-area baseline achieved eval-group mean Dice = **0.7402** with
high recall (0.8623) but variable precision (0.7199). The standard deviation is
high (0.172), indicating patient-dependent performance.

The W8 dilation-merging improvement was investigated based on the P285 failure
case (13 fragmented components). The improvement **decreased** mean eval Dice
to 0.6327 — a result driven by increased false positives from merged non-tumour
bright regions in P167 and P285. The W6 baseline remains the better method.

**The correct conclusion from the data is: the dilation improvement is not recommended
for this dataset.** The primary failure mode (P167, small tumour) is not fragmentation
but misidentification — the tumour is not the largest bright region. Fixing this
requires a fundamentally different component-selection rule, not just merging.

---

## Assumption
- All 5 slices (dev + 4 eval) were tested using identical pipeline parameters.
- Dilation radius r=5 was not grid-searched using GT metrics on the test set.
- Results for P001 (dev) are reported separately and excluded from primary averages.
- The comparison is fair: both methods use the same Otsu prediction as input.
