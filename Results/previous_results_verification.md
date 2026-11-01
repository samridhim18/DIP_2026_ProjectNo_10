# Week 7 — Audit: Previous Results Verification
_Generated: 2026-11-01 13:40:29_

## Section C — Implementation Checks (before recomputing metrics)

### C1. Correct patient and axial slice loaded?
- FLAIR file: `BraTS20_Training_001_flair.nii`  → **YES**
- Slice index: `67` (axial dim, 0-based)  → **CONFIRMED by script constant**

### Loading NIfTI data ...
- FLAIR volume shape: (240, 240, 155)
- SEG volume shape:   (240, 240, 155)
- Extracted slice shape: (240, 240)
- GT mask shape:         (240, 240)
- GT tumour pixels (seg > 0): 5048

### C2. Prediction and ground-truth masks have matching dimensions?
- FLAIR slice: (240, 240)   GT mask: (240, 240)  → **MATCH**

### C3. Ground-truth whole tumour defined as `seg > 0`?
- Unique seg values in slice 67: [0.0, 1.0, 2.0, 4.0]
- `seg > 0` includes all non-zero labels (sub-regions 1,2,4) → **CONFIRMED**

### Pipeline reconstruction (same parameters as Week 5/6)
- Brain-mask pixels: 16345
- Otsu threshold (on brain pixels): 0.450232
- Candidate pixels before cleanup: 4268
- Total connected components found: 2
- Components retained (≥ 50 px): 2
  - C1: area=127 px  centroid=(55.24,141.91)
  - C2: area=4126 px  centroid=(109.71,89.59)

### C4. Largest component selected using area alone (no GT)?
- Ranking by area descending: C2(4126px), C1(127px)
- Selected: C2 (4126 px)  → **AREA ONLY, no GT used**

### C5. Ground truth excluded from component selection?
- Code path: `localize_tumor()` sorts by `area_px` only.
- GT mask is first accessed inside `analyze_components_vs_gt()`, which is called
  AFTER the localization decision is made (step 7 in week6_tumor_localization.py).
- **CONFIRMED: GT is not used in selection.**

### C6. Same Week 5 and Week 6 parameters used?
- DISK_R_BRAIN_OPEN=3  DISK_R_TUMOR_OPEN=2
  DISK_R_TUMOR_CLOSE=2  MIN_COMPONENT_AREA=50
- Audit uses identical constants from week5_evaluation.csv → **CONFIRMED**

### C7. Metrics calculated from actual binary masks?
- `compute_metrics(pred, gt)` operates on boolean arrays derived from
  `pred > 0` and `gt > 0` (lines 179-190 of week6_tumor_localization.py).
- **CONFIRMED: no shortcut or manual entry of values.**

### C (extra). Does Week 5 candidate mask contain the smaller component that Week 6 removes?
- Smallest component: C1 (127 px)
- Pixels of small component in Week 5 mask: 127  (= area, as expected)
- Week 5 mask contains this component: **YES**
- Week 6 mask excludes it: **YES** (only C2 retained)

## Section A — Week 5 Metric Verification

Claimed (from results/week5_evaluation.csv):
  TP=4088  FP=165  FN=960
  Dice=0.879  IoU=0.7842  Precision=0.9612  Recall=0.8098

Recomputed (this audit):
  TP=4088  FP=165  FN=960
  Dice=0.879  IoU=0.7842  Precision=0.9612  Recall=0.8098

  tp        : claimed=4088  recalc=4088  → ✓ MATCH
  fp        : claimed=165  recalc=165  → ✓ MATCH
  fn        : claimed=960  recalc=960  → ✓ MATCH
  dice      : claimed=0.879  recalc=0.879  → ✓ MATCH
  iou       : claimed=0.7842  recalc=0.7842  → ✓ MATCH
  precision : claimed=0.9612  recalc=0.9612  → ✓ MATCH
  recall    : claimed=0.8098  recalc=0.8098  → ✓ MATCH

**Week 5 audit: ALL PASS**

## Section B — Week 6 Metric Verification

Claimed (from results/week6_localization.csv):
  TP=4088  FP=38  FN=960
  Dice=0.8912  IoU=0.8038  Precision=0.9908  Recall=0.8098

Recomputed (this audit):
  TP=4088  FP=38  FN=960
  Dice=0.8912  IoU=0.8038  Precision=0.9908  Recall=0.8098

  tp        : claimed=4088  recalc=4088  → ✓ MATCH
  fp        : claimed=38  recalc=38  → ✓ MATCH
  fn        : claimed=960  recalc=960  → ✓ MATCH
  dice      : claimed=0.8912  recalc=0.8912  → ✓ MATCH
  iou       : claimed=0.8038  recalc=0.8038  → ✓ MATCH
  precision : claimed=0.9908  recalc=0.9908  → ✓ MATCH
  recall    : claimed=0.8098  recalc=0.8098  → ✓ MATCH

**Week 6 audit: ALL PASS**

## Section D — Development vs Evaluation Patient Split

**Patient 001 (BraTS20_Training_001)** was used throughout Weeks 1–6 for:
  - parameter exploration (brain-mask radii, Otsu, morphology parameters)
  - slice selection (slice 67 was chosen after inspecting GT area distribution)
  - method comparison grid (Stage 5 adaptive-threshold tuning on dev slices)

**Classification: Patient 001 = DEVELOPMENT / REFERENCE patient.**
Its results must NOT be included in the primary evaluation average.

For Week 7 multi-patient evaluation, four NEW patients must be selected
that were NEVER used for parameter tuning. These form the independent
EVALUATION group. Their average Dice/IoU/Precision/Recall is the primary result.

If any newly selected patient appears in results/slice_list.csv with
split='dev', it was used for tuning and must be replaced.

Patients in results/slice_list.csv (previously used): ['BraTS20_Training_013', 'BraTS20_Training_016', 'BraTS20_Training_017', 'BraTS20_Training_045', 'BraTS20_Training_048', 'BraTS20_Training_053', 'BraTS20_Training_058', 'BraTS20_Training_072', 'BraTS20_Training_112', 'BraTS20_Training_115', 'BraTS20_Training_120', 'BraTS20_Training_126', 'BraTS20_Training_141', 'BraTS20_Training_217', 'BraTS20_Training_259', 'BraTS20_Training_280', 'BraTS20_Training_303', 'BraTS20_Training_309', 'BraTS20_Training_328', 'BraTS20_Training_347']

Any new patient NOT in this list is confirmed as unused in earlier tuning.

## Overall Audit Verdict

| Check | Status |
|-------|--------|
| C1: Correct patient/slice loaded | ✓ |
| C2: Mask dimensions match | ✓ |
| C3: GT = seg > 0 | ✓ |
| C4: Largest-area localization rule | ✓ |
| C5: GT excluded from selection | ✓ |
| C6: Same parameters Week 5/6 | ✓ |
| C7: Metrics from binary masks | ✓ |
| A: Week 5 metrics verified | ✓ ALL PASS |
| B: Week 6 metrics verified | ✓ ALL PASS |
| D: Patient 001 = Dev (excluded from primary eval avg) | ✓ |

