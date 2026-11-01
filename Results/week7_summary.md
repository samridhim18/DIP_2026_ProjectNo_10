# Week 7 — Multi-Patient Evaluation Summary
_Generated: 2026-11-01 13:45:40_

## 1. Patient Selection
- **Random seed:** 42
- **Dev patient (excluded from primary eval):** BraTS20_Training_001
- **Previously used patients (excluded):** BraTS20_Training_013, BraTS20_Training_016, BraTS20_Training_017, BraTS20_Training_045, BraTS20_Training_048, BraTS20_Training_053, BraTS20_Training_058, BraTS20_Training_072, BraTS20_Training_112, BraTS20_Training_115, BraTS20_Training_120, BraTS20_Training_126, BraTS20_Training_141, BraTS20_Training_217, BraTS20_Training_259, BraTS20_Training_280, BraTS20_Training_303, BraTS20_Training_309, BraTS20_Training_328, BraTS20_Training_347
- **Evaluation patients (4, never used for tuning):**
  - BraTS20_Training_035
  - BraTS20_Training_167
  - BraTS20_Training_243
  - BraTS20_Training_285

## 2. Slice Selection Rule
- Rule: select the axial slice with the maximum number of pixels
  above 30% of the normalised intensity range.
- GT was NOT used to select slices.
- Dev patient uses fixed slice 67 (same as Weeks 1–6).

## 3. Pipeline (identical to Week 6)
- Min-max normalisation + 3×3 median filter
- Brain mask: Otsu → opening r=3 → fill holes → largest CC → closing
- Tumour candidate: Otsu on brain pixels
- Morphological cleanup: open r=2, close r=2, min area=50 px
- Localization: largest-area connected component (no GT used)

## 4. Per-Patient Results

| Patient | Group | Slice | GT (px) | W5 Dice | W5 IoU | W6 Dice | W6 IoU | W6 Prec | W6 Rec |
|---------|-------|-------|---------|---------|--------|---------|--------|---------|--------|
| BraTS20_Training_001 | dev | 67 | 5048 | 0.879 | 0.7842 | 0.8912 | 0.8038 | 0.9908 | 0.8098 |
| BraTS20_Training_035 | eval | 61 | 2274 | 0.7666 | 0.6215 | 0.7979 | 0.6637 | 0.9409 | 0.6926 |
| BraTS20_Training_167 | eval | 85 | 797 | 0.3394 | 0.2044 | 0.4808 | 0.3165 | 0.3165 | 1.0 |
| BraTS20_Training_243 | eval | 66 | 3410 | 0.9398 | 0.8864 | 0.9581 | 0.9196 | 0.9641 | 0.9522 |
| BraTS20_Training_285 | eval | 72 | 1156 | 0.3912 | 0.2432 | 0.724 | 0.5674 | 0.6582 | 0.8045 |

## 5. Evaluation Group Aggregate (4 patients — primary result)

| Method | Metric | Mean | Std | Min | Max |
|--------|--------|------|-----|-----|-----|
| W5 all-comp | Dice | 0.6092 | 0.2522 | 0.3394 | 0.9398 |
| W5 all-comp | IoU | 0.4889 | 0.2815 | 0.2044 | 0.8864 |
| W5 all-comp | Precision | 0.5622 | 0.3322 | 0.2044 | 0.9277 |
| W5 all-comp | Recall | 0.8623 | 0.1216 | 0.6926 | 1.0 |
| W6 localized | Dice | 0.7402 | 0.172 | 0.4808 | 0.9581 |
| W6 localized | IoU | 0.6168 | 0.2159 | 0.3165 | 0.9196 |
| W6 localized | Precision | 0.7199 | 0.2622 | 0.3165 | 0.9641 |
| W6 localized | Recall | 0.8623 | 0.1216 | 0.6926 | 1.0 |

## 6. Dev Patient (BraTS20_Training_001, slice 67) — separate reference

| Method | Dice | IoU | Precision | Recall |
|--------|------|-----|-----------|--------|
| W5 all-comp | 0.879 | 0.7842 | 0.9612 | 0.8098 |
| W6 localized | 0.8912 | 0.8038 | 0.9908 | 0.8098 |

## 7. Assumption
- Assumed: BraTS2020 patients not in slice_list.csv and not Patient 001
  were never used for any parameter decision in this project.
- Slice selection uses a content-based rule (maximum bright pixels at
  normalised intensity > 0.3), not GT tumour area.
- Results for Patient 001 are provided as a development/reference figure
  only; they are excluded from all primary evaluation averages.
