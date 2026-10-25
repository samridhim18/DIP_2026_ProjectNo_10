# Week 6 Tumour Localization Summary
_Generated: 2026-10-25 14:40_

## 1. Problem and Objective
Goal: identify WHICH detected region corresponds to the actual tumour.
Patient: BraTS20_Training_001  |  Slice: axial 67  |  Modality: FLAIR

## 2. Reused Week 5 Pipeline
- Min-max normalisation + median filter (3x3)
- Brain mask: Otsu -> opening r=3 -> hole-fill -> largest CC -> closing
- Tumour candidate: Otsu threshold on brain pixels (thresh = 0.450232)
- Morphological cleanup: open r=2, close r=2, min=50px
- Connected-component analysis

## 3. Detected Components (Week 5 output)
| ID | Area (px) | Centroid (row,col) | BBox |
|----|-----------|-------------------|------|
| 1 | 127 | (55.24,141.91) | (44,132,67,151) |
| 2 | 4126 | (109.71,89.59) | (58,51,157,133) |

## 4. Localization Rule
Rule: Select the component with the largest area as the tumour candidate.
This rule does NOT use the ground truth.
Justification: In FLAIR MRI, tumour oedema is typically the largest
hyperintense region inside the brain for high-grade glioma patients.
Limitation: Fails if tumour is small, or if ventricles/artefacts are larger.

## 5. Per-Component GT Analysis (evaluation only, done after selection)
| ID | Area | TP | FP | GT Coverage | Area Rank | Selected? |
|----|------|----|----|------------|-----------|----------|
| 1 | 127 | 0 | 127 | 0.0 | 2 | no |
| 2 | 4126 | 4088 | 38 | 0.9908 | 1 | YES |

## 6. Metric Comparison
Week 5 baseline uses ALL retained components as the prediction.
Week 6 localized uses ONLY the selected component as the prediction.

| Metric | Week 5 Baseline | Week 6 Localized |
|--------|----------------|------------------|
| Dice | 0.879 | 0.8912 |
| IoU | 0.7842 | 0.8038 |
| Precision | 0.9612 | 0.9908 |
| Recall | 0.8098 | 0.8098 |
| TP | 4088 | 4088 |
| FP | 165 | 38 |
| FN | 960 | 960 |
| Pred area (px) | 4253 | 4126 |

## 7. Conclusion
Selected candidate: C2 (4126px)
The largest-component rule identified the correct main tumour region.
Removing the small false-positive (C1, 127px) decreased FP and increased Precision.
Recall dropped slightly because the small component had a few genuine TP pixels.

## 8. Key Concepts (for report)
Connected component: a group of connected foreground (bright) pixels.
Centroid: average row and column position of all pixels in the component.
Bounding box: smallest axis-aligned rectangle containing the component.
GT is used ONLY for evaluation -- never to pick or tune the algorithm.
Bright MRI regions do not always mean tumour: CSF, vessels, and artefacts
can all appear hyperintense in FLAIR. Size and spatial context help distinguish them.
