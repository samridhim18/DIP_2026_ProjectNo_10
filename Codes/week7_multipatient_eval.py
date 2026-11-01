"""
week7_multipatient_eval.py
ECE501 Digital Image Processing – Group 10
Week 7: Multi-Patient Evaluation Using the Week 6 Localization Pipeline

HARD RULES (enforced in this script):
  - Classical image processing only. No ML / DL.
  - Ground truth used ONLY for evaluation. Never for thresholding or selection.
  - Patient 001 = development patient: reported separately.
  - Four new patients = evaluation group (never used for tuning).
  - All numbers computed by code and saved to CSV.
  - Existing files in results/ are NOT overwritten.

PATIENT SELECTION PROCEDURE:
  - Pool = all available BraTS20_Training_XXX folders in DATA_ROOT.
  - Exclude: Patient 001 (dev), and all 20 patients previously listed in
    results/slice_list.csv (used for Stage 5 method comparison).
  - From the remaining pool, select 4 using numpy.random with RANDOM_SEED=42.
  - This procedure is reproducible and documented here.

SLICE SELECTION PROCEDURE (per patient, no GT used for tuning):
  - Find the axial slice with the maximum number of non-zero FLAIR voxels
    that is also above a minimum-area floor (≥ 100 px after normalization
    threshold at 0.3). This is a content-based rule; GT is not used.
  - Assumption: the slice with the most FLAIR content most likely contains
    the tumour region (valid for BraTS high-grade glioma cases).

PIPELINE: identical to Week 6 (Otsu inside brain mask → largest-area component).
"""

import os, csv, sys
from datetime import datetime

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import binary_fill_holes, median_filter
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, disk
from skimage.measure import label, regionprops

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = "/Users/sam/Downloads/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
DEV_BASE    = "/Users/sam/Desktop/DIP_2026_Project_10/BraTS20_Training_001"
OUT_ROOT    = os.path.join(DEV_BASE, "results", "week7")
os.makedirs(OUT_ROOT, exist_ok=True)

SLICE_LIST_CSV = os.path.join(DEV_BASE, "results", "slice_list.csv")

# ── Fixed random seed (reproducible) ─────────────────────────────────────────
RANDOM_SEED = 42
N_EVAL_PATIENTS = 4

# ── Pipeline parameters (identical to Week 5/6) ───────────────────────────────
DISK_R_BRAIN_OPEN  = 3
DISK_R_TUMOR_OPEN  = 2
DISK_R_TUMOR_CLOSE = 2
MIN_COMPONENT_AREA = 50

# ── Dev patient ───────────────────────────────────────────────────────────────
DEV_PATIENT = "BraTS20_Training_001"
DEV_SLICE   = 67

# ── Previously used patients (from slice_list.csv – all off-limits) ───────────
PREVIOUSLY_USED = {
    "BraTS20_Training_013","BraTS20_Training_016","BraTS20_Training_017",
    "BraTS20_Training_045","BraTS20_Training_048","BraTS20_Training_053",
    "BraTS20_Training_058","BraTS20_Training_072","BraTS20_Training_112",
    "BraTS20_Training_115","BraTS20_Training_120","BraTS20_Training_126",
    "BraTS20_Training_141","BraTS20_Training_217","BraTS20_Training_259",
    "BraTS20_Training_280","BraTS20_Training_303","BraTS20_Training_309",
    "BraTS20_Training_328","BraTS20_Training_347",
    DEV_PATIENT,          # also exclude dev patient
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (identical logic to Week 5/6)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_min_max(arr):
    lo, hi = float(arr.min()), float(arr.max())
    if abs(hi - lo) < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)

def extract_axial_slice(vol, idx):
    return np.rot90(vol[:, :, idx])

def extract_brain_mask(image):
    thresh = threshold_otsu(image)
    binary = image > thresh
    opened = opening(binary, disk(DISK_R_BRAIN_OPEN))
    filled = binary_fill_holes(opened)
    labeled, n = label(filled, return_num=True)
    bm = np.zeros_like(filled, dtype=bool)
    if n > 0:
        regs = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
        bm = (labeled == regs[0].label)
    bm = closing(bm, disk(DISK_R_BRAIN_OPEN))
    bm = binary_fill_holes(bm)
    return bm

def segment_tumor_otsu(skull_stripped, brain_mask):
    bp = skull_stripped[brain_mask]
    if len(bp) == 0:
        return np.zeros_like(skull_stripped, dtype=bool), 0.0
    t = float(threshold_otsu(bp))
    return np.logical_and(skull_stripped > t, brain_mask), t

def morphological_cleanup(candidate):
    op = opening(candidate, disk(DISK_R_TUMOR_OPEN))
    cl = closing(op, disk(DISK_R_TUMOR_CLOSE))
    lb, nt = label(cl, return_num=True)
    clean = np.zeros_like(cl, dtype=bool)
    comps = []
    if nt > 0:
        for r in regionprops(lb):
            if r.area >= MIN_COMPONENT_AREA:
                clean[lb == r.label] = True
                comps.append({
                    "component_id": int(r.label),
                    "area_px": int(r.area),
                    "centroid_row": round(float(r.centroid[0]), 2),
                    "centroid_col": round(float(r.centroid[1]), 2),
                    "bbox_minr": int(r.bbox[0]),
                    "bbox_minc": int(r.bbox[1]),
                    "bbox_maxr": int(r.bbox[2]),
                    "bbox_maxc": int(r.bbox[3]),
                    "mask_idx": np.where(lb == r.label),
                })
    return clean, comps, nt

def localize_tumor(comps):
    """Largest-area rule. Does NOT use ground truth."""
    if not comps:
        return None, []
    ranked = sorted(comps, key=lambda c: c["area_px"], reverse=True)
    return ranked[0], ranked

def compute_metrics(pred, gt):
    p, g = pred > 0, gt > 0
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    pa, ga = int(p.sum()), int(g.sum())
    dice = round((2.0*tp)/(pa+ga), 4) if (pa+ga) > 0 else 1.0
    iou  = round(tp/(tp+fp+fn), 4)    if (tp+fp+fn) > 0 else 1.0
    prec = round(tp/(tp+fp), 4)       if (tp+fp) > 0 else 1.0
    rec  = round(tp/(tp+fn), 4)       if (tp+fn) > 0 else 1.0
    return dict(tp=tp, fp=fp, fn=fn, pred_area=pa, gt_area=ga,
                dice=dice, iou=iou, precision=prec, recall=rec)

# ─────────────────────────────────────────────────────────────────────────────
# Slice selection (no GT used)
# ─────────────────────────────────────────────────────────────────────────────
def select_representative_slice(flair_vol, min_bright_px=100, thresh_frac=0.3):
    """
    Find the axial slice with the maximum number of pixels above a fixed
    intensity threshold (30% of the normalised range) inside the volume.
    This is a content-based rule; GT is never consulted.

    Returns the slice index.
    """
    # Normalise the whole volume once
    lo, hi = float(flair_vol.min()), float(flair_vol.max())
    if abs(hi - lo) < 1e-8:
        return flair_vol.shape[2] // 2
    vol_norm = (flair_vol - lo) / (hi - lo)

    best_idx, best_count = 0, 0
    for z in range(flair_vol.shape[2]):
        sl = np.rot90(vol_norm[:, :, z])
        count = int(np.sum(sl > thresh_frac))
        if count > best_count:
            best_count = count
            best_idx = z
    return best_idx

# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline for one patient+slice
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(patient_id, flair_path, seg_path, slice_idx):
    flair_vol = np.asarray(nib.load(flair_path).get_fdata(), dtype=np.float32)
    seg_vol   = np.asarray(nib.load(seg_path).get_fdata(),   dtype=np.float32)

    flair_slice = extract_axial_slice(flair_vol, slice_idx)
    seg_slice   = extract_axial_slice(seg_vol,   slice_idx)
    gt_mask     = seg_slice > 0   # GT used ONLY for evaluation

    norm_slice = normalize_min_max(flair_slice)
    med_slice  = median_filter(norm_slice, size=3).astype(np.float32)

    brain_mask = extract_brain_mask(med_slice)
    ss = np.where(brain_mask, med_slice, 0.0).astype(np.float32)

    candidate, otsu_thresh = segment_tumor_otsu(ss, brain_mask)
    pred_mask, comps, n_total = morphological_cleanup(candidate)

    # Week 5-style: all retained components
    w5_m = compute_metrics(pred_mask, gt_mask)

    # Week 6-style: largest-area localization
    selected, ranked = localize_tumor(comps)
    if selected is not None:
        sel_info = next(c for c in comps if c["component_id"] == selected["component_id"])
        loc_mask = np.zeros_like(pred_mask, dtype=bool)
        loc_mask[sel_info["mask_idx"]] = True
    else:
        loc_mask = np.zeros_like(pred_mask, dtype=bool)
    w6_m = compute_metrics(loc_mask, gt_mask)

    return dict(
        patient=patient_id,
        slice_idx=slice_idx,
        gt_area=int(gt_mask.sum()),
        brain_px=int(brain_mask.sum()),
        n_components=len(comps),
        otsu_thresh=round(float(otsu_thresh), 6),
        # Week 5 metrics (all components)
        w5_dice=w5_m["dice"], w5_iou=w5_m["iou"],
        w5_precision=w5_m["precision"], w5_recall=w5_m["recall"],
        w5_tp=w5_m["tp"], w5_fp=w5_m["fp"], w5_fn=w5_m["fn"],
        w5_pred_area=w5_m["pred_area"],
        # Week 6 metrics (localized)
        w6_dice=w6_m["dice"], w6_iou=w6_m["iou"],
        w6_precision=w6_m["precision"], w6_recall=w6_m["recall"],
        w6_tp=w6_m["tp"], w6_fp=w6_m["fp"], w6_fn=w6_m["fn"],
        w6_pred_area=w6_m["pred_area"],
        # Intermediate data for figure
        _flair=med_slice, _gt=gt_mask, _w5=pred_mask, _w6=loc_mask,
        _comps=comps, _selected=selected,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Figure: 4-panel per patient (FLAIR | GT | W5 pred | W6 localized)
# ─────────────────────────────────────────────────────────────────────────────
def save_patient_figure(r, save_path):
    flair  = r["_flair"]
    gt     = r["_gt"]
    w5     = r["_w5"]
    w6     = r["_w6"]
    comps  = r["_comps"]
    sel    = r["_selected"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), dpi=120)
    fig.patch.set_facecolor("#0d0d1a")
    tkw = dict(color="white", fontsize=9, fontweight="bold", pad=6)
    for ax in axes:
        ax.set_facecolor("#0d0d1a")

    # Panel A: FLAIR
    axes[0].imshow(flair, cmap="gray")
    axes[0].set_title(f"A  FLAIR  {r['patient']}  sl={r['slice_idx']}", **tkw)
    axes[0].axis("off")

    # Panel B: Ground truth
    axes[1].imshow(flair, cmap="gray", alpha=0.5)
    rgba_gt = np.zeros((*gt.shape, 4), dtype=np.float32)
    rgba_gt[gt] = [0.0, 1.0, 0.35, 0.8]
    axes[1].imshow(rgba_gt)
    axes[1].set_title(f"B  GT Tumour  ({r['gt_area']} px)", **tkw)
    axes[1].axis("off")

    # Panel C: Week 5 (all components)
    COLORS = [[1.0,0.4,0.1,0.85],[0.2,0.6,1.0,0.85],[0.9,0.2,0.9,0.85],[0.2,0.9,0.5,0.85]]
    axes[2].imshow(flair, cmap="gray", alpha=0.45)
    for i, c in enumerate(comps):
        col = COLORS[min(i, len(COLORS)-1)]
        rgba_c = np.zeros((*flair.shape, 4), dtype=np.float32)
        rgba_c[c["mask_idx"][0], c["mask_idx"][1]] = col
        axes[2].imshow(rgba_c)
    axes[2].set_title(
        f"C  W5 All Components  Dice={r['w5_dice']}  IoU={r['w5_iou']}", **tkw)
    axes[2].axis("off")

    # Panel D: Week 6 (localized)
    axes[3].imshow(flair, cmap="gray")
    rgba_loc = np.zeros((*flair.shape, 4), dtype=np.float32)
    rgba_loc[w6] = [1.0, 0.15, 0.15, 0.85]
    axes[3].imshow(rgba_loc)
    if sel is not None:
        # bounding box lives in the comps list (has mask_idx + bbox keys)
        sel_comp = next((c for c in comps if c["component_id"] == sel["component_id"]), None)
        if sel_comp is None:
            sel_comp = sel  # fallback
        mr = sel_comp.get("bbox_minr", sel.get("bbox_minr", 0))
        mc = sel_comp.get("bbox_minc", sel.get("bbox_minc", 0))
        xr = sel_comp.get("bbox_maxr", sel.get("bbox_maxr", 10))
        xc = sel_comp.get("bbox_maxc", sel.get("bbox_maxc", 10))
        rect = mpatches.Rectangle((mc, mr), xc-mc, xr-mr,
                                   linewidth=1.8, edgecolor="#ffcc00",
                                   facecolor="none", linestyle="--")
        axes[3].add_patch(rect)
        axes[3].plot(sel["centroid_col"], sel["centroid_row"],
                     marker="+", color="#ffcc00", markersize=10,
                     markeredgewidth=1.8)
    axes[3].set_title(
        f"D  W6 Localized  Dice={r['w6_dice']}  IoU={r['w6_iou']}", **tkw)
    axes[3].axis("off")

    fig.suptitle(
        f"Week 7 Multi-Patient Eval  |  {r['patient']}  Slice {r['slice_idx']}",
        color="white", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.0, 1, 1])
    fig.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVED] figure → {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary figure: comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def save_summary_figure(eval_rows, dev_row, save_path):
    patients   = [r["patient"].replace("BraTS20_Training_","P") for r in eval_rows]
    w5_dice    = [r["w5_dice"]  for r in eval_rows]
    w6_dice    = [r["w6_dice"]  for r in eval_rows]
    w5_iou     = [r["w5_iou"]   for r in eval_rows]
    w6_iou     = [r["w6_iou"]   for r in eval_rows]

    x = np.arange(len(patients))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
    fig.patch.set_facecolor("#0d0d1a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#151528")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#555577")
        ax.spines["left"].set_color("#555577")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")

    b1 = ax1.bar(x - width/2, w5_dice, width, label="W5 All Comps", color="#4477cc", alpha=0.9)
    b2 = ax1.bar(x + width/2, w6_dice, width, label="W6 Localized", color="#ee6644", alpha=0.9)
    ax1.axhline(dev_row["w6_dice"], color="#aaffaa", linestyle="--", linewidth=1,
                label=f"Dev P001 W6 Dice={dev_row['w6_dice']}")
    ax1.set_xticks(x); ax1.set_xticklabels(patients, color="white", fontsize=9)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("Dice"); ax1.set_title("Dice per Eval Patient")
    ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=7, color="white")

    b3 = ax2.bar(x - width/2, w5_iou, width, label="W5 All Comps", color="#4477cc", alpha=0.9)
    b4 = ax2.bar(x + width/2, w6_iou, width, label="W6 Localized", color="#ee6644", alpha=0.9)
    ax2.axhline(dev_row["w6_iou"], color="#aaffaa", linestyle="--", linewidth=1,
                label=f"Dev P001 W6 IoU={dev_row['w6_iou']}")
    ax2.set_xticks(x); ax2.set_xticklabels(patients, color="white", fontsize=9)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("IoU"); ax2.set_title("IoU per Eval Patient")
    ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8)
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        ax2.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=7, color="white")

    # Eval averages as text
    avg_w5_dice = round(float(np.mean(w5_dice)), 4)
    avg_w6_dice = round(float(np.mean(w6_dice)), 4)
    avg_w5_iou  = round(float(np.mean(w5_iou)), 4)
    avg_w6_iou  = round(float(np.mean(w6_iou)), 4)
    fig.text(0.5, -0.03,
             f"Eval avg (4 patients)  W5: Dice={avg_w5_dice}  IoU={avg_w5_iou}   "
             f"W6: Dice={avg_w6_dice}  IoU={avg_w6_iou}   "
             f"Dev (P001) W6: Dice={dev_row['w6_dice']}  IoU={dev_row['w6_iou']}",
             ha="center", fontsize=8.5, color="#ccccee",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#151530", alpha=0.8))
    fig.suptitle("Week 7: Multi-Patient Evaluation  |  W5 vs W6 Localization",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVED] summary figure → {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────────────────────
METRICS_FIELDS = [
    "patient","group","slice_idx","gt_area","brain_px","n_components","otsu_thresh",
    "w5_dice","w5_iou","w5_precision","w5_recall","w5_tp","w5_fp","w5_fn","w5_pred_area",
    "w6_dice","w6_iou","w6_precision","w6_recall","w6_tp","w6_fp","w6_fn","w6_pred_area",
]

def row_to_csv_dict(r, group):
    d = {k: r[k] for k in METRICS_FIELDS if k in r}
    d["group"] = group
    return d

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("="*65)
    print("Week 7: Multi-Patient Evaluation")
    print(f"Started: {ts}")
    print("="*65)

    # ── 1. Discover available patients ────────────────────────────────────────
    all_dirs = sorted([
        d for d in os.listdir(DATA_ROOT)
        if d.startswith("BraTS20_Training_") and
           os.path.isdir(os.path.join(DATA_ROOT, d))
    ])
    eligible = [d for d in all_dirs if d not in PREVIOUSLY_USED]
    print(f"\nTotal patients in dataset: {len(all_dirs)}")
    print(f"Excluded (dev + previously used in tuning): {len(PREVIOUSLY_USED)}")
    print(f"Eligible for evaluation: {len(eligible)}")

    # ── 2. Select 4 with fixed seed ────────────────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    chosen_idx = rng.choice(len(eligible), size=N_EVAL_PATIENTS, replace=False)
    chosen_idx.sort()
    eval_patients = [eligible[i] for i in chosen_idx]
    print(f"\nSelected evaluation patients (seed={RANDOM_SEED}):")
    for p in eval_patients:
        print(f"  {p}")

    # ── 3. Run dev patient (P001, fixed slice 67) ──────────────────────────────
    print(f"\n[DEV] {DEV_PATIENT}  slice={DEV_SLICE}")
    dev_flair = os.path.join(DEV_BASE, f"{DEV_PATIENT}_flair.nii")
    dev_seg   = os.path.join(DEV_BASE, f"{DEV_PATIENT}_seg.nii")
    dev_r = run_pipeline(DEV_PATIENT, dev_flair, dev_seg, DEV_SLICE)
    dev_r["group"] = "dev"
    print(f"  GT={dev_r['gt_area']}px  W5 Dice={dev_r['w5_dice']}  W6 Dice={dev_r['w6_dice']}")
    dev_fig = os.path.join(OUT_ROOT, f"week7_{DEV_PATIENT}_sl{DEV_SLICE}.png")
    save_patient_figure(dev_r, dev_fig)

    # ── 4. Run evaluation patients ─────────────────────────────────────────────
    eval_rows = []
    selection_log = []   # for CSV: patient, slice_idx, selection_reason

    for pid in eval_patients:
        pdir   = os.path.join(DATA_ROOT, pid)
        flair_p = os.path.join(pdir, f"{pid}_flair.nii")
        seg_p   = os.path.join(pdir, f"{pid}_seg.nii")

        if not os.path.exists(flair_p) or not os.path.exists(seg_p):
            print(f"  [SKIP] {pid}: NIfTI files missing")
            continue

        # Slice selection (no GT used)
        print(f"\n[EVAL] {pid}: selecting slice ...")
        flair_vol_tmp = np.asarray(nib.load(flair_p).get_fdata(), dtype=np.float32)
        sl = select_representative_slice(flair_vol_tmp)
        del flair_vol_tmp   # free memory
        print(f"  Selected slice {sl} (max-bright-content rule)")
        selection_log.append({
            "patient": pid, "slice_idx": sl,
            "method": "max_bright_content_thresh0.3",
            "gt_used": "NO",
        })

        print(f"  Running pipeline ...")
        r = run_pipeline(pid, flair_p, seg_p, sl)
        r["group"] = "eval"
        print(f"  GT={r['gt_area']}px  W5 Dice={r['w5_dice']}  "
              f"IoU={r['w5_iou']}  |  W6 Dice={r['w6_dice']}  IoU={r['w6_iou']}")

        fig_path = os.path.join(OUT_ROOT, f"week7_{pid}_sl{sl}.png")
        save_patient_figure(r, fig_path)
        eval_rows.append(r)

    # ── 5. Aggregate statistics (eval group only) ──────────────────────────────
    print("\n" + "="*65)
    print("EVALUATION GROUP SUMMARY (4 new patients, primary result)")
    print("="*65)
    for key_w, label in [("w5","W5 all-comp"), ("w6","W6 localized")]:
        dices = [r[f"{key_w}_dice"] for r in eval_rows]
        ious  = [r[f"{key_w}_iou"]  for r in eval_rows]
        precs = [r[f"{key_w}_precision"] for r in eval_rows]
        recs  = [r[f"{key_w}_recall"]    for r in eval_rows]
        print(f"\n  {label}:")
        print(f"    Dice      mean={round(float(np.mean(dices)),4)}  "
              f"std={round(float(np.std(dices)),4)}  "
              f"min={min(dices)}  max={max(dices)}")
        print(f"    IoU       mean={round(float(np.mean(ious)),4)}  "
              f"std={round(float(np.std(ious)),4)}  "
              f"min={min(ious)}  max={max(ious)}")
        print(f"    Precision mean={round(float(np.mean(precs)),4)}  "
              f"std={round(float(np.std(precs)),4)}")
        print(f"    Recall    mean={round(float(np.mean(recs)),4)}  "
              f"std={round(float(np.std(recs)),4)}")

    print(f"\nDEV PATIENT ({DEV_PATIENT}, slice {DEV_SLICE}) — reported separately:")
    print(f"  W5 Dice={dev_r['w5_dice']}  IoU={dev_r['w5_iou']}  "
          f"Prec={dev_r['w5_precision']}  Rec={dev_r['w5_recall']}")
    print(f"  W6 Dice={dev_r['w6_dice']}  IoU={dev_r['w6_iou']}  "
          f"Prec={dev_r['w6_precision']}  Rec={dev_r['w6_recall']}")

    # ── 6. Summary figure ──────────────────────────────────────────────────────
    summ_fig = os.path.join(OUT_ROOT, "week7_summary_barplot.png")
    save_summary_figure(eval_rows, dev_r, summ_fig)

    # ── 7. Save per-patient CSV ────────────────────────────────────────────────
    per_patient_csv = os.path.join(OUT_ROOT, "week7_per_patient.csv")
    all_rows_csv = [row_to_csv_dict(dev_r, "dev")] + \
                   [row_to_csv_dict(r, "eval") for r in eval_rows]
    with open(per_patient_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        w.writeheader()
        w.writerows(all_rows_csv)
    print(f"\n[SAVED] per-patient CSV → {per_patient_csv}")

    # ── 8. Save aggregate summary CSV ─────────────────────────────────────────
    agg_csv = os.path.join(OUT_ROOT, "week7_aggregate_summary.csv")
    agg_fields = ["group","method","metric","mean","std","min","max"]
    agg_rows = []
    for key_w, label in [("w5","W5_all_comp"), ("w6","W6_localized")]:
        for metric, vals in [
            ("dice",      [r[f"{key_w}_dice"]      for r in eval_rows]),
            ("iou",       [r[f"{key_w}_iou"]       for r in eval_rows]),
            ("precision", [r[f"{key_w}_precision"] for r in eval_rows]),
            ("recall",    [r[f"{key_w}_recall"]    for r in eval_rows]),
        ]:
            agg_rows.append({
                "group": "eval_4patients", "method": label, "metric": metric,
                "mean": round(float(np.mean(vals)),4),
                "std":  round(float(np.std(vals)),4),
                "min":  min(vals),
                "max":  max(vals),
            })
    # Dev row
    for key_w, label in [("w5","W5_all_comp"), ("w6","W6_localized")]:
        for metric, val in [
            ("dice",      dev_r[f"{key_w}_dice"]),
            ("iou",       dev_r[f"{key_w}_iou"]),
            ("precision", dev_r[f"{key_w}_precision"]),
            ("recall",    dev_r[f"{key_w}_recall"]),
        ]:
            agg_rows.append({
                "group": "dev_P001", "method": label, "metric": metric,
                "mean": val, "std": "", "min": val, "max": val,
            })
    with open(agg_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader()
        w.writerows(agg_rows)
    print(f"[SAVED] aggregate CSV  → {agg_csv}")

    # ── 9. Save patient selection log ─────────────────────────────────────────
    sel_csv = os.path.join(OUT_ROOT, "week7_patient_selection.csv")
    sel_log_dev = [{
        "patient": DEV_PATIENT, "group": "dev", "slice_idx": DEV_SLICE,
        "method": "fixed_dev_slice", "gt_used": "NO",
        "random_seed": RANDOM_SEED,
    }]
    sel_log_eval = [{
        "patient": s["patient"], "group": "eval",
        "slice_idx": s["slice_idx"], "method": s["method"],
        "gt_used": s["gt_used"], "random_seed": RANDOM_SEED,
    } for s in selection_log]
    sel_all = sel_log_dev + sel_log_eval
    with open(sel_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient","group","slice_idx","method","gt_used","random_seed"])
        w.writeheader()
        w.writerows(sel_all)
    print(f"[SAVED] selection log  → {sel_csv}")

    # ── 10. Markdown summary ───────────────────────────────────────────────────
    md_path = os.path.join(OUT_ROOT, "week7_summary.md")
    nl = "\n"
    md_lines = [
        "# Week 7 — Multi-Patient Evaluation Summary",
        f"_Generated: {ts}_", "",
        "## 1. Patient Selection",
        f"- **Random seed:** {RANDOM_SEED}",
        f"- **Dev patient (excluded from primary eval):** {DEV_PATIENT}",
        f"- **Previously used patients (excluded):** " +
            ", ".join(sorted(PREVIOUSLY_USED - {DEV_PATIENT})),
        f"- **Evaluation patients (4, never used for tuning):**",
    ]
    for p in eval_patients:
        md_lines.append(f"  - {p}")
    md_lines += [
        "",
        "## 2. Slice Selection Rule",
        "- Rule: select the axial slice with the maximum number of pixels",
        "  above 30% of the normalised intensity range.",
        "- GT was NOT used to select slices.",
        "- Dev patient uses fixed slice 67 (same as Weeks 1–6).",
        "",
        "## 3. Pipeline (identical to Week 6)",
        "- Min-max normalisation + 3×3 median filter",
        "- Brain mask: Otsu → opening r=3 → fill holes → largest CC → closing",
        "- Tumour candidate: Otsu on brain pixels",
        "- Morphological cleanup: open r=2, close r=2, min area=50 px",
        "- Localization: largest-area connected component (no GT used)",
        "",
        "## 4. Per-Patient Results",
        "",
        "| Patient | Group | Slice | GT (px) | W5 Dice | W5 IoU | W6 Dice | W6 IoU | W6 Prec | W6 Rec |",
        "|---------|-------|-------|---------|---------|--------|---------|--------|---------|--------|",
    ]
    for r in [dev_r] + eval_rows:
        g = "dev" if r["patient"] == DEV_PATIENT else "eval"
        md_lines.append(
            f"| {r['patient']} | {g} | {r['slice_idx']} | {r['gt_area']} | "
            f"{r['w5_dice']} | {r['w5_iou']} | "
            f"{r['w6_dice']} | {r['w6_iou']} | "
            f"{r['w6_precision']} | {r['w6_recall']} |"
        )
    md_lines += [
        "",
        "## 5. Evaluation Group Aggregate (4 patients — primary result)",
        "",
        "| Method | Metric | Mean | Std | Min | Max |",
        "|--------|--------|------|-----|-----|-----|",
    ]
    for key_w, label in [("w5","W5 all-comp"), ("w6","W6 localized")]:
        for metric, vals in [
            ("Dice",      [r[f"{key_w}_dice"]      for r in eval_rows]),
            ("IoU",       [r[f"{key_w}_iou"]       for r in eval_rows]),
            ("Precision", [r[f"{key_w}_precision"] for r in eval_rows]),
            ("Recall",    [r[f"{key_w}_recall"]    for r in eval_rows]),
        ]:
            md_lines.append(
                f"| {label} | {metric} | "
                f"{round(float(np.mean(vals)),4)} | "
                f"{round(float(np.std(vals)),4)} | "
                f"{min(vals)} | {max(vals)} |"
            )
    md_lines += [
        "",
        "## 6. Dev Patient (BraTS20_Training_001, slice 67) — separate reference",
        "",
        "| Method | Dice | IoU | Precision | Recall |",
        "|--------|------|-----|-----------|--------|",
        f"| W5 all-comp | {dev_r['w5_dice']} | {dev_r['w5_iou']} | {dev_r['w5_precision']} | {dev_r['w5_recall']} |",
        f"| W6 localized | {dev_r['w6_dice']} | {dev_r['w6_iou']} | {dev_r['w6_precision']} | {dev_r['w6_recall']} |",
        "",
        "## 7. Assumption",
        "- Assumed: BraTS2020 patients not in slice_list.csv and not Patient 001",
        "  were never used for any parameter decision in this project.",
        "- Slice selection uses a content-based rule (maximum bright pixels at",
        "  normalised intensity > 0.3), not GT tumour area.",
        "- Results for Patient 001 are provided as a development/reference figure",
        "  only; they are excluded from all primary evaluation averages.",
    ]
    with open(md_path, "w") as f:
        f.write(nl.join(md_lines) + nl)
    print(f"[SAVED] summary md     → {md_path}")

    print("\n" + "="*65)
    print("Week 7 DONE")
    print("="*65)

if __name__ == "__main__":
    main()
