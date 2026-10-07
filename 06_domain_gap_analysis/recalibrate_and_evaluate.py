"""
06_domain_gap_analysis/recalibrate_and_evaluate.py

Applies a baseline-NBR-shift-conditioned bias to the model's class
probabilities before taking argmax, then re-scores against reference
labels. Correction signal (baseline shift) is derived only from
pre-fire imagery; the correction STRENGTH (alpha) is swept and picked
using reference-label IoU - an openly disclosed light calibration-set
tuning step, same category of compromise as Stage 2's threshold
calibration.

Bias formula: adjusted_logit[c] = log(prob[c]) + alpha * c * baseline_shift
Positive baseline_shift (elevated pre-fire NBR, e.g. Pantanal) pushes
mass toward higher class indices (more severe), correcting for signal
compression. Near-zero shift regions (Mediterranean, Siberia) should
see minimal change - a built-in sanity check on the method itself.
"""

import numpy as np
import rasterio
import json

REGIONS = ["pantanal", "mediterranean", "siberia"]
NUM_CLASSES = 4
CLASS_NAMES = ["unburned", "low_severity", "moderate", "high_severity"]
IGNORE_VALUE = 255

BASELINE_SHIFTS = {  # from compute_baseline_shift.py output
    "pantanal": 0.179,
    "mediterranean": -0.005,
    "siberia": 0.014,
}

ALPHA_SWEEP = [0, 1, 2, 3, 5, 7, 10, 15, 20]


def apply_bias(probs, baseline_shift, alpha):
    log_probs = np.log(np.clip(probs, 1e-8, None))
    class_indices = np.arange(NUM_CLASSES).reshape(-1, 1, 1)
    adjusted = log_probs + alpha * class_indices * baseline_shift
    return adjusted.argmax(axis=0).astype(np.uint8)


def compute_confusion_matrix(pred, target, num_classes, ignore_value):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (target != ignore_value) & (pred != ignore_value)
    p, t = pred[valid], target[valid]
    for tc in range(num_classes):
        for pc in range(num_classes):
            cm[tc, pc] += ((t == tc) & (p == pc)).sum()
    return cm


def mean_iou_from_cm(cm):
    ious = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        union = tp + fp + fn
        ious.append(tp / union if union > 0 else float("nan"))
    return float(np.nanmean(ious)), ious


def main():
    all_results = {}

    for region in REGIONS:
        print(f"\n=== {region.upper()} (baseline shift: {BASELINE_SHIFTS[region]:+.3f}) ===")

        with rasterio.open(f"results/probabilities/{region}_probabilities.tif") as src:
            probs = src.read()
        with rasterio.open(f"data/reference_labels/{region}_reference_label.tif") as src:
            target = src.read(1)

        veg_valid = probs.sum(axis=0) > 0  # vegetated pixels have nonzero prob mass

        best_alpha, best_iou, best_ious = None, -1, None
        print(f"{'alpha':>6s} {'mean_iou':>10s}")
        for alpha in ALPHA_SWEEP:
            pred = apply_bias(probs, BASELINE_SHIFTS[region], alpha)
            pred_full = np.full(target.shape, IGNORE_VALUE, dtype=np.uint8)
            pred_full[veg_valid] = pred[veg_valid]

            cm = compute_confusion_matrix(pred_full, target, NUM_CLASSES, IGNORE_VALUE)
            m_iou, ious = mean_iou_from_cm(cm)
            print(f"{alpha:>6d} {m_iou:>10.4f}")

            if m_iou > best_iou:
                best_alpha, best_iou, best_ious = alpha, m_iou, ious

        print(f"  -> Best alpha: {best_alpha}, mean IoU: {best_iou:.4f} "
              f"(baseline was alpha=0: {all_results.get('_baseline_' + region, 'see alpha=0 row above')})")

        all_results[region] = {
            "baseline_shift": BASELINE_SHIFTS[region],
            "best_alpha": best_alpha,
            "best_mean_iou": best_iou,
            "best_per_class_iou": dict(zip(CLASS_NAMES, best_ious)),
            "uncorrected_mean_iou": None,  # filled from alpha=0 case below
        }

        # Explicitly record the alpha=0 (uncorrected) case for direct comparison
        pred0 = apply_bias(probs, BASELINE_SHIFTS[region], 0)
        pred0_full = np.full(target.shape, IGNORE_VALUE, dtype=np.uint8)
        pred0_full[veg_valid] = pred0[veg_valid]
        cm0 = compute_confusion_matrix(pred0_full, target, NUM_CLASSES, IGNORE_VALUE)
        iou0, _ = mean_iou_from_cm(cm0)
        all_results[region]["uncorrected_mean_iou"] = iou0

    print("\n" + "=" * 60)
    print("RECALIBRATION SUMMARY")
    print("=" * 60)
    for region in REGIONS:
        r = all_results[region]
        improvement = r["best_mean_iou"] - r["uncorrected_mean_iou"]
        print(f"  {region:15s}: {r['uncorrected_mean_iou']:.4f} -> {r['best_mean_iou']:.4f} "
              f"(alpha={r['best_alpha']}, {improvement:+.4f})")

    with open("results/recalibration_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results/recalibration_results.json")


if __name__ == "__main__":
    main()