"""
06_domain_gap_analysis/evaluate_domain_gap.py

Scores zero-shot predictions against the lightweight reference labels,
per region. Same confusion-matrix-based IoU/Dice approach as Stage 5,
so results are directly comparable to the in-domain Similipal numbers
(mean IoU 0.6659).
"""

import json
import numpy as np
import rasterio

REGIONS = ["pantanal", "mediterranean", "siberia"]
NUM_CLASSES = 4
CLASS_NAMES = ["unburned", "low_severity", "moderate", "high_severity"]
IGNORE_VALUE = 255


def compute_confusion_matrix(pred, target, num_classes, ignore_value):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (target != ignore_value) & (pred != ignore_value)
    p, t = pred[valid], target[valid]
    for true_c in range(num_classes):
        for pred_c in range(num_classes):
            cm[true_c, pred_c] += ((t == true_c) & (p == pred_c)).sum()
    return cm


def iou_and_dice_from_cm(cm):
    ious, dices = [], []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        union = tp + fp + fn
        iou = tp / union if union > 0 else float("nan")
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        ious.append(iou)
        dices.append(dice)
    return ious, dices


def main():
    all_results = {}

    for region in REGIONS:
        print(f"\n=== {region.upper()} ===")

        with rasterio.open(f"results/predictions/{region}_prediction.tif") as src:
            pred = src.read(1)
        with rasterio.open(f"data/reference_labels/{region}_reference_label.tif") as src:
            target = src.read(1)

        cm = compute_confusion_matrix(pred, target, NUM_CLASSES, IGNORE_VALUE)
        ious, dices = iou_and_dice_from_cm(cm)

        print(f"{'Class':15s} {'IoU':>8s} {'Dice':>8s}")
        for name, iou, dice in zip(CLASS_NAMES, ious, dices):
            print(f"{name:15s} {iou:8.4f} {dice:8.4f}")

        mean_iou = float(np.nanmean(ious))
        mean_dice = float(np.nanmean(dices))
        print(f"\nMean IoU:  {mean_iou:.4f}")
        print(f"Mean Dice: {mean_dice:.4f}")

        all_results[region] = {
            "per_class_iou": dict(zip(CLASS_NAMES, [float(x) for x in ious])),
            "per_class_dice": dict(zip(CLASS_NAMES, [float(x) for x in dices])),
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "confusion_matrix": cm.tolist(),
        }

    all_results["similipal_in_domain_reference"] = {"mean_iou": 0.6659, "mean_dice": 0.7598}

    print("\n" + "=" * 50)
    print("SUMMARY — Domain gap vs. Similipal (in-domain mean IoU: 0.6659)")
    print("=" * 50)
    for region in REGIONS:
        drop = 0.6659 - all_results[region]["mean_iou"]
        drop_pct = (drop / 0.6659) * 100
        print(f"  {region:15s}: mean IoU {all_results[region]['mean_iou']:.4f} "
              f"(drop: {drop:+.4f}, {drop_pct:+.1f}%)")

    with open("results/domain_gap_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results/domain_gap_results.json")


if __name__ == "__main__":
    main()