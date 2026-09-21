"""
05_evaluation/evaluate.py

Loads the best Stage 4 checkpoint and runs full evaluation on the
held-out validation set: per-class IoU/Dice, a confusion matrix, and
a handful of visual examples (RGB + ground truth + prediction side
by side) so results can be sanity-checked visually, not just trusted
as numbers.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

sys.path.append("../04_training")
from dataset import BurnSeverityDataset, IGNORE_INDEX, REFLECTANCE_SCALE

CHECKPOINT_PATH = "../04_training/checkpoints/best_model.pt"
VAL_MANIFEST = "../03_tiling/data/patches/val/manifest.json"
OUT_DIR = "results"

NUM_CLASSES = 4
IN_CHANNELS = 24
CLASS_NAMES = ["unburned", "low_severity", "moderate", "high_severity"]
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_VISUAL_EXAMPLES = 8  # how many val patches to render as images


def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def compute_confusion_matrix(preds, targets, num_classes, ignore_index):
    """Accumulates a confusion matrix across all pixels, all batches."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = targets != ignore_index
    p = preds[valid].cpu().numpy()
    t = targets[valid].cpu().numpy()
    for pred_c in range(num_classes):
        for true_c in range(num_classes):
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


def rgb_from_stacked(image_tensor):
    """
    image is (24, H, W): bands 0-11 = pre-fire (B1..B12), 12-23 = post-fire.
    Band order per Stage 1: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12
    B4=index3(red), B3=index2(green), B2=index1(blue) -> post-fire RGB
    """
    post_fire = image_tensor[12:].numpy()  # (12, H, W)
    rgb = np.stack([post_fire[3], post_fire[2], post_fire[1]], axis=-1).astype(float)
    for i in range(3):
        lo, hi = np.percentile(rgb[:, :, i], (2, 98))
        rgb[:, :, i] = np.clip((rgb[:, :, i] - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb


def save_visual_examples(model, val_ds, out_dir, n_examples):
    os.makedirs(out_dir, exist_ok=True)

    # Prefer patches with meaningful fire content for visual inspection
    with open(VAL_MANIFEST) as f:
        manifest = json.load(f)
    fire_indices = [i for i, e in enumerate(manifest) if e.get("has_fire")]
    chosen = fire_indices[:n_examples] if len(fire_indices) >= n_examples else range(n_examples)

    colors = ["#2ca02c", "#ffff99", "#ff6600", "#990000"]  # unburned green for contrast here
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    for idx in chosen:
        image, label = val_ds[idx]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(DEVICE)).argmax(dim=1).squeeze(0).cpu()

        rgb = rgb_from_stacked(image)
        label_np = label.numpy().copy()
        pred_np = pred.numpy().copy()

        # Mask out ignore_index pixels for display
        label_display = np.ma.masked_where(label_np == IGNORE_INDEX, label_np)
        pred_display = np.ma.masked_where(label_np == IGNORE_INDEX, pred_np)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("Post-fire RGB")
        axes[0].axis("off")

        axes[1].imshow(label_display, cmap=cmap, vmin=0, vmax=3)
        axes[1].set_title("Ground truth")
        axes[1].axis("off")

        axes[2].imshow(pred_display, cmap=cmap, vmin=0, vmax=3)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"example_{idx:05d}.png"), dpi=100)
        plt.close()

    print(f"Saved {len(chosen)} visual examples to {out_dir}/")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading model from {CHECKPOINT_PATH} on {DEVICE}...")
    model = load_model()

    val_ds = BurnSeverityDataset(VAL_MANIFEST)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False)

    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            cm = compute_confusion_matrix(preds, labels, NUM_CLASSES, IGNORE_INDEX)
            total_cm += cm

    ious, dices = iou_and_dice_from_cm(total_cm)

    print("\n=== Final Evaluation Results (held-out validation set) ===")
    print(f"{'Class':15s} {'IoU':>8s} {'Dice':>8s}")
    for name, iou, dice in zip(CLASS_NAMES, ious, dices):
        print(f"{name:15s} {iou:8.4f} {dice:8.4f}")
    print(f"\nMean IoU:  {np.nanmean(ious):.4f}")
    print(f"Mean Dice: {np.nanmean(dices):.4f}")

    print("\nConfusion matrix (rows=ground truth, cols=predicted):")
    print(f"{'':15s}" + "".join(f"{n:>15s}" for n in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:15s}" + "".join(f"{total_cm[i,j]:>15,}" for j in range(NUM_CLASSES)))

    # Save numeric results
    results = {
        "per_class_iou": dict(zip(CLASS_NAMES, ious)),
        "per_class_dice": dict(zip(CLASS_NAMES, dices)),
        "mean_iou": float(np.nanmean(ious)),
        "mean_dice": float(np.nanmean(dices)),
        "confusion_matrix": total_cm.tolist(),
    }
    with open(os.path.join(OUT_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nGenerating visual examples...")
    save_visual_examples(model, val_ds, os.path.join(OUT_DIR, "visual_examples"), NUM_VISUAL_EXAMPLES)

    print(f"\nDone. Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()