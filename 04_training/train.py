"""
04_training/train.py

Trains U-Net (ResNet34 encoder) on Similipal burn severity patches.
24-channel input (12 pre-fire + 12 post-fire Sentinel-2 bands).
4 output classes (unburned/low/moderate/high) - not_applicable is
excluded via ignore_index, not predicted.

Class weights computed from Stage 3's class_pixel_counts.json via
inverse frequency, since high_severity is only 0.24% of all pixels
and would otherwise be effectively ignored by the loss.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import BurnSeverityDataset, IGNORE_INDEX

TRAIN_MANIFEST = "../03_tiling/data/patches/train/manifest.json"
VAL_MANIFEST   = "../03_tiling/data/patches/val/manifest.json"
CLASS_COUNTS_PATH = "../03_tiling/data/patches/class_pixel_counts.json"

CHECKPOINT_DIR = "checkpoints"
NUM_CLASSES = 4  # unburned, low, moderate, high (not_applicable excluded)
IN_CHANNELS = 24  # 12 pre-fire + 12 post-fire bands

BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = ["unburned", "low_severity", "moderate", "high_severity"]


def compute_class_weights(class_counts_path):
    with open(class_counts_path) as f:
        counts = json.load(f)

    # Exclude not_applicable - it's not a class the model predicts
    relevant = {k: v for k, v in counts.items() if k != "not_applicable"}
    total = sum(relevant.values())
    n_classes = len(relevant)

    # Inverse frequency weighting: rarer classes get higher weight
    weights = []
    for name in CLASS_NAMES:
        count = relevant[name]
        weight = total / (n_classes * count)
        weights.append(weight)

    print("Class weights (inverse frequency):")
    for name, w in zip(CLASS_NAMES, weights):
        print(f"  {name:15s}: {w:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


def compute_iou_per_class(pred, target, num_classes, ignore_index):
    """Accumulates per-class intersection/union for one batch."""
    ious = []
    valid = target != ignore_index
    for c in range(num_classes):
        pred_c = (pred == c) & valid
        target_c = (target == c) & valid
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        ious.append((intersection, union))
    return ious


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    train_ds = BurnSeverityDataset(TRAIN_MANIFEST)
    val_ds = BurnSeverityDataset(VAL_MANIFEST)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # ImageNet weights don't apply - not 3-channel RGB input
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    ).to(DEVICE)

    class_weights = compute_class_weights(CLASS_COUNTS_PATH).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_mean_iou = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        class_intersections = [0] * NUM_CLASSES
        class_unions = [0] * NUM_CLASSES

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                preds = outputs.argmax(dim=1)
                ious = compute_iou_per_class(preds, labels, NUM_CLASSES, IGNORE_INDEX)
                for c, (inter, union) in enumerate(ious):
                    class_intersections[c] += inter
                    class_unions[c] += union

        val_loss /= len(val_ds)

        per_class_iou = [
            (class_intersections[c] / class_unions[c]) if class_unions[c] > 0 else float("nan")
            for c in range(NUM_CLASSES)
        ]
        mean_iou = np.nanmean(per_class_iou)

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        print(f"  Per-class IoU: " + ", ".join(
            f"{name}={iou:.3f}" for name, iou in zip(CLASS_NAMES, per_class_iou)
        ))
        print(f"  Mean IoU: {mean_iou:.4f}")

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  -> New best model saved (mean IoU: {mean_iou:.4f})")

    print(f"\nTraining complete. Best mean IoU: {best_mean_iou:.4f}")


if __name__ == "__main__":
    main()