"""
03_tiling/tile_dataset.py

Extracts fixed-size patches from the Stage 1/2 rasters for training.
Input stack per patch: 24 channels (12 pre-fire bands + 12 post-fire
bands). Label: corresponding severity class patch from Stage 2.

Patches with too much not_applicable/nodata are dropped. Remaining
patches are split into train/val, stratified on burned fraction.
Also reports pixel-level class distribution across all patches,
since that (not patch-level has_fire%) is what determines whether
Stage 4 needs class-weighted loss.
"""

import os
import json
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split

PRE_FIRE_PATH  = "../01_data_acquisition/data/raw/similipal_pre_fire.tif"
POST_FIRE_PATH = "../01_data_acquisition/data/raw/similipal_post_fire.tif"
SEVERITY_PATH  = "../02_label_generation/data/dnbr/similipal_severity_classes_final.tif"

OUT_DIR = "data/patches"
PATCH_SIZE = 256
STRIDE = 128  # 50% overlap

NOT_APPLICABLE_CODE = 4
MAX_NOT_APPLICABLE_FRACTION = 0.5  # drop patch if >50% is non-vegetated
MAX_NODATA_FRACTION = 0.05         # drop patch if >5% is nodata

HAS_FIRE_THRESHOLD = 0.05  # at least 5% of patch must be burned

VAL_FRACTION = 0.15
RANDOM_SEED = 42

CLASS_NAMES = {0: "unburned", 1: "low_severity", 2: "moderate",
               3: "high_severity", 4: "not_applicable"}


def load_rasters():
    with rasterio.open(PRE_FIRE_PATH) as src:
        pre = src.read()  # (12, H, W)
        nodata_val = src.nodata
    with rasterio.open(POST_FIRE_PATH) as src:
        post = src.read()  # (12, H, W)
    with rasterio.open(SEVERITY_PATH) as src:
        severity = src.read(1)  # (H, W)
        severity_nodata = src.nodata

    return pre, post, severity, nodata_val, severity_nodata


def extract_patches(pre, post, severity, nodata_val, severity_nodata):
    _, H, W = pre.shape
    patches = []

    for y in range(0, H - PATCH_SIZE + 1, STRIDE):
        for x in range(0, W - PATCH_SIZE + 1, STRIDE):
            pre_patch = pre[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            post_patch = post[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            sev_patch = severity[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            if nodata_val is not None:
                nodata_frac = (pre_patch[0] == nodata_val).sum() / pre_patch[0].size
                if nodata_frac > MAX_NODATA_FRACTION:
                    continue

            na_frac = (sev_patch == NOT_APPLICABLE_CODE).sum() / sev_patch.size
            if na_frac > MAX_NOT_APPLICABLE_FRACTION:
                continue

            stacked = np.concatenate([pre_patch, post_patch], axis=0).astype(np.int16)

            burned_pixel_count = ((sev_patch >= 1) & (sev_patch <= 3)).sum()
            burned_fraction = burned_pixel_count / sev_patch.size
            has_fire = burned_fraction >= HAS_FIRE_THRESHOLD

            patches.append({
                "image": stacked,
                "label": sev_patch.astype(np.uint8),
                "has_fire": bool(has_fire),
                "burned_fraction": float(burned_fraction),
                "y": y, "x": x,
            })

    return patches


def save_split(patches, split_name, out_dir):
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    manifest = []
    for i, p in enumerate(patches):
        img_path = os.path.join(split_dir, f"{split_name}_{i:05d}_img.npy")
        lbl_path = os.path.join(split_dir, f"{split_name}_{i:05d}_lbl.npy")
        np.save(img_path, p["image"])
        np.save(lbl_path, p["label"])
        manifest.append({
            "index": i,
            "image_path": img_path,
            "label_path": lbl_path,
            "has_fire": p["has_fire"],
            "burned_fraction": p["burned_fraction"],
            "source_y": p["y"],
            "source_x": p["x"],
        })

    with open(os.path.join(split_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def print_pixel_class_distribution(patches, label):
    """
    NEW: pixel-level class distribution across a set of patches.
    This is what actually determines whether Stage 4 needs
    class-weighted loss — patch-level has_fire% only tells you
    how many patches contain SOME fire, not how much of the actual
    pixel mass belongs to each class.
    """
    all_labels = np.concatenate([p["label"].flatten() for p in patches])
    print(f"\nPixel-level class distribution ({label}):")
    counts = {}
    for code, name in CLASS_NAMES.items():
        count = (all_labels == code).sum()
        pct = count / all_labels.size * 100
        counts[name] = int(count)
        print(f"  {name:15s}: {pct:6.2f}%  ({count:,} px)")
    return counts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading rasters...")
    pre, post, severity, nodata_val, severity_nodata = load_rasters()

    print("Extracting patches...")
    patches = extract_patches(pre, post, severity, nodata_val, severity_nodata)
    print(f"Total patches after filtering: {len(patches)}")

    burned_fractions = [p["burned_fraction"] for p in patches]
    print(f"\nBurned-fraction-per-patch stats:")
    print(f"  min: {min(burned_fractions):.4f}, max: {max(burned_fractions):.4f}")
    print(f"  median: {np.median(burned_fractions):.4f}")
    print(f"  patches with <1% burned:  {sum(f < 0.01 for f in burned_fractions)}")
    print(f"  patches with <5% burned:  {sum(f < 0.05 for f in burned_fractions)}")
    print(f"  patches with >=5% burned: {sum(f >= 0.05 for f in burned_fractions)}")

    fire_count = sum(p["has_fire"] for p in patches)
    print(f"\nPatches classified has_fire (>= {HAS_FIRE_THRESHOLD*100:.0f}% burned): "
          f"{fire_count} ({fire_count / len(patches) * 100:.1f}%)")

    # --- NEW: pixel-level distribution, the number that actually
    # determines Stage 4 loss weighting ---
    class_pixel_counts = print_pixel_class_distribution(patches, "all patches")

    stratify_labels = [p["has_fire"] for p in patches]
    train_patches, val_patches = train_test_split(
        patches, test_size=VAL_FRACTION, random_state=RANDOM_SEED,
        stratify=stratify_labels
    )

    print(f"\nTrain: {len(train_patches)} patches "
          f"({sum(p['has_fire'] for p in train_patches)} with fire)")
    print(f"Val:   {len(val_patches)} patches "
          f"({sum(p['has_fire'] for p in val_patches)} with fire)")

    save_split(train_patches, "train", OUT_DIR)
    save_split(val_patches, "val", OUT_DIR)

    # Save class pixel counts for Stage 4 to use directly for loss weighting
    with open(os.path.join(OUT_DIR, "class_pixel_counts.json"), "w") as f:
        json.dump(class_pixel_counts, f, indent=2)

    print(f"\nSaved to {OUT_DIR}/train/ and {OUT_DIR}/val/")
    print(f"Class pixel counts saved to {OUT_DIR}/class_pixel_counts.json (for Stage 4 loss weighting)")


if __name__ == "__main__":
    main()