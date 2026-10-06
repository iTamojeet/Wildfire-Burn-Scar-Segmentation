"""
06_domain_gap_analysis/zero_shot_inference.py

Runs the Similipal-trained model (Stage 4 best checkpoint) on each
of the three zero-shot regions. No fine-tuning, no retraining - pure
domain-gap test. Tiles each region the same way Stage 3 tiled
Similipal, runs inference per patch, and stitches predictions back
into a full severity map per region.
"""

import os
import sys
import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp

sys.path.append("../04_training")

CHECKPOINT_PATH = "../04_training/checkpoints/best_model.pt"
REGIONS = ["pantanal", "mediterranean", "siberia"]

NUM_CLASSES = 4
IN_CHANNELS = 24
PATCH_SIZE = 256
STRIDE = 128
REFLECTANCE_SCALE = 10000.0
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUT_DIR = "results/predictions"


def load_model():
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=IN_CHANNELS, classes=NUM_CLASSES,
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def run_inference_on_region(model, region_name):
    print(f"\n=== {region_name.upper()} ===")

    with rasterio.open(f"data/raw/{region_name}_pre_fire.tif") as src:
        pre = src.read()
        profile = src.profile
    with rasterio.open(f"data/raw/{region_name}_post_fire.tif") as src:
        post = src.read()
    with rasterio.open(f"data/reference/{region_name}_vegetation_mask.tif") as src:
        veg_mask = src.read(1)

    _, H, W = pre.shape
    prediction = np.full((H, W), 255, dtype=np.uint8)  # 255 = not inferred (edge/non-vegetated)
    logits_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    coverage_count = np.zeros((H, W), dtype=np.uint8)

    with torch.no_grad():
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                pre_patch = pre[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                post_patch = post[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                stacked = np.concatenate([pre_patch, post_patch], axis=0).astype(np.float32) / REFLECTANCE_SCALE

                tensor = torch.from_numpy(stacked).unsqueeze(0).to(DEVICE)
                output = model(tensor).squeeze(0).cpu().numpy()  # (NUM_CLASSES, 256, 256)

                logits_sum[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += output
                coverage_count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    # Average overlapping predictions (stride < patch size means overlap)
    covered = coverage_count > 0
    for c in range(NUM_CLASSES):
        logits_sum[c][covered] /= coverage_count[covered]

    predicted_class = logits_sum.argmax(axis=0).astype(np.uint8)
    prediction[covered] = predicted_class[covered]

    # Apply vegetation mask - same restriction as training
    prediction[veg_mask == 0] = 255

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{region_name}_prediction.tif")
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=rasterio.uint8, nodata=255)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(prediction, 1)

    print(f"Saved {out_path}")

    total = prediction.size
    class_names = ["unburned", "low_severity", "moderate", "high_severity", "not_applicable/edge"]
    for code, name in enumerate([0, 1, 2, 3, 255]):
        pct = (prediction == name).sum() / total * 100
        print(f"  {class_names[code]:20s}: {pct:.2f}%")


def main():
    model = load_model()
    for region_name in REGIONS:
        run_inference_on_region(model, region_name)


if __name__ == "__main__":
    main()