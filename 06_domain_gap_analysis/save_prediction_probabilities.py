"""
06_domain_gap_analysis/save_prediction_probabilities.py

Re-runs zero-shot inference but saves per-class SOFTMAX PROBABILITIES
(4-band float32 raster) instead of just the final argmax class. Needed
for the recalibration step, which adjusts class probabilities using
each region's baseline NBR shift before making the final decision.
"""

import os
import sys
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

CHECKPOINT_PATH = "../04_training/checkpoints/best_model.pt"
REGIONS = ["pantanal", "mediterranean", "siberia"]
NUM_CLASSES = 4
IN_CHANNELS = 24
PATCH_SIZE = 256
STRIDE = 128
REFLECTANCE_SCALE = 10000.0
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "results/probabilities"


def load_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=IN_CHANNELS, classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def run_region(model, region_name):
    print(f"\n=== {region_name.upper()} ===")
    with rasterio.open(f"data/raw/{region_name}_pre_fire.tif") as src:
        pre = src.read()
        profile = src.profile
    with rasterio.open(f"data/raw/{region_name}_post_fire.tif") as src:
        post = src.read()
    with rasterio.open(f"data/reference/{region_name}_vegetation_mask.tif") as src:
        veg_mask = src.read(1)

    _, H, W = pre.shape
    prob_sum = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=np.uint8)

    with torch.no_grad():
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                pre_p = pre[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                post_p = post[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                stacked = np.concatenate([pre_p, post_p], axis=0).astype(np.float32) / REFLECTANCE_SCALE
                tensor = torch.from_numpy(stacked).unsqueeze(0).to(DEVICE)
                probs = F.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
                prob_sum[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += probs
                coverage[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    covered = coverage > 0
    for c in range(NUM_CLASSES):
        prob_sum[c][covered] /= coverage[covered]
    prob_sum[:, veg_mask == 0] = 0  # zero out non-vegetated

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{region_name}_probabilities.tif")
    out_profile = profile.copy()
    out_profile.update(count=NUM_CLASSES, dtype=rasterio.float32, nodata=None)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(prob_sum)
    print(f"Saved {out_path}")


def main():
    model = load_model()
    for region in REGIONS:
        run_region(model, region)


if __name__ == "__main__":
    main()