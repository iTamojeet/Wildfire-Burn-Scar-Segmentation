"""
06_domain_gap_analysis/verify_rgb_preview.py

Visual sanity check for all three zero-shot regions: RGB composite
preview of pre-fire and post-fire tiles. Same purpose as Stage 1's
check - catch cloud/haze/smoke contamination that scene-wide cloud
percentage metadata can miss over a specific AOI.

Siberia is the specific risk here: active-fire smoke isn't always
flagged by CLOUDY_PIXEL_PERCENTAGE (smoke has different spectral
properties than cloud), and the post-fire window ran up to 14% cloud
on individual scenes.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

REGIONS = ["pantanal", "mediterranean", "siberia"]
RGB_BAND_INDICES = (4, 3, 2)  # R, G, B (B4, B3, B2 - 1-indexed for rasterio)


def load_rgb(path, percentile_clip=2):
    with rasterio.open(path) as src:
        rgb = np.stack([src.read(b) for b in RGB_BAND_INDICES], axis=-1).astype(float)

    for i in range(3):
        band = rgb[:, :, i]
        low, high = np.percentile(band, (percentile_clip, 100 - percentile_clip))
        rgb[:, :, i] = np.clip((band - low) / (high - low + 1e-6), 0, 1)

    return rgb


def main():
    for region in REGIONS:
        pre_path = f"data/raw/{region}_pre_fire.tif"
        post_path = f"data/raw/{region}_post_fire.tif"

        pre_rgb = load_rgb(pre_path)
        post_rgb = load_rgb(post_path)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(pre_rgb)
        axes[0].set_title(f"{region.capitalize()} — Pre-fire")
        axes[0].axis("off")

        axes[1].imshow(post_rgb)
        axes[1].set_title(f"{region.capitalize()} — Post-fire")
        axes[1].axis("off")

        plt.tight_layout()
        out_path = f"data/raw/{region}_rgb_preview.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()