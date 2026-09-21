"""
01_data_acquisition/verify_rgb_preview.py

Quick visual sanity check: RGB composite preview of pre-fire and
post-fire Similipal tiles, to catch residual cloud/shadow patches
that scene-wide CLOUDY_PIXEL_PERCENTAGE metadata can miss.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

PRE_FIRE_PATH  = "data/raw/similipal_pre_fire.tif"
POST_FIRE_PATH = "data/raw/similipal_post_fire.tif"

# Band order from acquisition script: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12
RGB_BAND_INDICES = (4, 3, 2)  # R, G, B


def load_rgb(path, percentile_clip=2):
    with rasterio.open(path) as src:
        rgb = np.stack([src.read(b) for b in RGB_BAND_INDICES], axis=-1).astype(float)

    for i in range(3):
        band = rgb[:, :, i]
        low, high = np.percentile(band, (percentile_clip, 100 - percentile_clip))
        rgb[:, :, i] = np.clip((band - low) / (high - low + 1e-6), 0, 1)

    return rgb


def main():
    pre_rgb = load_rgb(PRE_FIRE_PATH)
    post_rgb = load_rgb(POST_FIRE_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(pre_rgb)
    axes[0].set_title("Pre-fire (Jan\u2013Feb 2021)")
    axes[0].axis("off")

    axes[1].imshow(post_rgb)
    axes[1].set_title("Post-fire (Mar\u2013Apr 2021)")
    axes[1].axis("off")

    plt.tight_layout()
    out_path = "data/raw/rgb_preview.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved preview to {out_path}")


if __name__ == "__main__":
    main()