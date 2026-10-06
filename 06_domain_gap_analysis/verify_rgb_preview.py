"""
06_domain_gap_analysis/verify_rgb_preview.py — FIXED

BUG FOUND: the percentile stretch was computed over the full array
including nodata (-32768) pixels. Even a small nodata fraction at
the AOI edge drags the 2nd-percentile calculation to a huge negative
number, compressing all real data into a tiny sliver near the top of
the output range - producing a washed-out near-white image. This was
likely affecting earlier previews too, not just this one.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

REGIONS = ["pantanal", "mediterranean", "siberia"]
RGB_BAND_INDICES = (4, 3, 2)


def load_rgb(path, percentile_clip=2):
    with rasterio.open(path) as src:
        rgb = np.stack([src.read(b) for b in RGB_BAND_INDICES], axis=-1).astype(float)
        nodata_val = src.nodata

    # NEW: build a mask excluding nodata pixels BEFORE computing percentiles
    valid_mask = np.ones(rgb.shape[:2], dtype=bool)
    if nodata_val is not None:
        for i in range(3):
            valid_mask &= (rgb[:, :, i] != nodata_val)

    for i in range(3):
        band = rgb[:, :, i]
        valid_pixels = band[valid_mask]
        low, high = np.percentile(valid_pixels, (percentile_clip, 100 - percentile_clip))
        band = np.clip((band - low) / (high - low + 1e-6), 0, 1)
        band[~valid_mask] = 0  # nodata rendered as black, same as before, but no longer skews the stretch
        rgb[:, :, i] = band

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