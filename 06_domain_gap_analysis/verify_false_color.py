"""
06_domain_gap_analysis/verify_false_color.py

Switches Siberia to a SWIR2/NIR/Red false-color composite instead of
true-color RGB - standard for burn-scar visual inspection, and avoids
the narrow-blue-channel stretch distortion seen in the true-color
preview. Healthy vegetation renders bright red/pink; burned ground
renders dark brown/black.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

FALSE_COLOR_BAND_INDICES = (12, 8, 4)  # B12 (SWIR2), B8 (NIR), B4 (Red) -> displayed as R,G,B


def load_false_color(path, percentile_clip=2):
    with rasterio.open(path) as src:
        img = np.stack([src.read(b) for b in FALSE_COLOR_BAND_INDICES], axis=-1).astype(float)
        nodata_val = src.nodata

    valid_mask = np.ones(img.shape[:2], dtype=bool)
    if nodata_val is not None:
        for i in range(3):
            valid_mask &= (img[:, :, i] != nodata_val)

    for i in range(3):
        band = img[:, :, i]
        valid_pixels = band[valid_mask]
        low, high = np.percentile(valid_pixels, (percentile_clip, 100 - percentile_clip))
        band = np.clip((band - low) / (high - low + 1e-6), 0, 1)
        band[~valid_mask] = 0
        img[:, :, i] = band

    return img


pre_fc = load_false_color("data/raw/siberia_pre_fire.tif")
post_fc = load_false_color("data/raw/siberia_post_fire.tif")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(pre_fc)
axes[0].set_title("Siberia — Pre-fire (false color: SWIR2/NIR/Red)")
axes[0].axis("off")
axes[1].imshow(post_fc)
axes[1].set_title("Siberia — Post-fire (false color: SWIR2/NIR/Red)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("data/raw/siberia_false_color.png", dpi=150)
plt.close()
print("Saved data/raw/siberia_false_color.png")