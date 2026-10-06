"""
06_domain_gap_analysis/check_siberia_false_positive.py — FIXED

Two bugs in the previous version:
1. Missing nodata masking in the percentile stretch (same bug we
   fixed in verify_rgb_preview.py) - produced a different-looking
   base image than our earlier false-color preview, making this
   non-comparable.
2. Red overlay blended into the base image's own pink/magenta tones,
   making it visually indistinguishable. Switched to bright cyan,
   which doesn't naturally occur in this SWIR2/NIR/Red scheme's
   vegetation/burn colors.
"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("results/predictions/siberia_prediction.tif") as src:
    pred = src.read(1)
with rasterio.open("data/reference_labels/siberia_reference_label.tif") as src:
    ref = src.read(1)

false_positive_high = (ref == 0) & (pred == 3)

with rasterio.open("data/raw/siberia_post_fire.tif") as src:
    fc = np.stack([src.read(b) for b in (12, 8, 4)], axis=-1).astype(float)
    nodata_val = src.nodata

valid_mask = np.ones(fc.shape[:2], dtype=bool)
if nodata_val is not None:
    for i in range(3):
        valid_mask &= (fc[:, :, i] != nodata_val)

for i in range(3):
    band = fc[:, :, i]
    valid_pixels = band[valid_mask]
    lo, hi = np.percentile(valid_pixels, (2, 98))
    band = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)
    band[~valid_mask] = 0
    fc[:, :, i] = band

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(fc)
overlay = np.ma.masked_where(~false_positive_high, np.ones_like(false_positive_high, dtype=float))
ax.imshow(overlay, cmap=plt.matplotlib.colors.ListedColormap(["#00FFFF"]), alpha=0.85)
ax.axis("off")
plt.tight_layout()
plt.savefig("results/siberia_false_positive_overlay_v2.png", dpi=150)
plt.close()
print(f"False positive high_severity pixels: {false_positive_high.sum():,}")
print("Saved results/siberia_false_positive_overlay_v2.png")