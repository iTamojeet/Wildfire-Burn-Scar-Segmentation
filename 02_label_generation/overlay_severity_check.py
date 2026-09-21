"""
02_label_generation/overlay_severity_check.py
Points at the final calibrated RdNBR severity output.
"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

POST_FIRE_PATH = "../01_data_acquisition/data/raw/similipal_post_fire.tif"
SEVERITY_PATH  = "data/dnbr/similipal_severity_classes_final.tif"

with rasterio.open(POST_FIRE_PATH) as src:
    rgb = np.stack([src.read(b) for b in (4, 3, 2)], axis=-1).astype(float)
for i in range(3):
    lo, hi = np.percentile(rgb[:, :, i], (2, 98))
    rgb[:, :, i] = np.clip((rgb[:, :, i] - lo) / (hi - lo + 1e-6), 0, 1)

with rasterio.open(SEVERITY_PATH) as src:
    severity = src.read(1)

colors = ["none", "#ffff99", "#ff6600", "#990000", "#888888"]
cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(rgb)
ax.imshow(np.ma.masked_where(severity == 0, severity), cmap=cmap, vmin=0, vmax=4, alpha=0.55)
ax.axis("off")
plt.tight_layout()
plt.savefig("data/dnbr/severity_overlay_final.png", dpi=150)
plt.show()