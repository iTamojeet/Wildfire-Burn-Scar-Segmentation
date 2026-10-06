"""
06_domain_gap_analysis/diagnose_siberia_post.py
"""
import rasterio
import numpy as np

BAND_NAMES = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]

with rasterio.open("data/raw/siberia_post_fire.tif") as src:
    data = src.read()

print("Band statistics (post-fire Siberia):")
for i, name in enumerate(BAND_NAMES):
    band = data[i].astype(float)
    band = band[band > 0]
    print(f"  {name:5s}: min={band.min():.0f} max={band.max():.0f} "
          f"mean={band.mean():.0f} p2={np.percentile(band,2):.0f} p98={np.percentile(band,98):.0f}")