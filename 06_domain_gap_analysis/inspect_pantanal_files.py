"""
06_domain_gap_analysis/inspect_pantanal_files.py

Full diagnostic on BOTH pantanal files at once - shape, bounds, CRS,
file modification time, and per-band stats for pre and post. This
replaces guessing: the previous diagnostic (post-fire only) showed
clean data, but the preview shows a black/white image, which can only
mean the preview read a different file state than the diagnostic did,
or there's a bug in the preview script itself specific to this file.
This will show us definitively which.
"""

import os
import time
import rasterio
import numpy as np

BAND_NAMES = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]

for label, path in [("PRE-FIRE", "data/raw/pantanal_pre_fire.tif"),
                     ("POST-FIRE", "data/raw/pantanal_post_fire.tif")]:
    print(f"\n{'='*20} {label}: {path} {'='*20}")

    if not os.path.exists(path):
        print("  FILE DOES NOT EXIST")
        continue

    mtime = os.path.getmtime(path)
    print(f"  Last modified: {time.ctime(mtime)}")
    print(f"  File size: {os.path.getsize(path) / 1e6:.1f} MB")

    with rasterio.open(path) as src:
        print(f"  Shape: {src.shape} (height, width)")
        print(f"  Band count: {src.count}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Nodata value: {src.nodata}")
        print(f"  Dtype: {src.dtypes[0]}")

        data = src.read()
        for i, name in enumerate(BAND_NAMES[:src.count]):
            band = data[i].astype(float)
            nonzero = band[band > 0]
            pct_nonzero = len(nonzero) / band.size * 100
            if len(nonzero) > 0:
                print(f"    {name:5s}: {pct_nonzero:5.1f}% nonzero | "
                      f"min={nonzero.min():.0f} max={nonzero.max():.0f} mean={nonzero.mean():.0f}")
            else:
                print(f"    {name:5s}: {pct_nonzero:5.1f}% nonzero | ALL ZERO")