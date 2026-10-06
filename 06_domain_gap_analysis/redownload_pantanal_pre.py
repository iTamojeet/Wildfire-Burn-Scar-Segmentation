"""
06_domain_gap_analysis/redownload_pantanal_pre.py

Pre-fire file was corrupted (0.5MB, only B1 populated, rest zero) -
an interrupted download from the original acquisition run, never
caught because download_ee_image doesn't verify completeness. Same
AOI/tile/dates as before (those were already correct) - just
re-fetching the one broken file.
"""

import os
import ee
import geemap

ee.Authenticate()
ee.Initialize(project="n8nlovetppanu")

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
AOI = ee.Geometry.Rectangle([-56.90, -17.40, -56.68, -17.18])
PRE_FIRE_WINDOW = ("2020-07-01", "2020-07-20")
CLOUD_THRESHOLD = 20

collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI)
    .filterDate(*PRE_FIRE_WINDOW)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESHOLD))
    .sort("CLOUDY_PIXEL_PERCENTAGE")
)
best = collection.first()
props = best.getInfo()["properties"]
print(f"Selected: {props['PRODUCT_ID']} | cloud%: {props['CLOUDY_PIXEL_PERCENTAGE']:.2f}")

final = best.select(BANDS).clip(AOI).toInt16()

out_path = "data/raw/pantanal_pre_fire.tif"
geemap.download_ee_image(final, filename=out_path, region=AOI, scale=10, crs="EPSG:4326", max_requests=1, max_cpus=1)

# Verify it actually worked this time before declaring success
import rasterio
with rasterio.open(out_path) as src:
    data = src.read()
    for i in range(12):
        nonzero_pct = (data[i] != 0).sum() / data[i].size * 100
        if nonzero_pct < 90:
            print(f"  WARNING: band {i+1} only {nonzero_pct:.1f}% nonzero - may still be incomplete!")
    print(f"File size: {os.path.getsize(out_path)/1e6:.1f} MB (expect ~80+ MB for a complete file)")

print("Done.")