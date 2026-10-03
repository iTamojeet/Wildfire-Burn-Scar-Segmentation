"""
06_domain_gap_analysis/acquire_pantanal_v3.py

Adds AOT (Aerosol Optical Thickness) filtering alongside cloud%.
Diagnostic on the Nov post-fire mosaic showed anomalous SWIR values
(B11/B12 max >12,000) consistent with lingering active-fire thermal
contamination, plus a broad haze-driven color cast - neither of which
CLOUDY_PIXEL_PERCENTAGE catches, since it measures cloud, not smoke/
aerosol or fire-affected pixels.
"""

import os
import ee
import geemap

ee.Authenticate()
ee.Initialize(project="n8nlovetppanu")

AOI = ee.Geometry.Rectangle([-57.10, -17.55, -56.50, -17.05])
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
CLOUD_THRESHOLD = 15
AOT_THRESHOLD = 150  # AOT band is scaled x1000; ~0.15 is a reasonable "low haze" cutoff

# Pushed further into the wet season to maximize distance from any
# lingering fire/smoke activity, combined with the new AOT filter
POST_FIRE_WINDOW = ("2020-12-01", "2020-12-20")


def get_mosaic_image(aoi, start_date, end_date):
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESHOLD))
        .filter(ee.Filter.lt("AOT_RETRIEVAL_ACCURACY", AOT_THRESHOLD))  # fallback if AOT band stat unavailable at scene level
    )

    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(f"No images passed cloud+AOT filters in {start_date} to {end_date}.")

    ids = collection.aggregate_array("PRODUCT_ID").getInfo()
    clouds = collection.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    print(f"Compositing {count} scene(s):")
    for pid, c in zip(ids, clouds):
        print(f"  {pid} | cloud%: {c:.2f}")

    return collection.median()


def main():
    os.makedirs("data/raw", exist_ok=True)
    print(f"Fetching Pantanal post-fire: {POST_FIRE_WINDOW}")
    img = get_mosaic_image(AOI, *POST_FIRE_WINDOW)
    final = img.select(BANDS).clip(AOI).toInt16()

    out_path = "data/raw/pantanal_post_fire.tif"
    geemap.download_ee_image(final, filename=out_path, region=AOI, scale=10, crs="EPSG:4326", max_requests=1, max_cpus=1)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()