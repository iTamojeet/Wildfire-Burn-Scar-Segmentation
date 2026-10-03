"""
06_domain_gap_analysis/acquire_regions.py — SECOND CORRECTION

Fixes:
1. Mediterranean AOI corrected using real coordinates (Limni/Agia Anna/
   Mantoudi) - previous two attempts used wrong coordinates entirely,
   missing the actual burn zone both times.
2. Siberia AOI narrowed to stay within a single UTM zone (51),
   eliminating the cross-zone tile seam affecting the post-fire mosaic.
3. Pantanal: unchanged for now - see diagnostic script below first.
"""

import os
import ee
import geemap

ee.Authenticate()
ee.Initialize(project="n8nlovetppanu")

CLOUD_THRESHOLD = 15
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

REGIONS = {
    "mediterranean": {
        # CHANGED: real coordinates this time. Limni (23.317,38.767),
        # Agia Anna (23.399,38.860), Mantoudi (23.478,38.798) all fall
        # inside this box. Previous two boxes both missed this area
        # entirely (started too far east / too far north).
        "aoi": ee.Geometry.Rectangle([23.20, 38.70, 23.55, 38.95]),
        "pre_fire_window": ("2021-07-20", "2021-08-02"),
        "post_fire_window": ("2021-08-12", "2021-08-25"),
    },
    "siberia": {
        # CHANGED: narrowed from 125.00-126.00 to 125.00-125.80 to stay
        # entirely within UTM zone 51, avoiding the zone-51/52 tile
        # seam visible in the post-fire mosaic.
        "aoi": ee.Geometry.Rectangle([125.00, 61.60, 125.80, 62.00]),
        "pre_fire_window": ("2021-05-10", "2021-05-31"),
        "post_fire_window": ("2021-08-20", "2021-09-10"),
    },
}

OUT_DIR = "data/raw"


def get_mosaic_image(aoi, start_date, end_date, cloud_thresh):
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
    )
    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(f"No images under {cloud_thresh}% cloud cover in {start_date} to {end_date}.")
    ids = collection.aggregate_array("PRODUCT_ID").getInfo()
    clouds = collection.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    print(f"  Compositing {count} scene(s) via per-pixel median:")
    for pid, c in zip(ids, clouds):
        print(f"    {pid} | cloud%: {c:.2f}")
    return collection.median()


def prep_image(img, aoi):
    return img.select(BANDS).clip(aoi).toInt16()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for region_name, config in REGIONS.items():
        print(f"\n=== {region_name.upper()} ===")
        aoi = config["aoi"]
        print(f"Fetching pre-fire window: {config['pre_fire_window']}")
        pre_final = prep_image(get_mosaic_image(aoi, *config["pre_fire_window"], CLOUD_THRESHOLD), aoi)
        print(f"Fetching post-fire window: {config['post_fire_window']}")
        post_final = prep_image(get_mosaic_image(aoi, *config["post_fire_window"], CLOUD_THRESHOLD), aoi)

        pre_path = os.path.join(OUT_DIR, f"{region_name}_pre_fire.tif")
        post_path = os.path.join(OUT_DIR, f"{region_name}_post_fire.tif")
        print(f"Downloading {pre_path}...")
        geemap.download_ee_image(pre_final, filename=pre_path, region=aoi, scale=10, crs="EPSG:4326", max_requests=1, max_cpus=1)
        print(f"Downloading {post_path}...")
        geemap.download_ee_image(post_final, filename=post_path, region=aoi, scale=10, crs="EPSG:4326", max_requests=1, max_cpus=1)
        print(f"{region_name} done.")

    print("\nMediterranean and Siberia re-acquired. Pantanal untouched - see diagnostic.")


if __name__ == "__main__":
    main()