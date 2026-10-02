"""
06_domain_gap_analysis/acquire_regions.py

Fetches pre-fire and post-fire Sentinel-2 imagery for three zero-shot
test regions (Pantanal, Mediterranean, Siberian boreal). No severity
labeling needed here — these are evaluation-only, testing the
Similipal-trained model's generalization, not training on them.

Reuses Stage 1's mosaic approach (median composite across all
qualifying scenes in each window) since each AOI may span multiple
MGRS tiles, same as Similipal did.
"""

import os
import ee
import geemap

ee.Authenticate()
ee.Initialize(project="n8nlovetppanu")

CLOUD_THRESHOLD = 15  # slightly relaxed vs Similipal's 10 - these are eval-only, not label-critical
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

REGIONS = {
    "pantanal": {
        "aoi": ee.Geometry.Rectangle([-57.10, -17.55, -56.50, -17.05]),  # Encontro das Aguas SP, MT
        "pre_fire_window": ("2020-07-01", "2020-07-20"),
        "post_fire_window": ("2020-10-01", "2020-10-20"),
    },
    "mediterranean": {
        "aoi": ee.Geometry.Rectangle([23.45, 38.75, 23.90, 38.95]),  # North Evia, Greece
        "pre_fire_window": ("2021-07-20", "2021-08-02"),
        "post_fire_window": ("2021-08-12", "2021-08-25"),
    },
    "siberia": {
        "aoi": ee.Geometry.Rectangle([125.00, 61.60, 126.00, 62.00]),  # Gorny Ulus, Central Yakutia
        "pre_fire_window": ("2021-06-01", "2021-06-20"),
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
        raise ValueError(
            f"No images under {cloud_thresh}% cloud cover in {start_date} to "
            f"{end_date}. Consider widening the date window or relaxing the threshold."
        )

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
        pre_img = get_mosaic_image(aoi, *config["pre_fire_window"], CLOUD_THRESHOLD)
        pre_final = prep_image(pre_img, aoi)

        print(f"Fetching post-fire window: {config['post_fire_window']}")
        post_img = get_mosaic_image(aoi, *config["post_fire_window"], CLOUD_THRESHOLD)
        post_final = prep_image(post_img, aoi)

        pre_path = os.path.join(OUT_DIR, f"{region_name}_pre_fire.tif")
        post_path = os.path.join(OUT_DIR, f"{region_name}_post_fire.tif")

        print(f"Downloading {pre_path}...")
        geemap.download_ee_image(
            pre_final, filename=pre_path, region=aoi, scale=10, crs="EPSG:4326",
            max_requests=1, max_cpus=1,
        )

        print(f"Downloading {post_path}...")
        geemap.download_ee_image(
            post_final, filename=post_path, region=aoi, scale=10, crs="EPSG:4326",
            max_requests=1, max_cpus=1,
        )

        print(f"{region_name} done.")

    print("\nAll three regions acquired. Saved to data/raw/")


if __name__ == "__main__":
    main()