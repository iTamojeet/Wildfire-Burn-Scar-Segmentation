"""
06_domain_gap_analysis/acquire_regions_v4.py — FIXED coverage check
"""

import os
import ee
import geemap

ee.Authenticate()
ee.Initialize(project="n8nlovetppanu")

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
CLOUD_THRESHOLD = 20

REGIONS = {
    "pantanal": {
        "aoi": ee.Geometry.Rectangle([-56.95, -17.45, -56.65, -17.15]),
        "pre_fire_window": ("2020-07-01", "2020-07-20"),
        "post_fire_window": ("2020-12-01", "2021-01-15"),
    },
    "mediterranean": {
        "aoi": ee.Geometry.Rectangle([23.27, 38.74, 23.53, 38.98]),
        "pre_fire_window": ("2021-07-15", "2021-08-02"),
        "post_fire_window": ("2021-08-12", "2021-09-10"),
    },
    "siberia": {
        "aoi": ee.Geometry.Rectangle([125.25, 61.68, 125.55, 61.92]),
        "pre_fire_window": ("2021-05-01", "2021-06-20"),
        "post_fire_window": ("2021-08-15", "2021-09-20"),
    },
}

OUT_DIR = "data/raw"


def get_best_single_scene(aoi, start_date, end_date, cloud_thresh):
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(f"No scenes under {cloud_thresh}% cloud in {start_date} to {end_date}.")

    best = collection.first()
    props = best.getInfo()["properties"]
    print(f"  Selected: {props['PRODUCT_ID']} | cloud%: {props['CLOUDY_PIXEL_PERCENTAGE']:.2f}")
    return best


def check_coverage(img, aoi, band="B8"):
    """Fraction of the AOI covered by valid (non-null) pixels."""
    valid_count = img.select(band).reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=aoi, scale=10, maxPixels=1e9
    ).get(band).getInfo()

    total_pixels = ee.Geometry(aoi).area().divide(100).getInfo()  # 10m x 10m pixels
    coverage = valid_count / total_pixels
    print(f"  Coverage check: {coverage*100:.1f}% of AOI has valid data")
    return coverage


def prep_image(img, aoi):
    return img.select(BANDS).clip(aoi).toInt16()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for region_name, config in REGIONS.items():
        print(f"\n=== {region_name.upper()} ===")
        aoi = config["aoi"]

        print(f"Pre-fire window: {config['pre_fire_window']}")
        pre_img = get_best_single_scene(aoi, *config["pre_fire_window"], CLOUD_THRESHOLD)
        check_coverage(pre_img, aoi)
        pre_final = prep_image(pre_img, aoi)

        print(f"Post-fire window: {config['post_fire_window']}")
        post_img = get_best_single_scene(aoi, *config["post_fire_window"], CLOUD_THRESHOLD)
        check_coverage(post_img, aoi)
        post_final = prep_image(post_img, aoi)

        pre_path = os.path.join(OUT_DIR, f"{region_name}_pre_fire.tif")
        post_path = os.path.join(OUT_DIR, f"{region_name}_post_fire.tif")

        geemap.download_ee_image(pre_final, filename=pre_path, region=aoi, scale=10, crs="EPSG:4326", max_requests=1, max_cpus=1)
        geemap.download_ee_image(post_final, filename=post_path, region=aoi, scale=10, crs="EPSG:4326", max_requests=1, max_cpus=1)
        print(f"{region_name} done.")

    print("\nAll three regions re-acquired with single-scene strategy.")


if __name__ == "__main__":
    main()