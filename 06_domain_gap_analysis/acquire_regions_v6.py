"""
06_domain_gap_analysis/acquire_regions_v6.py

CHANGED: instead of independently picking "lowest cloud" per window
(which let pre/post land on different MGRS tiles for Mediterranean -
41.7% vs 100.1% coverage), this determines the correct tile ONCE via
actual geometric coverage of the AOI, then pins both pre- and
post-fire queries to that single tile. Guarantees consistency by
construction instead of detecting mismatches after the fact.

Pantanal already succeeded in the previous run (97.6% coverage both
dates, same tile) - left in REGIONS for completeness but you can
comment it out to skip re-downloading.
"""

import os
import ee
import geemap

ee.Authenticate()
ee.Initialize(project="n8nlovetppanu")

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
CLOUD_THRESHOLD = 20
MIN_COVERAGE = 0.95

REGIONS = {
    # "pantanal": {
    #     "aoi": ee.Geometry.Rectangle([-56.90, -17.40, -56.68, -17.18]),
    #     "pre_fire_window": ("2020-07-01", "2020-07-20"),
    #     "post_fire_window": ("2020-12-01", "2021-01-15"),
    # },
    "mediterranean": {
        "aoi": ee.Geometry.Rectangle([23.36, 38.82, 23.50, 38.92]),
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


def determine_best_tile(aoi, start_date, end_date, cloud_thresh):
    """
    Finds the single MGRS tile with the greatest geometric overlap
    with the AOI, using scene footprints (no image data needed yet) -
    cheap, and guarantees pre/post selection later pulls from the
    SAME tile instead of independently picking per-date.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
    )
    tile_ids = collection.aggregate_array("MGRS_TILE").distinct().getInfo()
    if not tile_ids:
        raise ValueError(f"No scenes found at all in {start_date} to {end_date}.")

    best_tile, best_coverage = None, -1
    for tile in tile_ids:
        sample_img = collection.filter(ee.Filter.eq("MGRS_TILE", tile)).first()
        footprint = sample_img.geometry()
        overlap_area = footprint.intersection(aoi, 1).area(1).getInfo()
        aoi_area = aoi.area(1).getInfo()
        coverage = overlap_area / aoi_area
        print(f"    Tile {tile}: {coverage*100:.1f}% geometric overlap with AOI")
        if coverage > best_coverage:
            best_tile, best_coverage = tile, coverage

    print(f"  -> Pinning to tile {best_tile} ({best_coverage*100:.1f}% coverage)")
    return best_tile


def get_best_scene_from_tile(aoi, start_date, end_date, cloud_thresh, tile_id):
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("MGRS_TILE", tile_id))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    count = collection.size().getInfo()
    if count == 0:
        raise ValueError(f"No scenes from tile {tile_id} under {cloud_thresh}% cloud in {start_date} to {end_date}.")

    best = collection.first()
    props = best.getInfo()["properties"]
    print(f"  Selected: {props['PRODUCT_ID']} | cloud%: {props['CLOUDY_PIXEL_PERCENTAGE']:.2f} | tile: {tile_id}")
    return best


def check_coverage(img, aoi, band="B8"):
    valid_count = img.select(band).reduceRegion(
        reducer=ee.Reducer.count(), geometry=aoi, scale=10, maxPixels=1e9
    ).get(band).getInfo()
    total_pixels = ee.Geometry(aoi).area().divide(100).getInfo()
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

        print(f"Determining correct tile from pre-fire window: {config['pre_fire_window']}")
        tile_id = determine_best_tile(aoi, *config["pre_fire_window"], CLOUD_THRESHOLD)

        print(f"Fetching pre-fire scene (tile {tile_id}):")
        pre_img = get_best_scene_from_tile(aoi, *config["pre_fire_window"], CLOUD_THRESHOLD, tile_id)
        pre_coverage = check_coverage(pre_img, aoi)

        print(f"Fetching post-fire scene (same tile {tile_id}):")
        post_img = get_best_scene_from_tile(aoi, *config["post_fire_window"], CLOUD_THRESHOLD, tile_id)
        post_coverage = check_coverage(post_img, aoi)

        if pre_coverage < MIN_COVERAGE or post_coverage < MIN_COVERAGE:
            raise RuntimeError(
                f"{region_name}: coverage below {MIN_COVERAGE*100:.0f}% threshold "
                f"(pre={pre_coverage*100:.1f}%, post={post_coverage*100:.1f}%) even "
                f"after pinning to tile {tile_id}. AOI may genuinely not fit in one tile - shrink it."
            )

        print(f"  Checks passed (tile: {tile_id}, both >= {MIN_COVERAGE*100:.0f}% coverage)")

        pre_final = prep_image(pre_img, aoi)
        post_final = prep_image(post_img, aoi)

        pre_path = os.path.join(OUT_DIR, f"{region_name}_pre_fire.tif")
        post_path = os.path.join(OUT_DIR, f"{region_name}_post_fire.tif")

        geemap.download_ee_image(pre_final, filename=pre_path, region=aoi, scale=10, crs="EPSG:4326", max_requests=1, max_cpus=1)
        geemap.download_ee_image(post_final, filename=post_path, region=aoi, scale=10, crs="EPSG:4326", max_requests=1, max_cpus=1)
        print(f"{region_name} done.")

    print("\nAll regions re-acquired, all checks passed.")


if __name__ == "__main__":
    main()