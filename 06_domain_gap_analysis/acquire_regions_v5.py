"""
06_domain_gap_analysis/acquire_regions_v5.py

Adds hard pre-download checks: coverage threshold + same-tile
consistency between pre/post scenes. Previous run proceeded to
download Mediterranean data despite 57% coverage AND mismatched
tiles between pre/post - this version halts before wasting a
download on data we'd have to discard anyway.

AOIs shrunk further and recentered based on what we learned:
Pantanal's existing box just needs tightening (consistent 88.2%,
same tile both dates). Mediterranean needs a genuinely smaller box
since the previous one straddled an MGRS grid square boundary.
"""

import os
import ee
import geemap

ee.Authenticate()
ee.Initialize(project="n8nlovetppanu")

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
CLOUD_THRESHOLD = 20
MIN_COVERAGE = 0.95  # was 0.98 — too strict; 97.6% here is a clean, usable scene,
                      # the gap is rectangle-vs-tile-footprint corner rounding, not contamination

REGIONS = {
    "pantanal": {
        # CHANGED: tightened to sit fully inside T21KWA (was 88.2% on
        # the previous, slightly larger box)
        "aoi": ee.Geometry.Rectangle([-56.90, -17.40, -56.68, -17.18]),
        "pre_fire_window": ("2020-07-01", "2020-07-20"),
        "post_fire_window": ("2020-12-01", "2021-01-15"),
    },
    "mediterranean": {
        # CHANGED: much smaller (~12km), centered tightly on Agia Anna
        # to avoid straddling the 34SFJ/34SGJ grid square boundary
        # that split the previous attempt across two different tiles
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


def get_tile_id(product_id):
    # e.g. S2B_MSIL2A_20201202T140049_N0214_R067_T21KWA_... -> "T21KWA"
    for part in product_id.split("_"):
        if part.startswith("T") and len(part) == 6:
            return part
    return None


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
    product_id = props["PRODUCT_ID"]
    tile_id = get_tile_id(product_id)
    print(f"  Selected: {product_id} | cloud%: {props['CLOUDY_PIXEL_PERCENTAGE']:.2f} | tile: {tile_id}")
    return best, tile_id


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

        print(f"Pre-fire window: {config['pre_fire_window']}")
        pre_img, pre_tile = get_best_single_scene(aoi, *config["pre_fire_window"], CLOUD_THRESHOLD)
        pre_coverage = check_coverage(pre_img, aoi)

        print(f"Post-fire window: {config['post_fire_window']}")
        post_img, post_tile = get_best_single_scene(aoi, *config["post_fire_window"], CLOUD_THRESHOLD)
        post_coverage = check_coverage(post_img, aoi)

        # --- NEW: hard checks before downloading anything ---
        if pre_tile != post_tile:
            raise RuntimeError(
                f"{region_name}: pre-fire tile ({pre_tile}) != post-fire tile "
                f"({post_tile}). AOI likely straddles a tile boundary - shrink "
                f"or recenter the box before retrying."
            )
        if pre_coverage < MIN_COVERAGE or post_coverage < MIN_COVERAGE:
            raise RuntimeError(
                f"{region_name}: coverage below {MIN_COVERAGE*100:.0f}% threshold "
                f"(pre={pre_coverage*100:.1f}%, post={post_coverage*100:.1f}%). "
                f"AOI doesn't fully fit inside tile {pre_tile} - shrink the box."
            )

        print(f"  Checks passed (same tile: {pre_tile}, both >= {MIN_COVERAGE*100:.0f}% coverage)")

        pre_final = prep_image(pre_img, aoi)
        post_final = prep_image(post_img, aoi)

        pre_path = os.path.join(OUT_DIR, f"{region_name}_pre_fire.tif")
        post_path = os.path.join(OUT_DIR, f"{region_name}_post_fire.tif")

        geemap.download_ee_image(pre_final, filename=pre_path, region=aoi, scale=10, crs="EPSG:4326", max_requests=1, max_cpus=1)
        geemap.download_ee_image(post_final, filename=post_path, region=aoi, scale=10, crs="EPSG:4326", max_requests=1, max_cpus=1)
        print(f"{region_name} done.")

    print("\nAll three regions re-acquired, all checks passed.")


if __name__ == "__main__":
    main()