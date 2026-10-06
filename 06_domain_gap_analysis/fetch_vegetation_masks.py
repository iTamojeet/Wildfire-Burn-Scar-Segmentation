"""
06_domain_gap_analysis/fetch_vegetation_masks.py — CORRECTED

Pantanal is wetland, not forest — reusing Similipal's tree-cover-only
mask excluded most of its real vegetation (grassland, herbaceous
wetland), producing a misleadingly high "not_applicable" fraction and
hiding real burned area from the model entirely. Vegetation classes
now set per-region based on actual biome composition.
"""

import os
import ee
import geemap

ee.Authenticate()
ee.Initialize(project="n8nlovetppanu")

# WorldCover classes: 10=Tree cover, 20=Shrubland, 30=Grassland, 90=Herbaceous wetland
REGIONS = {
    "pantanal": {
        "aoi": ee.Geometry.Rectangle([-56.90, -17.40, -56.68, -17.18]),
        "veg_classes": [10, 20, 30, 90],  # forest + grassland + wetland - matches real Pantanal mosaic
    },
    "mediterranean": {
        "aoi": ee.Geometry.Rectangle([23.36, 38.82, 23.50, 38.92]),
        "veg_classes": [10, 20, 30],  # forest + shrubland + grassland (Mediterranean maquis/garrigue)
    },
    "siberia": {
        "aoi": ee.Geometry.Rectangle([125.25, 61.68, 125.55, 61.92]),
        "veg_classes": [10, 20],  # boreal taiga - tree cover + shrubland as before
    },
}

OUT_DIR = "data/reference"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

    for region_name, config in REGIONS.items():
        print(f"\n=== {region_name.upper()} ===")
        aoi = config["aoi"]

        is_vegetated = ee.Image(0)
        for cls in config["veg_classes"]:
            is_vegetated = is_vegetated.Or(worldcover.eq(cls))

        out_path = os.path.join(OUT_DIR, f"{region_name}_vegetation_mask.tif")
        geemap.download_ee_image(
            is_vegetated.clip(aoi).toUint8(),
            filename=out_path, region=aoi, scale=10, crs="EPSG:4326",
            max_requests=1, max_cpus=1,
        )
        print(f"Saved {out_path} (classes: {config['veg_classes']})")

    print("\nAll vegetation masks re-fetched with region-appropriate classes.")


if __name__ == "__main__":
    main()