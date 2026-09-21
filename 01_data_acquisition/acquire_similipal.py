"""
01_data_acquisition/acquire_similipal.py

Downloads pre-fire and post-fire Sentinel-2 imagery for the Similipal
Tiger Reserve, 2021 fire event. Mosaics all scenes intersecting the AOI
within each date window (cleanest scene's pixels take priority) rather
than picking a single image, since the AOI can span multiple MGRS tiles.
"""

import ee
import geemap

# ---------------------------------------------------------------
# 1. Authenticate + Initialize
# ---------------------------------------------------------------
ee.Authenticate()
ee.Initialize(project="n8nlovetppanu")

# ---------------------------------------------------------------
# 2. AOI — approximate bounding box for Similipal Tiger Reserve
# ---------------------------------------------------------------
AOI = ee.Geometry.Rectangle([86.05, 21.75, 86.75, 22.25])

# ---------------------------------------------------------------
# 3. Date windows
# ---------------------------------------------------------------
PRE_FIRE_WINDOW  = ("2021-01-15", "2021-02-05")
POST_FIRE_WINDOW = ("2021-03-20", "2021-04-10")

CLOUD_THRESHOLD = 10  # percent, per CLOUDY_PIXEL_PERCENTAGE metadata

# ---------------------------------------------------------------
# 4. Helper: mosaic all scenes covering the AOI in a window
#    (single-image selection left the AOI partially uncovered when
#    it spanned more than one MGRS tile)
# ---------------------------------------------------------------
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
    print(f"Compositing {count} scene(s) via per-pixel median:")
    for pid, c in zip(ids, clouds):
        print(f"  {pid} | cloud%: {c:.2f}")

    return collection.median()

# ---------------------------------------------------------------
# 5. Fetch pre-fire and post-fire mosaics
# ---------------------------------------------------------------
pre_fire_img = get_mosaic_image(AOI, *PRE_FIRE_WINDOW, CLOUD_THRESHOLD)
post_fire_img = get_mosaic_image(AOI, *POST_FIRE_WINDOW, CLOUD_THRESHOLD)

# ---------------------------------------------------------------
# 6. Select bands needed for dNBR + future multispectral use
# ---------------------------------------------------------------
BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B11", "B12"
]

def prep_image(img, aoi):
    return img.select(BANDS).clip(aoi).toInt16()

pre_fire_final = prep_image(pre_fire_img, AOI)
post_fire_final = prep_image(post_fire_img, AOI)

# ---------------------------------------------------------------
# 7. Download locally via geemap (auto-tiles large exports)
# ---------------------------------------------------------------
geemap.download_ee_image(
    pre_fire_final,
    filename="data/raw/similipal_pre_fire.tif",
    region=AOI,
    scale=10,
    crs="EPSG:4326",
)

geemap.download_ee_image(
    post_fire_final,
    filename="data/raw/similipal_post_fire.tif",
    region=AOI,
    scale=10,
    crs="EPSG:4326",
)

print("Done. Pre-fire and post-fire GeoTIFFs saved to 01_data_acquisition/data/raw/")