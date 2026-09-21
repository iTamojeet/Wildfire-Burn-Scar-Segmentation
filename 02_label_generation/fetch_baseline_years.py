"""
02_label_generation/fetch_baseline_years.py

Fetches pre/post NBR for non-fire reference years (2019, 2020, 2022)
to build a seasonal baseline. Purpose: isolate the seasonal Sal-forest
leaf-drop signal (present every year, fire or not) from the actual
2021 fire signal. Every prior correction attempt (MODIS/FIRMS reference
masking, RdNBR) addressed spatial heterogeneity or reference-pixel
selection, but never this — the root confound identified at the very
first overlay check back in Stage 2.
"""
import os
import ee
import geemap

ee.Authenticate()
ee.Initialize(project="n8nlovetppanu")

AOI = ee.Geometry.Rectangle([86.05, 21.75, 86.75, 22.25])
os.makedirs("data/baseline", exist_ok=True)

REFERENCE_YEARS = [2019, 2020, 2022]  # non-fire years; excludes 2021, 2024 (known fires)
NIR_BAND, SWIR_BAND = "B8", "B12"


def get_median_nbr(year, start_md, end_md):
    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(AOI)
        .filterDate(f"{year}-{start_md}", f"{year}-{end_md}")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )
    composite = coll.median()
    nir, swir = composite.select(NIR_BAND), composite.select(SWIR_BAND)
    return nir.subtract(swir).divide(nir.add(swir))


pre_nbrs, post_nbrs = [], []
for year in REFERENCE_YEARS:
    pre_nbrs.append(get_median_nbr(year, "01-15", "02-05"))
    post_nbrs.append(get_median_nbr(year, "03-20", "04-10"))

baseline_pre = ee.ImageCollection(pre_nbrs).median()
baseline_post = ee.ImageCollection(post_nbrs).median()
baseline_dnbr = baseline_pre.subtract(baseline_post).multiply(1000)  # matches *1000 scale used elsewhere

geemap.download_ee_image(
    baseline_dnbr.clip(AOI).toFloat(),
    filename="data/baseline/similipal_baseline_dnbr.tif",
    region=AOI, scale=10, crs="EPSG:4326",
    max_requests=1, max_cpus=1,
)
print("Saved: data/baseline/similipal_baseline_dnbr.tif")
print("This is the EXPECTED seasonal dNBR from natural leaf-drop alone (non-fire years).")