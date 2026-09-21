"""
02_label_generation/fetch_landcover_mask.py

Fetches ESA WorldCover 10m land cover to restrict severity classification
to actual vegetated land. The dNBR fallback for non-vegetated pixels was
misclassifying agricultural/bare land as "burned" — those land types
undergo seasonal reflectance change unrelated to fire, and burn severity
indices aren't designed to score them anyway (nothing there to combust).
"""
import os
import ee
import geemap

ee.Authenticate()
ee.Initialize(project="n8nlovetppanu")

AOI = ee.Geometry.Rectangle([86.05, 21.75, 86.75, 22.25])
os.makedirs("data/reference", exist_ok=True)

worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
# WorldCover classes: 10=Tree cover, 20=Shrubland — both count as vegetated
is_vegetated = worldcover.eq(10).Or(worldcover.eq(20))

geemap.download_ee_image(
    is_vegetated.clip(AOI).toUint8(),
    filename="data/reference/vegetation_mask.tif",
    region=AOI,
    scale=10,
    crs="EPSG:4326",
    max_requests=1,
    max_cpus=1,
)
print("Saved: data/reference/vegetation_mask.tif (1=tree/shrub cover, 0=other)")