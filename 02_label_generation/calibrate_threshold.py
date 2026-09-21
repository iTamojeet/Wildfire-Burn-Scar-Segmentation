"""
02_label_generation/calibrate_threshold.py

Scans a range of RdNBR threshold offsets and reports burned-area %
of vegetated land for each, so we can pick the offset that lands
near the ~30% target in one pass instead of repeated manual guesses.
"""

import rasterio
import numpy as np
from scipy.ndimage import zoom

PRE_FIRE_PATH  = "../01_data_acquisition/data/raw/similipal_pre_fire.tif"
POST_FIRE_PATH = "../01_data_acquisition/data/raw/similipal_post_fire.tif"
VEG_MASK_PATH  = "data/reference/vegetation_mask.tif"

NIR_BAND, SWIR_BAND = 8, 12
OFFSETS_TO_TEST = [150, 300, 450, 600, 800, 1000, 1300, 1600, 2000]


def compute_nbr(nir, swir):
    denom = nir.astype(float) + swir.astype(float)
    denom[denom == 0] = np.nan
    return (nir.astype(float) - swir.astype(float)) / denom


def resample_to_match(arr, target_shape):
    zf = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    resized = zoom(arr, zf, order=0)
    return resized[:target_shape[0], :target_shape[1]]


with rasterio.open(PRE_FIRE_PATH) as src:
    pre_nir, pre_swir = src.read(NIR_BAND), src.read(SWIR_BAND)
    target_shape = pre_nir.shape
with rasterio.open(POST_FIRE_PATH) as src:
    post_nir, post_swir = src.read(NIR_BAND), src.read(SWIR_BAND)
with rasterio.open(VEG_MASK_PATH) as src:
    veg_raw = src.read(1)
is_vegetated = resample_to_match(veg_raw, target_shape) == 1

nbr_pre = compute_nbr(pre_nir, pre_swir)
nbr_post = compute_nbr(post_nir, post_swir)
nbr_pre_scaled, nbr_post_scaled = nbr_pre * 1000, nbr_post * 1000
dnbr = nbr_pre_scaled - nbr_post_scaled

denom = np.sqrt(np.abs(nbr_pre_scaled) / 1000)
denom[denom == 0] = np.nan
rdnbr = dnbr / denom

veg_rdnbr = rdnbr[is_vegetated]
veg_total = is_vegetated.sum()

print(f"Vegetated pixel count: {veg_total}")
print(f"RdNBR (vegetated) — min: {np.nanmin(veg_rdnbr):.1f}, max: {np.nanmax(veg_rdnbr):.1f}, "
      f"median: {np.nanmedian(veg_rdnbr):.1f}")
print(f"\n{'Offset':>8} | {'Burned % of vegetated':>22}")
print("-" * 33)
for offset in OFFSETS_TO_TEST:
    unburned_cutoff = 69 + offset
    burned_pct = (veg_rdnbr >= unburned_cutoff).sum() / veg_total * 100
    print(f"{offset:>8} | {burned_pct:>21.2f}%")