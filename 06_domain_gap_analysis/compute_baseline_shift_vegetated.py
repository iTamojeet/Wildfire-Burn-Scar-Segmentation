"""
06_domain_gap_analysis/compute_baseline_shift_vegetated.py   (NEW - not yet run)

Same idea as compute_baseline_shift.py, but the pre-fire NBR statistics are
computed ONLY over vegetated pixels (the WorldCover masks used for training /
inference). The original script averaged over every valid pixel, so water
(~1/3 of the Evia scene) and bare land contaminated the "baseline shift".

Reports mean, median, SD and the shift vs Similipal for both mean and median.
"""
import numpy as np
import rasterio
from scipy.ndimage import zoom

NIR_BAND, SWIR_BAND = 8, 12
REGIONS = {
    "similipal": ("../01_data_acquisition/data/raw/similipal_pre_fire.tif",
                  "../02_label_generation/data/reference/vegetation_mask.tif"),
    "pantanal": ("data/raw/pantanal_pre_fire.tif", "data/reference/pantanal_vegetation_mask.tif"),
    "mediterranean": ("data/raw/mediterranean_pre_fire.tif", "data/reference/mediterranean_vegetation_mask.tif"),
    "siberia": ("data/raw/siberia_pre_fire.tif", "data/reference/siberia_vegetation_mask.tif"),
}


def resample_to_match(arr, shape):
    z = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    return zoom(arr, z, order=0)[:shape[0], :shape[1]]


def veg_nbr(pre_path, mask_path):
    with rasterio.open(pre_path) as src:
        nir = src.read(NIR_BAND).astype(float)
        swir = src.read(SWIR_BAND).astype(float)
        nodata = src.nodata
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    mask = resample_to_match(mask, nir.shape) == 1
    denom = nir + swir
    valid = mask & (denom > 0)
    if nodata is not None:
        valid &= (nir != nodata) & (swir != nodata)
    return ((nir - swir)[valid] / denom[valid]), valid.mean()


stats = {}
for name, (pre, mask) in REGIONS.items():
    nbr, frac = veg_nbr(pre, mask)
    stats[name] = (nbr.mean(), np.median(nbr), nbr.std())
    print(f"{name:14s} vegetated px: {100 * frac:5.1f}%  mean={stats[name][0]:.3f} "
          f"median={stats[name][1]:.3f} sd={stats[name][2]:.3f}")

print("\nShift vs Similipal (vegetated pixels only):")
for name in ["pantanal", "mediterranean", "siberia"]:
    print(f"  {name:14s} mean shift {stats[name][0] - stats['similipal'][0]:+.3f}   "
          f"median shift {stats[name][1] - stats['similipal'][1]:+.3f}")