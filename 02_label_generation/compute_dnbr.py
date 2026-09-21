"""
02_label_generation/compute_dnbr.py

FINAL APPROACH: RdNBR + land-cover mask (proven visually coherent),
with thresholds manually calibrated upward from Miller & Thode
defaults to correct for the seasonal phenology confound this AOI's
Sal forest exhibits. This is a calibration compromise, not a fully
derived correction — documented as a limitation in the Stage 2
README, not hidden.
"""

import os
import rasterio
import numpy as np
from scipy.ndimage import zoom

PRE_FIRE_PATH  = "../01_data_acquisition/data/raw/similipal_pre_fire.tif"
POST_FIRE_PATH = "../01_data_acquisition/data/raw/similipal_post_fire.tif"
VEG_MASK_PATH  = "data/reference/vegetation_mask.tif"

DNBR_OUT     = "data/dnbr/similipal_dnbr.tif"
RDNBR_OUT    = "data/dnbr/similipal_rdnbr.tif"
SEVERITY_OUT = "data/dnbr/similipal_severity_classes_final.tif"

NIR_BAND, SWIR_BAND = 8, 12

# CALIBRATION KNOB: raise this to shrink burned-area extent.
# Start at 0 (= original Miller & Thode values), increase in steps
# of ~50-100 and rerun the overlay check each time.
THRESHOLD_OFFSET = 380  # starting guess — adjust based on overlay result

RDNBR_THRESHOLDS = {
    "unburned":      (-np.inf, 69 + THRESHOLD_OFFSET),
    "low_severity":  (69 + THRESHOLD_OFFSET, 315 + THRESHOLD_OFFSET),
    "moderate":      (315 + THRESHOLD_OFFSET, 640 + THRESHOLD_OFFSET),
    "high_severity": (640 + THRESHOLD_OFFSET, np.inf),
}
SEVERITY_CODES = {
    "unburned": 0, "low_severity": 1, "moderate": 2,
    "high_severity": 3, "not_applicable": 4,
}


def compute_nbr(nir, swir):
    denom = nir.astype(float) + swir.astype(float)
    denom[denom == 0] = np.nan
    return (nir.astype(float) - swir.astype(float)) / denom


def resample_to_match(arr, target_shape):
    zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    resized = zoom(arr, zoom_factors, order=0)
    return resized[:target_shape[0], :target_shape[1]]


def main():
    os.makedirs("data/dnbr", exist_ok=True)

    with rasterio.open(PRE_FIRE_PATH) as src:
        pre_nir, pre_swir = src.read(NIR_BAND), src.read(SWIR_BAND)
        profile = src.profile
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

    severity = np.full(dnbr.shape, SEVERITY_CODES["not_applicable"], dtype=np.uint8)
    for name, (low, high) in RDNBR_THRESHOLDS.items():
        mask = is_vegetated & (rdnbr >= low) & (rdnbr < high)
        severity[mask] = SEVERITY_CODES[name]

    veg_pct = is_vegetated.sum() / is_vegetated.size * 100
    total = severity.size
    print(f"Threshold offset: +{THRESHOLD_OFFSET}")
    print(f"Vegetated pixels: {veg_pct:.2f}%")
    print("\nSeverity class distribution:")
    for name, code in SEVERITY_CODES.items():
        pct = (severity == code).sum() / total * 100
        print(f"  {name:15s} (code {code}): {pct:.2f}%")

    float_profile = profile.copy()
    float_profile.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(DNBR_OUT, "w", **float_profile) as dst:
        dst.write(dnbr.astype(np.float32), 1)
    with rasterio.open(RDNBR_OUT, "w", **float_profile) as dst:
        dst.write(rdnbr.astype(np.float32), 1)

    sev_profile = profile.copy()
    sev_profile.update(dtype=rasterio.uint8, count=1, nodata=255)
    with rasterio.open(SEVERITY_OUT, "w", **sev_profile) as dst:
        dst.write(severity, 1)


if __name__ == "__main__":
    main()