"""
06_domain_gap_analysis/generate_reference_labels.py

Lightweight RdNBR reference labels for zero-shot regions, using
literature-default Miller & Thode thresholds (no calibration search
like Similipal's Stage 2). These serve as a comparison baseline for
scoring zero-shot predictions - not a claim of ground-truth accuracy.
This is a deliberate scope decision: Similipal's labels needed full
rigor because the model trained on them; these only need to be
reasonable enough to measure relative performance drop.
"""

import os
import rasterio
import numpy as np

REGIONS = ["pantanal", "mediterranean", "siberia"]
NIR_BAND, SWIR_BAND = 8, 12

RDNBR_THRESHOLDS = {
    0: (-np.inf, 69), 1: (69, 315), 2: (315, 640), 3: (640, np.inf),
}


def compute_rdnbr(pre_path, post_path):
    with rasterio.open(pre_path) as src:
        pre_nir = src.read(NIR_BAND).astype(float)
        pre_swir = src.read(SWIR_BAND).astype(float)
    with rasterio.open(post_path) as src:
        post_nir = src.read(NIR_BAND).astype(float)
        post_swir = src.read(SWIR_BAND).astype(float)

    def nbr(nir, swir):
        d = nir + swir
        d[d == 0] = np.nan
        return (nir - swir) / d

    nbr_pre, nbr_post = nbr(pre_nir, pre_swir) * 1000, nbr(post_nir, post_swir) * 1000
    dnbr = nbr_pre - nbr_post
    denom = np.sqrt(np.abs(nbr_pre) / 1000)
    denom[denom == 0] = np.nan
    return dnbr / denom


def main():
    os.makedirs("data/reference_labels", exist_ok=True)
    for region in REGIONS:
        print(f"\n=== {region.upper()} ===")
        rdnbr = compute_rdnbr(f"data/raw/{region}_pre_fire.tif", f"data/raw/{region}_post_fire.tif")

        with rasterio.open(f"data/reference/{region}_vegetation_mask.tif") as src:
            veg = src.read(1)
        with rasterio.open(f"data/raw/{region}_pre_fire.tif") as src:
            profile = src.profile

        label = np.full(rdnbr.shape, 255, dtype=np.uint8)
        for code, (lo, hi) in RDNBR_THRESHOLDS.items():
            mask = (veg == 1) & (rdnbr >= lo) & (rdnbr < hi)
            label[mask] = code

        out_profile = profile.copy()
        out_profile.update(count=1, dtype=rasterio.uint8, nodata=255)
        out_path = f"data/reference_labels/{region}_reference_label.tif"
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(label, 1)

        total = label.size
        for code, name in enumerate(["unburned", "low", "moderate", "high"]):
            pct = (label == code).sum() / total * 100
            print(f"  {name:12s}: {pct:.2f}%")
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()