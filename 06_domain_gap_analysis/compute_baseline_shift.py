"""
06_domain_gap_analysis/compute_baseline_shift.py

Compares each region's pre-fire NBR distribution against Similipal's,
quantifying the baseline vegetation/reflectance shift that's the
actual hypothesis behind this project: the same severity index means
something different depending on what the land looked like before
the fire.
"""

import rasterio
import numpy as np

REGIONS = {
    "similipal": "../02_label_generation/../01_data_acquisition/data/raw/similipal_pre_fire.tif",
    "pantanal": "data/raw/pantanal_pre_fire.tif",
    "mediterranean": "data/raw/mediterranean_pre_fire.tif",
    "siberia": "data/raw/siberia_pre_fire.tif",
}
NIR_BAND, SWIR_BAND = 8, 12


def compute_nbr(path):
    with rasterio.open(path) as src:
        nir = src.read(NIR_BAND).astype(float)
        swir = src.read(SWIR_BAND).astype(float)
    denom = nir + swir
    denom[denom == 0] = np.nan
    nbr = (nir - swir) / denom
    return nbr[np.isfinite(nbr)]


results = {}
for region, path in REGIONS.items():
    nbr = compute_nbr(path)
    results[region] = {"mean": np.mean(nbr), "median": np.median(nbr), "std": np.std(nbr)}
    print(f"{region:15s}: mean={results[region]['mean']:.3f}  "
          f"median={results[region]['median']:.3f}  std={results[region]['std']:.3f}")

print("\nBaseline NBR shift relative to Similipal (training domain):")
sim_mean = results["similipal"]["mean"]
for region in ["pantanal", "mediterranean", "siberia"]:
    shift = results[region]["mean"] - sim_mean
    print(f"  {region:15s}: {shift:+.3f}")