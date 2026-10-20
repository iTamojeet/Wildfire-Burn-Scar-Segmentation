"""
06_domain_gap_analysis/decompose_gap_by_label_rule.py   (NEW - not yet run)

Separates two things that are currently mixed in the Stage 6 numbers:

  (A) LABEL-PROTOCOL gap : training labels used RdNBR thresholds 69/315/640 + 380,
      the zero-shot reference labels used the literature defaults (+0).
  (B) GENERALISATION gap : how well the network reproduces ITS OWN training
      rule on new biomes.

For each region it scores the (uncorrected) model prediction against
    ref_default  = RdNBR defaults (+0)         -> reproduces the Stage 6 numbers
    ref_sim_rule = RdNBR defaults + 380         -> the rule the network was trained on
and also reports the agreement between the two references themselves
(an upper bound on how well ANY model of the Similipal rule can do against ref_default).

Needs: results/probabilities/<region>_probabilities.tif (save_prediction_probabilities.py).
"""
import numpy as np
import rasterio

REGIONS = ["pantanal", "mediterranean", "siberia"]
NIR, SWIR = 8, 12
BASE = np.array([69, 315, 640], dtype=float)
SIM_OFFSET = 380
IGNORE = 255


def rdnbr(pre_path, post_path):
    def nbr(path):
        with rasterio.open(path) as s:
            nir, swir = s.read(NIR).astype(float), s.read(SWIR).astype(float)
        d = nir + swir
        d[d == 0] = np.nan
        return 1000 * (nir - swir) / d
    a, b = nbr(pre_path), nbr(post_path)
    den = np.sqrt(np.abs(a) / 1000)
    den[den == 0] = np.nan
    return (a - b) / den


def label(r, veg, offset):
    out = np.full(r.shape, IGNORE, np.uint8)
    ok = (veg == 1) & np.isfinite(r)
    out[ok] = np.digitize(r[ok], BASE + offset).astype(np.uint8)
    return out


def mean_iou(pred, ref):
    ok = (pred != IGNORE) & (ref != IGNORE)
    p, t = pred[ok], ref[ok]
    ious = []
    for c in range(4):
        tp = ((p == c) & (t == c)).sum()
        fp = ((p == c) & (t != c)).sum()
        fn = ((p != c) & (t == c)).sum()
        ious.append(tp / (tp + fp + fn) if (tp + fp + fn) else np.nan)
    return float(np.nanmean(ious)), ious


print(f"{'region':14s} {'pred vs default':>16s} {'pred vs sim-rule':>17s} {'default vs sim-rule':>20s}   burned% default / sim-rule")
for r in REGIONS:
    with rasterio.open(f"results/probabilities/{r}_probabilities.tif") as s:
        probs = s.read()
    with rasterio.open(f"data/reference/{r}_vegetation_mask.tif") as s:
        veg = s.read(1)
    pred = probs.argmax(0).astype(np.uint8)
    pred[probs.sum(0) == 0] = IGNORE
    rd = rdnbr(f"data/raw/{r}_pre_fire.tif", f"data/raw/{r}_post_fire.tif")
    ref0, ref1 = label(rd, veg, 0), label(rd, veg, SIM_OFFSET)
    m0, _ = mean_iou(pred, ref0)
    m1, _ = mean_iou(pred, ref1)
    mr, _ = mean_iou(ref1, ref0)
    b0 = ((ref0 >= 1) & (ref0 <= 3)).sum() / max(((ref0 != IGNORE)).sum(), 1) * 100
    b1 = ((ref1 >= 1) & (ref1 <= 3)).sum() / max(((ref1 != IGNORE)).sum(), 1) * 100
    print(f"{r:14s} {m0:16.4f} {m1:17.4f} {mr:20.4f}   {b0:5.1f}% / {b1:5.1f}%")

print("\nInterpretation: if 'pred vs sim-rule' >> 'pred vs default', most of the measured gap is the labelling "
      "protocol; the shortfall of 'pred vs sim-rule' from the in-domain 0.666 is the genuine generalisation loss.")