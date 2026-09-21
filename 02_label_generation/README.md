# Stage 2: Label Generation (Burn Severity Classification)

Computes burn severity labels for Similipal's 2021 fire from the
Stage 1 Sentinel-2 pre/post-fire tiles, using RdNBR restricted to
vegetated land.

## Method

**Index:** RdNBR (Relative dNBR, Miller & Thode 2007), not standard
dNBR. RdNBR normalizes each pixel's dNBR by the square root of that
same pixel's own pre-fire NBR magnitude, correcting for canopy-
density heterogeneity across a landscape (dense forest vs. sparse/
degraded vegetation don't dry down or burn the same way, so a single
scene-wide correction can't represent both).

**Vegetation mask:** ESA WorldCover v200, tree-cover + shrubland
classes only. Non-vegetated pixels (bare rock, water, agriculture)
are excluded from severity classification entirely and labeled
`not_applicable` rather than forced into a severity bin — burn
severity indices aren't meaningful on land with nothing to combust,
and early attempts that let non-vegetated pixels through a dNBR
fallback misclassified agricultural/seasonal reflectance change as
fire.

**Classes:**
| Code | Class | Threshold (RdNBR, calibrated) |
|---|---|---|
| 0 | unburned | < 449 |
| 1 | low_severity | 449 – 695 |
| 2 | moderate | 695 – 1020 |
| 3 | high_severity | ≥ 1020 |
| 4 | not_applicable | non-vegetated (WorldCover mask) |

(Thresholds = Miller & Thode defaults of 69/315/640 + a calibrated
offset of 380 — see below.)

## Why the thresholds are calibrated, not the literature defaults

Similipal's Sal forest undergoes natural dry-season leaf-drop across
the same Jan–Apr window as the fire itself. Raw RdNBR/dNBR cannot
distinguish that seasonal defoliation from fire-driven canopy loss,
and literature-default thresholds (calibrated on Sierra Nevada mixed-
conifer forest in the original Miller & Thode study, not tropical
dry deciduous forest) produced implausible results — as high as 91%
of vegetated land reading as "burned" — because the seasonal signal
alone exceeded those absolute cutoffs.

**Two more rigorous correction methods were attempted first and
rejected:**

1. **Scalar background-drift correction** — used an independent
   reference source (MODIS MCD64A1 burned-area, later combined with
   FIRMS active-fire hotspots) to identify pixels confirmed unburned,
   then subtracted their average dNBR (the "natural drift") from the
   whole scene. Rejected: MCD64A1 has a known omission gap for
   surface fires under closed canopy — almost exactly Similipal's
   fire behavior — so the "confirmed unburned" reference pool was
   itself contaminated with real burned forest. Visual inspection
   showed the correction erasing fire signal from the AOI's most
   visually obvious burn scar. Mean vs. median made negligible
   difference, indicating systematic bias in the reference pool
   rather than outlier contamination.

2. **Multi-year seasonal anomaly baseline** — built a 3-year
   (2019/2020/2022) non-fire median composite of the same Jan–Apr
   NBR trajectory, then classified 2021 as an anomaly relative to
   that baseline (excess dNBR beyond the normal seasonal pattern).
   This correctly targeted the actual confound, but differencing two
   independently-built multi-year composites compounded per-pixel
   registration/atmospheric noise from both, producing scattered
   terrain-illumination speckle across the whole scene (colored
   pixels tracing ridgelines/valleys regardless of fire, not a
   coherent burn perimeter) — even after a 5×5 majority filter.
   Abandoned as producing an unusable, noise-dominated result.

**Final approach:** manually calibrated the RdNBR threshold offset
(+380) via a scan (`calibrate_threshold.py`) until total burned area
(~28.7% of vegetated AOI) matched the reported fire extent (~1/3 of
the reserve) and the resulting severity map visually traced the
known burn scar without false-positives on bare/agricultural land.
**This is a calibration compromise verified by RGB overlay, not a
physically derived correction.** Treat class boundaries — especially
moderate vs. high severity — as approximate; the offset was tuned
for overall extent and visual coherence, not per-class accuracy.

## Known remaining limitations

- **RdNBR denominator instability:** RdNBR divides by
  `sqrt(|NBR_prefire|)`. Some WorldCover-flagged "vegetated" pixels
  still have low pre-fire NBR (sparse/degraded canopy), so this can
  produce isolated extreme RdNBR values (observed range: -32,893 to
  +18,458 on the vegetated subset, median 316.9). Did not visibly
  affect the final overlay but is a known noise source in the raw
  RdNBR raster (`similipal_rdnbr.tif`) — be cautious using that raw
  raster directly without going through the classified severity map.
- **AOI vs. reserve boundary:** the AOI is a bounding-box rectangle,
  not the true reserve/buffer-division boundary, so burned-area
  percentages are relative to a larger area than the reported "~1/3
  of the reserve" figure. The 28.7% figure is a reasonable match but
  not a like-for-like comparison.
- **Post-fire window overlap with active burning** (see Stage 1
  README) — median compositing across an active-fire period can
  blend burned/unburned states at some pixel edges.

## Outputs

- `data/dnbr/similipal_dnbr.tif` — raw dNBR (float32, ×1000 scale)
- `data/dnbr/similipal_rdnbr.tif` — RdNBR (float32)
- `data/dnbr/similipal_severity_classes_final.tif` — final label
  raster (uint8, 5 classes as above) — **this is the Stage 3 input**
- `data/reference/vegetation_mask.tif` — ESA WorldCover binary mask,
  10m, resampled to match Sentinel-2 grid

## Scripts

- `compute_dnbr.py` — main pipeline (dNBR → RdNBR → classification)
- `calibrate_threshold.py` — one-off scan used to find the +380
  offset; not part of the regular pipeline, kept for reproducibility
- `overlay_severity_check.py` — visual QA (RGB + severity overlay)