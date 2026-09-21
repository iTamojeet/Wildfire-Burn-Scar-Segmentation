# Stage 3: Tiling

Extracts fixed-size patches from the Stage 1/2 rasters for model
training, and reports class distribution to inform Stage 4's loss
function design.

## Patch design

- **Input stack:** 24 channels per patch — 12 Sentinel-2 bands
  (pre-fire) concatenated with 12 bands (post-fire), so the model
  learns the pre/post relationship directly rather than only seeing
  a pre-computed index.
- **Size:** 256×256, stride 128 (50% overlap) — standard for U-Net-
  style segmentation; overlap increases the chance of capturing
  enough labeled fire pixels per patch given fire is a minority of
  the total AOI.
- **Label:** corresponding 256×256 crop from Stage 2's
  `similipal_severity_classes_final.tif`.

## Filtering

Patches are dropped if either:
- More than 5% of the patch is nodata (checked on the pre-fire
  stack's first band)
- More than 50% of the patch is `not_applicable` (non-vegetated
  land, per the Stage 2 WorldCover mask) — keeps the training set
  focused on land the model should actually learn to classify,
  rather than diluting patches with bare rock/agriculture

1,485 patches survived filtering from the full AOI.

## has_fire flagging: what changed and why

The original stratification check used `.any()` — flagging a patch
as `has_fire` if it contained even a single burned pixel out of
65,536. This produced 100% has_fire patches, which made the
stratified split meaningless (every patch qualified, so there was
nothing to stratify against).

**Root cause diagnosis:** rather than a bug, this reflected something
real about the AOI — burned area is spread across multiple valley
corridors (visible in the Stage 2 overlay) rather than one tight
blob, so with 50% overlap nearly every patch touches at least a few
burned pixels somewhere.

**Fix:** switched to a burned-pixel *fraction* threshold (≥5% of the
patch) instead of "any burned pixel." This still resulted in 92.9%
of patches qualifying as has_fire — but pixel-level analysis (below)
confirmed this is consistent with genuinely widespread, thin fire
distribution, not a broken check: median burned fraction per patch
is only 22.5%, meaning most patches are still majority-unburned even
though they contain some fire.

## Class distribution (pixel-level, across all 1,485 patches)

| Class | % of pixels | Pixel count |
|---|---|---|
| unburned | 59.62% | 58,022,862 |
| low_severity | 20.25% | 19,703,779 |
| moderate | 5.97% | 5,812,073 |
| high_severity | 0.24% | 229,518 |
| not_applicable | 13.93% | 13,552,728 |

**Implication for Stage 4:** `high_severity` at 0.24% of all pixels
is a genuinely hard minority class. Plain unweighted cross-entropy
would let the model achieve strong aggregate accuracy while
effectively ignoring it. Stage 4 uses inverse-frequency class
weighting (computed from `class_pixel_counts.json`, saved by this
stage) to address this, and reports per-class IoU rather than only
aggregate accuracy so this class's performance is visible rather
than hidden in an average.

## Train/val split

- 15% validation, stratified on the (fraction-based) `has_fire` flag
  so validation isn't accidentally starved of fire examples.
- Train: 1,262 patches (1,173 with fire)
- Val: 223 patches (207 with fire)
- Random seed 42, via `sklearn.model_selection.train_test_split`
  (installed as `pip install scikit-learn` — the `sklearn` PyPI
  package name is deprecated/broken, `scikit-learn` is the correct
  install target though the import name stays `sklearn`).

## Outputs

- `data/patches/train/` — 1,262 patches as `.npy` pairs
  (`*_img.npy` 24×256×256 int16, `*_lbl.npy` 256×256 uint8) +
  `manifest.json`
- `data/patches/val/` — 223 patches, same format + `manifest.json`
- `data/patches/class_pixel_counts.json` — pixel counts per class
  across all patches, used directly by Stage 4 for loss weighting

## Scripts

- `tile_dataset.py` — full pipeline: load rasters, extract patches,
  filter, compute diagnostics, split, save