# Stage 1: Data Acquisition

Downloads pre-fire and post-fire Sentinel-2 imagery for Similipal
Tiger Reserve, covering the 2021 fire event.

## Date windows
- Pre-fire: Jan 15 – Feb 5, 2021 (safely before Feb 11 ignition onset)
- Post-fire: Mar 20 – Apr 10, 2021 (after Mar 10 rain/containment,
  before pre-monsoon showers)

## AOI
Approximate bounding box for Similipal Tiger Reserve
(86.05–86.75°E, 21.75–22.25°N), Mayurbhanj district, Odisha.
Not the precise reserve boundary — a rectangle. Refine with the
WDPA (Protected Planet) polygon if exact reserve-boundary stats
are needed later.

## Mosaic approach
The AOI spans more than one Sentinel-2 MGRS tile (T45QUE / T45QVE).
A single least-cloudy-scene selection left part of the AOI as
nodata. Fixed by compositing all qualifying scenes (cloud % < 10)
in each date window via per-pixel median, which also avoids the
hard tile-seam visible when using a "cleanest scene wins" mosaic
(different tiles/sensor passes have slightly different atmospheric
conditions, causing a visible color discontinuity at tile
boundaries with a naive mosaic).

**Caveat:** the post-fire window (Mar 20 – Apr 10) overlaps a period
when parts of Similipal were still actively burning (fire activity
continued into mid-March per FSI data). Median compositing across
this window could blend burned/unburned states at a given pixel if
different satellite passes caught it before/after that pixel's
burn. This is a standard cost of median compositing during an
active-change period — not a labeling error if it shows up later
as ambiguous severity at some pixel edges.

## Band resolution: B12 (SWIR2) resampling
Sentinel-2 bands are captured at three native resolutions: B2/B3/
B4/B8 at 10m, B5/B6/B7/B8A/B11/B12 at 20m, B1/B9/B10 at 60m.

All bands were exported at a uniform 10m scale via
`geemap.download_ee_image(..., scale=10)`. This means B12 (SWIR2)
— used alongside B8 (NIR) for NBR/RdNBR — is resampled from its
native 20m resolution up to 10m, not natively captured at 10m.
The resampling is smoothing/interpolation, not new information;
any dNBR/RdNBR values computed from this data inherit a slightly
softened SWIR signal relative to a native 10m sensor. Standard
practice for NBR-based studies (SWIR bands are lower-resolution
across nearly all multispectral satellites), documented here so
it isn't mistaken for native resolution downstream.

## Outputs
- `data/raw/similipal_pre_fire.tif`
- `data/raw/similipal_post_fire.tif`

12 bands each (B1–B9, B11, B12), int16, EPSG:4326, 10m.