# Glossary — Wildfire Burn Scar Segmentation

Concepts used across this project, with what they mean and why they
mattered here specifically.

---

## 1. Satellite Imagery Fundamentals

**Sentinel-2**
A pair of European Space Agency satellites (Sentinel-2A and 2B)
that image the entire Earth's land surface every 5 days, capturing
13 spectral bands at 10m, 20m, or 60m resolution depending on the
band. Used here because it's free, high-resolution, and has the
specific bands (NIR, SWIR) burn-severity mapping requires.

**Spectral band**
A slice of the electromagnetic spectrum a sensor measures separately
— visible red, green, blue, near-infrared, short-wave infrared, etc.
Each band reveals different information because different materials
reflect/absorb differently at different wavelengths.

**RGB (Red, Green, Blue)**
The three visible-light bands humans see naturally. Useful for
visually sanity-checking imagery (clouds, obvious burn scars,
landscape features), but NOT the bands that make dNBR/RdNBR work —
those rely on NIR and SWIR, which are invisible to the eye.

**NIR (Near-Infrared)**
Wavelengths just beyond visible red. Healthy vegetation reflects NIR
very strongly due to internal leaf cell structure — this is the most
well-known "plant health" signal in remote sensing. Sentinel-2 band
B8.

**SWIR (Short-Wave Infrared)**
Longer wavelengths still, strongly absorbed by leaf water content in
healthy vegetation. Burned/charred vegetation and exposed soil
reflect much more SWIR than living plants, since the moisture and
leaf structure are gone. Sentinel-2 bands B11/B12 (this project used
B12).

**Reflectance**
The fraction of incoming sunlight a surface bounces back, per
wavelength band. This is what a satellite sensor actually measures —
not "color" in the everyday sense. Sentinel-2 Surface Reflectance
(SR) products are already atmospherically corrected, meaning
scattering/absorption by the atmosphere itself has been
mathematically removed, leaving (approximately) the true reflectance
of the ground.

**Spatial resolution**
The real-world ground distance one pixel represents (e.g. "10m
resolution" = each pixel covers a 10×10m square). Different
Sentinel-2 bands have different native resolutions (10m, 20m, 60m);
using them together requires resampling the lower-resolution bands
up or down to match — this project resampled everything to 10m,
meaning B12 (native 20m) is interpolated, not truly 10m-detailed
(documented in the Stage 1 README).

**MGRS tile**
Sentinel-2 imagery is distributed in fixed grid tiles (~110km×110km
squares) under the Military Grid Reference System. An area of
interest larger than one tile (as this project's AOI was) requires
combining multiple tiles into a mosaic.

**Mosaic / composite**
Combining multiple satellite images (different tiles, different
dates, or both) into one continuous image. A "median composite"
takes the median pixel value across all contributing images per
pixel — useful for removing clouds/outliers, since a genuinely
cloudy pixel in one image will be outvoted by clear pixels from
other images at the same location.

**Cloud masking / cloud percentage**
Satellite images with heavy cloud cover are useless for ground
analysis — clouds block the view entirely and can also cast
shadows that distort readings. `CLOUDY_PIXEL_PERCENTAGE` is a
scene-wide metadata statistic; this project learned the hard way
that it doesn't guarantee a specific AOI within that scene is
cloud-free — visual inspection is still necessary.

**Google Earth Engine (GEE)**
A cloud platform for querying and processing huge public satellite
imagery archives (Sentinel-2, MODIS, Landsat, etc.) without
downloading raw data yourself. This project used GEE's Python API to
filter, composite, and export Sentinel-2 imagery for Similipal.

---

## 2. Burn Severity Indices

**NBR (Normalized Burn Ratio)**
`NBR = (NIR − SWIR) / (NIR + SWIR)`. Healthy vegetation has high NIR
and low SWIR, giving a high (near +1) NBR. Burned/bare land has the
opposite pattern, giving a low or negative NBR. This single number
compresses the "is this alive vegetation or not" signal into one
value per pixel.

**dNBR (differenced NBR)**
`dNBR = NBR(pre-fire) − NBR(post-fire)`. Measures how much a pixel's
vegetation health *dropped* between two dates. A large positive dNBR
indicates severe change (likely fire damage); near-zero means little
change. This is the standard, most widely published burn-severity
index (Key & Benson).

**RdNBR (Relative dNBR)**
`RdNBR = dNBR / sqrt(|NBR_prefire|)`. A refinement (Miller & Thode,
2007) that normalizes dNBR by each pixel's own starting vegetation
density, so a dense forest and a sparse/degraded forest are each
scored relative to their own baseline rather than on one shared
absolute scale. Used in this project because Similipal's landscape
is visibly heterogeneous (dense forest, sparse/degraded vegetation,
bare rock, agriculture all present). Known limitation: the
denominator can approach zero for very low pre-fire NBR values,
producing unstable/extreme RdNBR values at those pixels — this
project handled it by restricting RdNBR classification to
vegetated land only (see land-cover mask, below).

**Severity classes**
Burn severity is typically bucketed into discrete categories
(unburned, low, moderate, high) rather than used as a continuous
number, since ecological/management decisions are usually made at
the category level. This project used 4 classes plus a
`not_applicable` class for non-vegetated land.

**Threshold calibration**
The cutoff values that separate severity classes on the dNBR/RdNBR
scale. Standard published thresholds (Key & Benson; Miller & Thode)
were derived from specific studied ecosystems (Sierra Nevada
mixed-conifer forest, for RdNBR) and don't automatically transfer to
a different ecosystem (tropical dry deciduous Sal forest, here).
This project's final thresholds were manually calibrated — offset
from the literature defaults until burned-area extent matched the
independently reported fire size and the result visually traced the
known scar in RGB imagery. This is a documented compromise, not a
physically derived correction (see Stage 2 README for the full
investigation).

**Phenology confound**
A source of error where NBR/dNBR changes because of a plant's
*natural seasonal cycle* (e.g. deciduous leaf-drop in the dry
season) rather than fire damage. Since fire and natural
defoliation both lower NIR reflectance, an uncorrected index can't
tell them apart. This was the central recurring problem in this
project's Stage 2 — several correction attempts (background-drift
subtraction, multi-year anomaly baselines) were tried and rejected
before settling on the calibrated-threshold approach.

**Land cover mask**
A separate dataset (this project used ESA WorldCover, 10m
resolution) classifying each pixel as tree cover, shrubland, bare
land, agriculture, water, etc. — independent of the fire-detection
imagery. Used to restrict severity classification to vegetated
pixels only, since burn severity indices aren't meaningful on land
with nothing to combust (bare rock, water) and applying them there
produced false positives from unrelated seasonal reflectance change.

**MCD64A1 / FIRMS**
Two independent satellite fire-detection products (from MODIS and
VIIRS sensors) used briefly in this project's investigation as
potential "ground truth" for identifying confirmed-unburned
reference pixels. MCD64A1 maps burned-area extent monthly; FIRMS
detects active-fire thermal hotspots near-daily. Both were
ultimately abandoned as a correction mechanism here after visual
inspection showed they had a known blind spot for surface fires
under closed canopy — Similipal's actual fire behavior — meaning
their "confirmed unburned" pixels sometimes weren't.

---

## 3. Machine Learning Fundamentals

**Semantic segmentation**
A computer vision task where the model assigns a class label to
*every individual pixel* in an image, rather than one label for the
whole image (classification) or a bounding box around an object
(detection). This project's task — predicting a severity class for
every pixel — is semantic segmentation.

**Training / validation split**
Splitting available labeled data into a set the model learns from
(train) and a separate set used only to check how well it
generalizes to data it hasn't seen (validation). This project used
an 85/15 split, stratified so validation wasn't accidentally starved
of rare fire-containing patches.

**Stratified split**
Splitting data such that both the train and validation sets have a
similar proportion of some important category (here, patches
containing meaningful fire vs. not) — prevents an unlucky random
split from putting almost all fire examples in one set.

**Epoch**
One full pass through the entire training dataset. Models are
typically trained for many epochs (this project used 50), with the
model's weights updated after each batch within an epoch.

**Batch size**
The number of training examples processed together before the
model's weights are updated once. Larger batches are more
computationally efficient per example but need more memory; smaller
batches update more often but noisier. This project used 8.

**Loss function**
A single number measuring how wrong the model's predictions are,
which the training process tries to minimize. Lower loss generally
(not always, see overfitting) means better predictions.

**Cross-entropy loss**
The standard loss function for classification/segmentation tasks —
penalizes the model based on how confident and wrong its predicted
class probabilities were. Used here as the base loss, combined with
class weighting (below).

**Class imbalance**
When some classes appear far more often than others in the training
data — here, `unburned` (59.6% of pixels) vastly outnumbers
`high_severity` (0.24%). Left unaddressed, a model can achieve
deceptively good-looking aggregate accuracy by essentially ignoring
the rare class entirely.

**Class weighting (inverse frequency)**
A fix for class imbalance: multiply each class's contribution to the
loss by a weight inversely proportional to how common it is, so
mistakes on rare classes count for more than mistakes on common
ones, forcing the model to pay attention to them. This project's
`high_severity` class received a ~91x weight relative to its raw
frequency.

**ignore_index**
A special label value telling the loss function "don't compute loss
here at all" — used for pixels that shouldn't influence training in
either direction. This project used it for `not_applicable`
(non-vegetated land), since that class isn't a real severity
category the model should learn to predict.

**Optimizer (Adam)**
The algorithm that actually updates a model's internal weights based
on the loss, each training step. Adam is a widely used general-
purpose default, adjusting its own step size per-parameter based on
recent gradient history.

**Learning rate**
How large a step the optimizer takes when updating weights each
time. Too high risks the model never settling into a good solution;
too low makes training very slow. This project used 1e-4, a common
starting point for Adam.

**Overfitting**
When a model starts memorizing quirks of the specific training
examples rather than learning generalizable patterns — visible as
training loss continuing to drop while validation loss stops
improving or gets worse. Worth watching for given this project's
relatively small training set (1,262 patches) for a model trained
from scratch.

**Checkpoint**
A saved snapshot of a model's weights at a particular point in
training. This project saves the checkpoint with the best validation
mean IoU seen so far, not simply whatever the model looks like after
the final epoch — since the last epoch isn't necessarily the best
one (epoch 50 here scored lower than epoch 47's saved checkpoint).

**IoU (Intersection over Union)**
The primary evaluation metric for segmentation: for a given class,
`IoU = (predicted AND actual) / (predicted OR actual)`, measured in
pixels. A perfect prediction scores 1.0; no overlap at all scores 0.
Reported per-class here (not just one aggregate number) precisely
because aggregate accuracy can hide poor performance on rare classes
like `high_severity`.

**Mean IoU**
The average of per-class IoU scores across all classes. Used as the
single number to decide which training epoch's checkpoint is "best,"
while still reporting the individual per-class numbers alongside it
so no class's performance is hidden inside the average.

**Dice coefficient**
Another common segmentation overlap metric, closely related to IoU
(`Dice = 2×IoU / (1+IoU)`) — mentioned here since it appears
frequently in burn-severity literature alongside IoU, even though
this project standardized on IoU for its own reporting.

---

## 4. Model Architecture

**Convolutional Neural Network (CNN)**
A neural network architecture built around learned filters that
slide across an image detecting patterns (edges, textures, shapes)
at increasing levels of abstraction through successive layers. The
foundation underlying both the encoder and the overall U-Net
architecture used here.

**Encoder / decoder**
In segmentation architectures, the encoder progressively compresses
an image into smaller, more abstract feature representations
(learning "what's in this image"), while the decoder progressively
expands those features back up to full resolution (learning "where
exactly is it, pixel by pixel").

**U-Net**
A widely used segmentation architecture shaped like the letter U —
an encoder path down, a decoder path back up, with "skip
connections" directly linking matching levels of the encoder and
decoder. Originally developed for biomedical image segmentation,
now a standard choice for satellite imagery segmentation tasks too.

**Skip connections**
Direct links between an encoder layer and its corresponding decoder
layer, letting fine spatial detail lost during compression get
reinjected during expansion. This is what lets U-Net produce sharp,
precise pixel boundaries rather than blurry ones.

**ResNet (Residual Network) / ResNet34**
A common CNN design using "residual" shortcut connections that let
gradients flow through very deep networks during training without
vanishing. ResNet34 (34 layers) was used here as the U-Net's
*encoder* — a proven, off-the-shelf feature extractor rather than
designing one from scratch.

**Pretrained weights / transfer learning**
Reusing a model's weights from training on one large dataset (often
ImageNet, millions of everyday photos) as a starting point for a new
task, instead of starting from random weights. This project could
NOT do this for the encoder, because ImageNet pretraining assumes
3-channel RGB input, while this project's input is 24 channels
(12 pre-fire + 12 post-fire Sentinel-2 bands) — a fundamentally
different kind of data. The model was trained from scratch
(`encoder_weights=None`).

**Input channels**
The number of separate data layers stacked together as one input to
the model — 3 for ordinary RGB photos, 24 here (all 12 Sentinel-2
bands, twice, for pre-fire and post-fire).

**Device (CPU / GPU / MPS)**
The hardware training actually runs on. CUDA is NVIDIA's GPU
acceleration (not available on a Mac); MPS (Metal Performance
Shaders) is Apple Silicon's equivalent GPU acceleration, used here
since this project runs on an M4 Mac.

---

## 5. Project-Specific Pipeline Concepts

**Patch / tile (in the ML sense, not the satellite-tile sense)**
A small, fixed-size crop (256×256 pixels here) taken from the full
satellite image, used as one training example. Necessary because
the full Similipal image is far too large to feed into a model
directly, both computationally and because segmentation models
expect fixed input sizes.

**Stride / overlap**
How far apart consecutive patches are cut from the source image.
A stride smaller than the patch size (128 vs. 256 here, i.e. 50%
overlap) means patches overlap each other — used to increase the
number of training examples containing fire, given fire covers a
minority of the total area.

**Majority filter**
A post-processing step replacing each pixel with the most common
class among its neighbors, used to remove small, isolated
misclassified "speckle" pixels while preserving genuinely large,
connected regions. Tried (and ultimately not used in the final
pipeline) when a multi-year anomaly-correction approach introduced
terrain-illumination noise.

**Zero-shot evaluation / domain gap**
Testing a model on data from a region or condition it was never
trained on, with no fine-tuning — to measure how well it
generalizes. This project's Stage 6 will run the Similipal-trained
model on Pantanal, Mediterranean, and Siberian fire imagery this
way, to study how much performance degrades across different
biomes and why (baseline vegetation differences, fuel type, fire
behavior).

**Working dude, please wait....**