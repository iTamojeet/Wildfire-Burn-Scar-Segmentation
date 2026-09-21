# Stage 4: Training

Trains a U-Net (ResNet34 encoder) to predict burn severity class per
pixel, using the Stage 3 tiled patches as input.

## Files

- `dataset.py` — PyTorch `Dataset` class that loads tiled patches
  and prepares them for training
- `train.py` — main training loop, model definition, class weighting,
  validation, checkpointing
- `checkpoints/` — saved model weights (`best_model.pt`)

## Input / output specification

- **Input:** 24-channel stack per patch — 12 Sentinel-2 bands
  (pre-fire) concatenated with 12 bands (post-fire), 256×256 pixels.
  Order: `[pre_fire_B1...B12, post_fire_B1...B12]`, matching the
  Stage 1 band order (B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12).
- **Output:** per-pixel class prediction across 4 classes:
  `unburned`, `low_severity`, `moderate`, `high_severity`.
  `not_applicable` (non-vegetated land, from the Stage 2 WorldCover
  mask) is excluded — the model never predicts it and receives no
  training signal from those pixels (see "not_applicable handling"
  below).

## dataset.py — what it does and why

**Normalization:** raw Sentinel-2 Surface Reflectance values are
scaled integers (roughly 0–10,000 representing 0–100% reflectance).
Dividing by `REFLECTANCE_SCALE = 10000.0` brings them into a
roughly 0–1 float range, which neural networks train on far more
stably than raw large integers.

**not_applicable handling:** Stage 2's severity raster uses code `4`
for `not_applicable`. This is remapped to `255` at load time, which
is then set as `ignore_index` in the loss function (see below) — so
those pixels contribute zero gradient during training, positive or
negative. The model is never taught to recognize non-vegetated land
as a class; that job was already done by the independent WorldCover
mask in Stage 2.

**Path resolution:** Stage 3's `manifest.json` files store `.npy`
paths relative to wherever `tile_dataset.py` was originally run from
(`03_tiling/`). Since `train.py` and `evaluate.py` run from different
directories, `dataset.py` resolves each manifest entry's path
relative to the manifest file's own location (taking just the
filename and rejoining it with the manifest's directory) rather than
trusting the stored path literally. This makes the dataset loadable
regardless of which directory a script is launched from.

## train.py — architecture and training setup

**Model: U-Net with ResNet34 encoder**
(`segmentation_models_pytorch.Unet`). ResNet34 provides the encoder
(the "understand what's in this image" half); U-Net's decoder and
skip connections reconstruct a full-resolution, pixel-precise output
from the encoder's compressed features. See `docs/GLOSSARY.md` for
what U-Net, ResNet, encoders, and skip connections mean if this is
unfamiliar.

**`encoder_weights=None` — trained from scratch, deliberately.**
ResNet34's usual pretrained weights come from ImageNet, a dataset of
ordinary 3-channel RGB photographs. This project's input is 24
channels of Sentinel-2 reflectance data — a fundamentally different
kind of signal with no meaningful correspondence to RGB photo
features. Using ImageNet weights here would provide no real benefit
and could actively mislead early training. The full encoder and
decoder are trained from random initialization using only this
project's own data.

**Practical consequence of training from scratch:** with only 1,262
training patches, this model has meaningfully less data to learn
from than a typical fine-tuned setup would need, and no pretrained
head start. This is a known limitation, not an oversight — watch
for overfitting (train loss still dropping while val loss plateaus
or worsens) in the printed epoch log.

**Loss function: weighted cross-entropy with `ignore_index`**
Standard `CrossEntropyLoss`, with two modifications:
1. `ignore_index=255` — excludes `not_applicable` pixels entirely
   (see above).
2. `weight=class_weights` — per-class multipliers computed via
   inverse frequency from Stage 3's `class_pixel_counts.json`:

   | Class | Pixel share | Weight |
   |---|---|---|
   | unburned | 59.62% | 0.361 |
   | low_severity | 20.25% | 1.063 |
   | moderate | 5.97% | 3.603 |
   | high_severity | 0.24% | 91.244 |

   Without this weighting, the loss function would be dominated by
   the `unburned` class simply because it's most common — the model
   could achieve a deceptively low loss while barely learning
   `high_severity` at all, since misclassifying it would barely move
   the (unweighted) average loss. The ~91x weight forces every
   `high_severity` mistake to count roughly as much as ~91 `unburned`
   mistakes, keeping the rare class relevant to what the model
   optimizes for.

**Optimizer:** Adam, learning rate `1e-4` — a standard, reasonable
default for a segmentation model trained from scratch; not
separately tuned for this project.

**Batch size:** 8. Chosen as a starting point for the M4 Mac's
memory; can be lowered to 4 if a memory error occurs.

**Device:** auto-detects Apple Silicon MPS acceleration (`mps`),
falling back to CUDA or CPU if unavailable. No CUDA exists on Mac
hardware, so MPS is the correct accelerator here.

**Epochs:** 50, no early stopping — the loop always runs the full
50 and separately tracks the best checkpoint seen (see below), so
running the full count costs time but never loses the best result.

## Evaluation during training

**Per-class IoU, not just loss.** After each epoch, the model is run
on the validation set and Intersection-over-Union is computed
separately for each of the 4 classes, then averaged into "mean IoU."
This is the metric used to decide the best checkpoint — loss alone
doesn't reveal whether the model is actually getting the rare
`high_severity` class right, since that class barely moves the loss
either way even when weighted.

**Checkpointing: best-mean-IoU, not final-epoch.**
`checkpoints/best_model.pt` is overwritten only when a new epoch
achieves a higher mean IoU than any previous epoch — **not** simply
saved once at the end. This matters because validation metrics are
noisy epoch-to-epoch (small validation set, rare class), so the
final epoch (50) is not necessarily the best one. In this project's
actual run, epoch 47 (mean IoU 0.6659) was better than epoch 50
(mean IoU 0.6093), and `best_model.pt`