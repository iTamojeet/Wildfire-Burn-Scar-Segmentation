# Stage 5: Evaluation (In-Domain, Held-Out Validation Set)

Evaluates the Stage 4 best checkpoint (`best_model.pt`, epoch 47) on
the held-out Similipal validation set (223 patches, never seen
during training).

## Files

- `evaluate.py` — loads the checkpoint, computes per-class IoU/Dice
  via a full confusion matrix, saves numeric results and visual
  examples
- `results/evaluation_results.json` — numeric results
- `results/visual_examples/` — 8 side-by-side RGB / ground truth /
  prediction images, prioritized to include fire-containing patches

## Results

| Class | IoU | Dice |
|---|---|---|
| unburned | 0.9368 | 0.9674 |
| low_severity | 0.7663 | 0.8677 |
| moderate | 0.7512 | 0.8579 |
| high_severity | 0.2094 | 0.3463 |

**Mean IoU: 0.6659** — matches Stage 4's best training-time checkpoint
exactly, confirming the checkpoint loaded and evaluation logic are
consistent with training-time metrics.

## Confusion matrix

Rows = ground truth, columns = predicted (pixel counts):

|  | unburned | low_severity | moderate | high_severity |
|---|---|---|---|---|
| **unburned** | 8,630,590 | 374,771 | 46 | 33,618 |
| **low_severity** | 169,026 | 2,402,686 | 68,304 | 32,461 |
| **moderate** | 3,310 | 87,862 | 596,554 | 37,451 |
| **high_severity** | 1,312 | 387 | 628 | 28,035 |

## Key finding: high_severity has strong recall but weak precision

Derived from the confusion matrix's `high_severity` row and column:

- **Recall (catch rate):** 28,035 / 30,362 true high_severity pixels
  = **92.3%** — the model rarely misses real high-severity fire.
- **Precision:** 28,035 / 131,565 pixels predicted high_severity
  = **21.3%** — roughly 3-4 pixels get incorrectly swept into this
  class for every one correctly identified.

**Why this happened:** Stage 4's ~91x inverse-frequency class weight
on `high_severity` (needed because it's only 0.24% of all pixels)
successfully stopped the model from ignoring the class, but pushed
it toward guessing `high_severity` liberally rather than
conservatively — high recall, low precision. IoU (which penalizes
both misses and false alarms) reflects this honestly at 0.21, well
below what the strong recall alone might suggest. This pattern is
directly visible in the visual examples (e.g. `example_00007.png`,
where a small false-positive high-severity pixel appears at the edge
of a real high-severity patch in the prediction but not ground truth).

**Decision:** documented as a known model characteristic rather than
corrected via retraining with a lower class weight. This is a
legitimate remaining limitation, not a blocking bug — the model
correctly finds nearly all real high-severity fire, at the cost of
over-flagging some adjacent/similar pixels. Carried forward into
Stage 6: **this same precision bias should be expected on the
zero-shot domain-gap regions too**, and is relevant context for
interpreting those results — a precision drop there may partly
reflect this pre-existing in-domain trait rather than being purely a
domain-gap effect.

## Visual inspection summary

Across all 8 saved examples, predicted severity maps closely track
ground truth shape and spatial location, including in scenes with
many small, scattered burned patches. White regions in the ground
truth/prediction panels are `not_applicable` (non-vegetated,
ignore_index) pixels, correctly excluded from both training and
display — not a rendering error.

## Known limitations carried into Stage 6

- `high_severity` precision (21.3%) is low; expect this pattern to
  persist or worsen on out-of-domain regions.
- Evaluation is in-domain only (same fire event, held-out spatial
  patches) — Stage 6 will test genuine cross-region generalization,
  which is a substantially harder test than this held-out split.