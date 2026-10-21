# Wildfire Burn Scar Segmentation

**Train a burn-severity model on one fire, then test it — unchanged — on three different ecosystems, and find out what breaks.**
Sentinel-2 · Google Earth Engine · U-Net/ResNet34 (PyTorch) · zero-shot domain-gap analysis

![PyTorch](https://img.shields.io/badge/PyTorch-U--Net%20%2B%20ResNet34-EE4C2C?logo=pytorch&logoColor=white)
![Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Sentinel--2%20L2A-4285F4?logo=google&logoColor=white)
![Status](https://img.shields.io/badge/status-Stages%201--6%20complete-2ea44f)

![Pipeline](docs/images/fig01_pipeline.png)

---

## 1. What this project asks

Burn-severity maps are normally produced with one index (dNBR/RdNBR) and thresholds tuned for the ecosystem they
were developed in. A neural network can learn those maps from raw bands — but **does it keep working when the
ecosystem changes?** I trained on the 2021 Similipal Biosphere Reserve fire (Odisha, India; tropical dry deciduous
Sal forest) and applied the model with no fine-tuning to:

| Region | Biome | Fire |
|---|---|---|
| **Pantanal** (Encontro das Águas), Brazil | seasonally flooded wetland / savanna | 2020 drought fires |
| **North Evia**, Greece | Mediterranean pine forest & maquis | Aug 2021 wind-driven fire |
| **Central Yakutia**, Russia | boreal larch taiga on permafrost | record 2021 season |

## 2. Headline results

Two reference labellings were built for each zero-shot scene from the same imagery: the **default RdNBR thresholds**, and the
**Similipal rule** (the calibrated thresholds the model was trained on). The unchanged model was scored against both.

| Mean IoU | vs default-rule reference | vs **Similipal-rule** reference |
|---|---|---|
| Similipal — in-domain (held-out patches) | — | **0.666** (optimistic, see §7) |
| Pantanal | 0.172 | **0.714** |
| North Evia | 0.312 | **0.665** |
| Central Yakutia | 0.246 | **0.490** |
| Pantanal + baseline-conditioned recalibration | 0.220 | — |

![Label-rule decomposition](docs/images/fig18_label_rule_decomposition.png)

**What the results support**

* My first reading — "the model loses 53–74 % of its IoU out of domain" — was **mostly wrong as an explanation**. That drop is
  against default-threshold references. Against the rule the model was actually trained on, it scores 0.71 in Pantanal and 0.66
  in Evia (about as well as at home) and 0.49 in Yakutia.
* The two threshold rules disagree enormously about how much burned (Pantanal 47 % vs 81 %, Evia 85 % vs 94 %, Yakutia 1.6 % vs
  11.9 % of classified pixels). **The threshold rule, not the network, is what fails to transfer** — and no independent reference
  says which rule is right in any region.
* Only **Yakutia** shows a genuine generalisation loss. Its vegetated pre-fire NBR is the most shifted from Similipal
  (−0.10 mean, −0.14 median; Pantanal +0.04, Evia −0.01). Three regions can only show a pattern, not a trend.
* The first version's baseline shift for Pantanal (+0.179) was a land-cover composition artefact (all pixels, including water
  and bare land). The recalibration built on it (+0.048 mean IoU against the default-rule reference) is better read as a shift
  toward the more severe default-rule labels than as a baseline correction.

![Domain gap](docs/images/fig07_domain_gap_overview.png)
![Per-class IoU](docs/images/fig08_per_class_by_region.png)
![Confusion matrices](docs/images/fig09_domain_confusion_matrices.png)

*Observation kept from the first pass (against the default-rule reference):* in Pantanal `high_severity` has precision 0.92 and
recall 0.30, the reverse of in-domain (0.21 / 0.92); in Yakutia the model misses most reference fire and scatters false `high`
calls over unburned forest. Their causes are untested.

---

## 3. The pipeline

| Stage | What happens | Key decisions | Docs |
|---|---|---|---|
| **01 Data acquisition** | Pre/post Sentinel-2 L2A (12 bands, 10 m) for Similipal via Earth Engine; median mosaic of ≈ 9 scenes per window | windows 15 Jan–5 Feb and 20 Mar–10 Apr 2021; B12 resampled 20→10 m | [01](01_data_acquisition/README.md) |
| **02 Label generation** | RdNBR inside an ESA WorldCover vegetation mask → 4 severity classes + `not_applicable` | thresholds shifted **+380** to counter dry-season leaf-drop (≈ 28.7 % burned); four alternatives rejected | [02](02_label_generation/README.md) |
| **03 Tiling** | 256 × 256 patches, stride 128, 24-channel pre+post stack; 1 485 patches, 85/15 split | pixel mix 59.6 / 20.3 / 6.0 / **0.24** % (+ 13.9 % n/a) | [03](03_tiling/README.md) |
| **04 Training** | U-Net/ResNet34 from scratch, weighted cross-entropy (weights 0.36 / 1.06 / 3.60 / **91.2**), `ignore_index` for n/a, Adam 1e-4, batch 8, 50 epochs | best checkpoint = epoch 47 | [04](04_training/README.md) |
| **05 Evaluation** | IoU/Dice/confusion on held-out patches | `high_severity` recall 0.92, precision 0.21 — documented, not retrained | [05](05_evaluation/README.md) |
| **06 Domain gap** | zero-shot inference on three biomes, baseline-shift analysis, reference labels, recalibration | single-scene tile-pinned acquisition after six failed attempts | [06](06_domain_gap_analysis/README.md) |

![Training](docs/images/fig05_training_curves.png)
![In-domain](docs/images/fig06_in_domain_evaluation.png)

### Evidence from the pipeline (generated by the scripts)

| Stage 1 — pre/post RGB | Stage 2 — severity overlay |
|---|---|
| ![s1](01_data_acquisition/data/raw/rgb_preview.png) | ![s2](02_label_generation/data/dnbr/severity_overlay_final.png) |

| Stage 5 — RGB / reference / prediction | Stage 5 — another patch |
|---|---|
| ![e0](05_evaluation/results/visual_examples/example_00000.png) | ![e5](05_evaluation/results/visual_examples/example_00005.png) |

| Pantanal | North Evia | Yakutia false-positive overlay (cyan) |
|---|---|---|
| ![p](06_domain_gap_analysis/data/raw/pantanal_rgb_preview.png) | ![m](06_domain_gap_analysis/data/raw/mediterranean_rgb_preview.png) | ![y](06_domain_gap_analysis/results/siberia_false_positive_overlay_v2.png) |

---

## 4. Ideas you may want to reuse

* **Debugging log as a deliverable.** The failures taught more than the model: see the table below and
  `06_domain_gap_analysis/README.md` §3.

![Debugging](docs/images/fig17_debugging_log.png)

* **Decompose a gap before explaining it.** Scoring the same predictions against two labelling rules (and against the ceiling of a perfect replica) showed that most of my first explanation was wrong — see `06_domain_gap_analysis/README.md` §6A.
* **Per-class metrics + prevalence + κ**, not just accuracy or mean IoU.
* **Assert on every download** (bands, % non-zero, file size, tile coverage).
* **Label the compromises.** The +380 offset, the in-sample α and the proxy labels are stated, not hidden.

## 5. Repository layout

```
.
├── README.md                     ← you are here
├── docs/
│   ├── GLOSSARY.md               concepts, why-we-used-it, references (figures included)
│   ├── make_figures.py           regenerates every figure from logged numbers
│   └── images/                   fig01 … fig17
├── 01_data_acquisition/          acquire_similipal.py, verify_rgb_preview.py
├── 02_label_generation/          fetch_landcover_mask.py, compute_dnbr.py (RdNBR + offset),
│                                 calibrate_threshold.py, overlay_severity_check.py
├── 03_tiling/                    tile_dataset.py  → patches/ (git-ignored) + manifests
├── 04_training/                  dataset.py, train.py → checkpoints/ (git-ignored)
├── 05_evaluation/                evaluate.py → results/
└── 06_domain_gap_analysis/       acquire_regions_v6.py, zero_shot_inference.py, …
                                  evaluate_domain_gap.py, recalibrate_and_evaluate.py
```

## 6. Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install earthengine-api geemap geedim rasterio numpy scipy scikit-learn matplotlib \
            torch segmentation-models-pytorch
earthengine authenticate        # then set YOUR Earth Engine Cloud project id inside the scripts
```

1. `01_data_acquisition/acquire_similipal.py` → `verify_rgb_preview.py` (≈ 10–25 min of downloads)
2. `02_label_generation/fetch_landcover_mask.py` → `compute_dnbr.py` → `overlay_severity_check.py`
3. `03_tiling/tile_dataset.py` → 4. `04_training/train.py` (≈ 1–2 h on Apple M-series MPS) → 5. `05_evaluation/evaluate.py`
6. Stage 6 run order is listed in `06_domain_gap_analysis/README.md` §12. Large rasters, patches and checkpoints are git-ignored;
   re-create them with the scripts (or attach the checkpoint to a GitHub Release).
7. `python docs/make_figures.py` rebuilds the figures.

## 7. Limitations (read before citing any number)

1. **Spatial leakage:** 50 %-overlapping patches were split randomly, so validation patches share pixels with training patches;
   the in-domain score is optimistic. **No independent test set**; the checkpoint was selected on the validation set it is reported on.
2. **Proxy labels:** labels are an RdNBR rule (with a hand-calibrated +380 offset) applied to the same bands the network sees —
   in-domain IoU measures how well the rule was re-learned, not independent fire detection.
3. **Label-protocol mismatch — measured:** zero-shot references with the default rule disagree strongly with the Similipal rule the model learned (the main reason for the first-pass 'gap'). Both rules are proxies; **no independent burned-area reference** was used in any region.
4. **Phenology/temporal confounds** (Pantanal Jul→Dec, Yakutia May→Aug, Similipal dry-season leaf-drop).
5. **n = 1 event per biome, one small AOI and one date pair each, one training run, no seeds or confidence intervals.**
6. **n = 3 regions:** the baseline-shift pattern (vegetated-only) is descriptive; the all-pixel version in the first analysis was misleading.
7. **Recalibration** was tuned in-sample, its best α sits at the edge of the search grid, and its input signal (+0.179) was a composition artefact.
8. Scene-level cloud filtering only; smoke/cirrus not masked per pixel. WorldCover 2021 used as mask for all years.

## 8. Next steps

Independent burned-area references to adjudicate the two threshold rules (Copernicus EMS for Evia, MapBiomas Fogo for Pantanal, GWIS for Yakutia); region-specific thresholds scored against them; seasonal-matched pre-fire imagery; a block-wise spatial split with several seeds; a Yakutia covariate-shift deep-dive; cheap domain adaptation (AdaBN/Tent, few-shot fine-tuning); RBR instead of RdNBR.

## 9. Data, credits, licence

Contains modified Copernicus Sentinel data (2020–2021). ESA WorldCover © ESA, CC BY 4.0. MODIS/FIRMS data courtesy of NASA.
Event information from Mongabay-India, Down To Earth, Copernicus EMS and the papers listed in `docs/GLOSSARY.md` §15.
**Licence:** MIT
**Author** Tamojeet

New to remote sensing or segmentation? Start with [`docs/GLOSSARY.md`](docs/GLOSSARY.md).