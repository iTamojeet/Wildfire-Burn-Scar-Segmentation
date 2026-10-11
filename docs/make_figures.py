"""
docs/make_figures.py

Regenerates every figure used by README.md, docs/GLOSSARY.md and
06_domain_gap_analysis/README.md.

* DATA figures use numbers copied verbatim from this project's own logs and
  JSON outputs (Stage 4 training log, Stage 5 evaluation, Stage 6
  domain_gap_results.json / recalibration_results.json, compute_baseline_shift.py).
* SCHEMATIC figures (spectral signature, U-Net, tiling, IoU, logit bias) are
  teaching diagrams. They are labelled "schematic" / "illustrative" and are
  NOT measured data.

Usage (from repo root):   python docs/make_figures.py
Prints per-class precision / recall / F1 / IoU tables (markdown) at the end,
which are the numbers quoted in the READMEs.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})

CLASS_NAMES = ["unburned", "low_severity", "moderate", "high_severity"]
CLASS_COLORS = ["#2ca02c", "#f2d33c", "#ff7f0e", "#8c1a1a"]
REG = {"Similipal": "#1f77b4", "Pantanal": "#2a9d8f", "N. Evia": "#e76f51", "C. Yakutia": "#6a4c93"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def human(n):
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{int(n)}"


# --------------------------------------------------------------------------
# DATA (copied from logs / JSON outputs)
# --------------------------------------------------------------------------
TRAIN_LOSS = [0.9081, 0.6390, 0.5439, 0.4866, 0.4431, 0.4339, 0.4155, 0.4067, 0.3863, 0.3741,
              0.3658, 0.3731, 0.3665, 0.3479, 0.3431, 0.3457, 0.3382, 0.3324, 0.3146, 0.3186,
              0.3125, 0.3014, 0.3010, 0.2928, 0.2899, 0.2949, 0.2815, 0.2828, 0.2778, 0.2697,
              0.2620, 0.2846, 0.2572, 0.2553, 0.2557, 0.2456, 0.2485, 0.2506, 0.2349, 0.2351,
              0.2382, 0.2323, 0.2309, 0.2229, 0.2271, 0.2245, 0.2244, 0.2172, 0.2142, 0.2227]
VAL_LOSS = [0.7569, 0.6686, 0.6037, 0.5059, 0.4786, 0.5116, 0.4802, 0.5812, 0.4388, 0.3416,
            0.5071, 0.4272, 0.3955, 0.4232, 0.5573, 0.2901, 0.3222, 0.5514, 0.3932, 0.4099,
            0.5483, 0.2825, 0.6876, 0.2789, 0.3361, 0.4612, 0.2713, 0.4934, 0.3925, 0.3700,
            0.3970, 0.2514, 0.2446, 0.2825, 0.2956, 0.2381, 0.2573, 0.3894, 0.3431, 0.3127,
            0.4508, 0.3456, 0.3986, 0.3034, 0.2402, 0.2994, 0.2504, 0.2955, 0.2493, 0.2445]
IOU_U = [0.802, 0.684, 0.846, 0.798, 0.828, 0.708, 0.766, 0.876, 0.875, 0.912, 0.907, 0.801, 0.812,
         0.757, 0.898, 0.886, 0.892, 0.886, 0.771, 0.825, 0.716, 0.916, 0.886, 0.918, 0.836, 0.768,
         0.915, 0.728, 0.914, 0.872, 0.829, 0.925, 0.916, 0.856, 0.878, 0.921, 0.906, 0.879, 0.834,
         0.848, 0.774, 0.940, 0.930, 0.869, 0.922, 0.932, 0.937, 0.887, 0.936, 0.892]
IOU_L = [0.472, 0.321, 0.571, 0.480, 0.494, 0.366, 0.373, 0.511, 0.605, 0.676, 0.603, 0.451, 0.509,
         0.440, 0.676, 0.644, 0.662, 0.521, 0.446, 0.469, 0.329, 0.723, 0.521, 0.717, 0.522, 0.398,
         0.691, 0.388, 0.666, 0.602, 0.490, 0.723, 0.704, 0.579, 0.587, 0.711, 0.700, 0.652, 0.510,
         0.544, 0.411, 0.760, 0.704, 0.559, 0.742, 0.753, 0.766, 0.599, 0.748, 0.658]
IOU_M = [0.491, 0.342, 0.509, 0.471, 0.455, 0.435, 0.401, 0.457, 0.589, 0.668, 0.553, 0.435, 0.511,
         0.490, 0.643, 0.646, 0.646, 0.545, 0.517, 0.448, 0.402, 0.712, 0.414, 0.712, 0.527, 0.439,
         0.653, 0.480, 0.689, 0.631, 0.481, 0.711, 0.699, 0.610, 0.581, 0.690, 0.696, 0.728, 0.537,
         0.560, 0.448, 0.676, 0.670, 0.550, 0.755, 0.732, 0.751, 0.580, 0.753, 0.692]
IOU_H = [0.155, 0.028, 0.142, 0.029, 0.038, 0.051, 0.060, 0.095, 0.034, 0.102, 0.123, 0.050, 0.059,
         0.059, 0.246, 0.077, 0.050, 0.093, 0.103, 0.051, 0.110, 0.109, 0.143, 0.096, 0.072, 0.091,
         0.068, 0.145, 0.108, 0.206, 0.098, 0.148, 0.145, 0.128, 0.172, 0.114, 0.075, 0.252, 0.168,
         0.132, 0.083, 0.248, 0.219, 0.169, 0.165, 0.223, 0.209, 0.185, 0.172, 0.195]
MEAN_IOU = [0.4800, 0.3439, 0.5169, 0.4445, 0.4534, 0.3899, 0.4000, 0.4850, 0.5258, 0.5894, 0.5466,
            0.4342, 0.4728, 0.4366, 0.6156, 0.5630, 0.5626, 0.5112, 0.4593, 0.4481, 0.3892, 0.6148,
            0.4909, 0.6109, 0.4894, 0.4238, 0.5818, 0.4353, 0.5943, 0.5775, 0.4745, 0.6268, 0.6159,
            0.5433, 0.5542, 0.6093, 0.5945, 0.6278, 0.5121, 0.5211, 0.4293, 0.6559, 0.6306, 0.5368,
            0.6460, 0.6600, 0.6659, 0.5628, 0.6524, 0.6093]
for _l in (TRAIN_LOSS, VAL_LOSS, IOU_U, IOU_L, IOU_M, IOU_H, MEAN_IOU):
    assert len(_l) == 50, len(_l)

PIXEL_SHARE = [59.62, 20.25, 5.97, 0.24, 13.93]          # Stage 3 pixel-level distribution
CLASS_WEIGHTS = [0.361, 1.063, 3.603, 91.244]             # Stage 4 inverse-frequency weights

CM_SIMILIPAL = np.array([[8630590, 374771, 46, 33618],
                         [169026, 2402686, 68304, 32461],
                         [3310, 87862, 596554, 37451],
                         [1312, 387, 628, 28035]])
CM_PANTANAL = np.array([[1046304, 2545, 307, 19761],
                        [1246335, 25017, 750, 11614],
                        [551232, 920635, 48468, 14726],
                        [9736, 250136, 988260, 545088]])
CM_EVIA = np.array([[27252, 33, 13, 99],
                    [35945, 2511, 179, 142],
                    [12083, 41607, 17854, 460],
                    [273, 7356, 152975, 577784]])
CM_YAKUTIA = np.array([[4942978, 158, 54, 255541],
                       [565627, 3010, 383, 19987],
                       [30268, 37138, 4384, 12115],
                       [553, 1886, 32878, 30758]])
CMS = {"Similipal (in-domain)": CM_SIMILIPAL, "Pantanal": CM_PANTANAL,
       "N. Evia": CM_EVIA, "C. Yakutia": CM_YAKUTIA}

MEAN_IOU_REGION = {"Similipal": 0.6659, "Pantanal": 0.17237801764274113,
                   "N. Evia": 0.31241495914429185, "C. Yakutia": 0.2455235374736281}
PER_CLASS_IOU = {
    "Similipal": [0.9368, 0.7663, 0.7512, 0.2094],
    "Pantanal": [0.36378, 0.01018, 0.01920, 0.29635],
    "N. Evia": [0.36001, 0.02861, 0.07929, 0.78175],
    "C. Yakutia": [0.85295, 0.00479, 0.03740, 0.08696],
}
PANTANAL_RECAL_MEAN = 0.22032882979742674
PANTANAL_RECAL_PERCLASS = [0.45076, 0.03321, 0.08974, 0.30760]

NBR_STATS = {  # mean, std, median of pre-fire NBR over ALL valid pixels (compute_baseline_shift.py)
    "Similipal": (0.270, 0.241, 0.256), "Pantanal": (0.449, 0.172, 0.481),
    "N. Evia": (0.266, 0.217, 0.236), "C. Yakutia": (0.284, 0.204, 0.295)}
BASELINE_SHIFT = {"Pantanal": 0.179, "N. Evia": -0.005, "C. Yakutia": 0.014}

ALPHAS = [0, 1, 2, 3, 5, 7, 10, 15, 20]
SWEEP = {
    "Pantanal": [0.1724, 0.1759, 0.1794, 0.1830, 0.1901, 0.1972, 0.2075, 0.2197, 0.2203],
    "N. Evia": [0.3124, 0.3123, 0.3121, 0.3120, 0.3117, 0.3114, 0.3110, 0.3104, 0.3097],
    "C. Yakutia": [0.2455, 0.2453, 0.2451, 0.2449, 0.2445, 0.2442, 0.2437, 0.2430, 0.2422],
}

# reference-label vs prediction class shares (% of total array), from the stage-6 logs
REF_SHARE = {"Pantanal": [18.01, 21.76, 25.99, 30.30], "N. Evia": [3.32, 3.31, 4.85, 44.62],
             "C. Yakutia": [62.16, 6.72, 0.96, 0.75]}
PRED_SHARE = {"Pantanal": [47.54, 19.96, 17.29, 11.83], "N. Evia": [4.35, 2.97, 9.85, 33.31],
              "C. Yakutia": [62.01, 0.47, 0.42, 3.57]}


def metrics(cm):
    cm = cm.astype(float)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    iou = tp / np.maximum(tp + fp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    total = cm.sum()
    acc = tp.sum() / total
    pe = (cm.sum(0) * cm.sum(1)).sum() / total ** 2
    kappa = (acc - pe) / (1 - pe)
    return dict(prec=prec, rec=rec, iou=iou, f1=f1, acc=acc, kappa=kappa,
                support=cm.sum(1), prevalence=cm.sum(1) / total)


def plot_cm(ax, cm, title, show_counts=True):
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            col = "white" if norm[i, j] > 0.55 else "black"
            txt = f"{100 * norm[i, j]:.1f}%"
            if show_counts:
                txt += f"\n({human(cm[i, j])})"
            ax.text(j, i, txt, ha="center", va="center", color=col, fontsize=8)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(["unb.", "low", "mod.", "high"])
    ax.set_yticklabels(["unb.", "low", "mod.", "high"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("reference / ground truth")
    ax.set_title(title)


# --------------------------------------------------------------------------
# FIGURES
# --------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(17, 5.4))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    stages = [
        ("01", "Data acquisition", "Sentinel-2 L2A pre/post\nGoogle Earth Engine\n12 bands · 10 m"),
        ("02", "Label generation", "RdNBR + WorldCover mask\ncalibrated thresholds\n4 severity classes"),
        ("03", "Tiling", "256×256 patches\nstride 128 (50 % overlap)\n24-channel pre+post stack"),
        ("04", "Training", "U-Net · ResNet34\nfrom scratch, 24 ch in\nweighted cross-entropy"),
        ("05", "In-domain eval", "IoU / Dice / confusion\nbest checkpoint\n(epoch 47)"),
        ("06", "Domain-gap study", "Zero-shot: Pantanal,\nN. Evia, C. Yakutia\n+ baseline recalibration"),
    ]
    cols = ["#264653", "#2a9d8f", "#3d7ea6", "#6a4c93", "#b5651d", "#a23b3b"]
    w, h, gap, x0 = 2.4, 2.5, 0.32, 0.25
    for i, (num, title, body) in enumerate(stages):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.9), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=cols[i], ec="none"))
        ax.text(x + 0.18, 1.9 + h - 0.32, num, color="white", fontsize=15, fontweight="bold", va="center")
        ax.text(x + w / 2, 1.9 + h - 0.85, title, color="white", fontsize=11.5, fontweight="bold",
                ha="center", va="center")
        ax.text(x + w / 2, 1.9 + 0.78, body, color="white", fontsize=8.8, ha="center", va="center",
                linespacing=1.5)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, 1.9 + h / 2), (x + w + gap - 0.02, 1.9 + h / 2),
                                         arrowstyle="-|>", mutation_scale=14, color="#555"))
    x_end5 = x0 + 4 * (w + gap) + w
    ax.plot([x0, x_end5], [1.45, 1.45], color="#264653", lw=2)
    ax.text((x0 + x_end5) / 2, 1.05, "TRAINING DOMAIN — Similipal Biosphere Reserve, Odisha, India "
            "(tropical dry deciduous Sal forest, 2021 fire)", ha="center", fontsize=10, color="#264653",
            fontweight="bold")
    x6 = x0 + 5 * (w + gap)
    ax.plot([x6, x6 + w], [1.45, 1.45], color="#a23b3b", lw=2)
    ax.text(x6 + w / 2, 1.05, "HELD-OUT BIOMES", ha="center", fontsize=10, color="#a23b3b", fontweight="bold")
    ax.text(8.5, 0.35, "Question: what happens when a burn-severity model trained in one ecosystem is applied "
            "— unchanged — to others?", ha="center", fontsize=10.5, style="italic", color="#333")
    save(fig, "fig01_pipeline.png")


def fig_spectral():
    wl = np.array([400, 450, 490, 560, 620, 665, 705, 740, 783, 842, 865, 945, 1100, 1250, 1450,
                   1610, 1800, 1940, 2190, 2300])
    healthy = np.array([0.03, 0.03, 0.04, 0.10, 0.06, 0.04, 0.12, 0.38, 0.46, 0.47, 0.47, 0.44, 0.46,
                        0.42, 0.28, 0.30, 0.30, 0.20, 0.14, 0.13])
    burned = np.array([0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.14, 0.14, 0.14, 0.16,
                       0.18, 0.15, 0.20, 0.22, 0.17, 0.17, 0.16])
    xs = np.linspace(400, 2300, 600)

    def smooth(y):
        d = np.interp(xs, wl, y)
        k = np.ones(25) / 25
        return np.convolve(np.pad(d, 12, mode="edge"), k, mode="valid")

    bands = {"B2": 490, "B3": 560, "B4": 665, "B5": 705, "B6": 740, "B7": 783, "B8": 842, "B8A": 865,
             "B9": 945, "B11": 1610, "B12": 2190}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axvspan(400, 700, color="#e8f4ea", alpha=0.6, lw=0)
    ax.axvspan(700, 1300, color="#fdf1e0", alpha=0.6, lw=0)
    ax.axvspan(1300, 2300, color="#f3e8f4", alpha=0.6, lw=0)
    for lab, x, y in [("visible", 550, 0.635), ("near-infrared (NIR)", 1000, 0.635), ("short-wave infrared (SWIR)", 1800, 0.635)]:
        ax.text(x, y, lab, ha="center", fontsize=9, color="#555")
    ax.plot(xs, smooth(healthy), color="#2e7d32", lw=2.6, label="healthy vegetation (schematic)")
    ax.plot(xs, smooth(burned), color="#5d4037", lw=2.6, label="burned surface (schematic)")
    for b, x in bands.items():
        hi = b in ("B8", "B12")
        ax.axvline(x, color="#d62728" if hi else "#999", lw=2.2 if hi else 0.8, ls="-" if hi else ":")
        ax.text(x, 0.004, b, rotation=90, ha="center", va="bottom", fontsize=8,
                color="#d62728" if hi else "#555", fontweight="bold" if hi else "normal",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
    nir_h, swir_h = np.interp(842, wl, healthy), np.interp(2190, wl, healthy)
    nir_b, swir_b = np.interp(842, wl, burned), np.interp(2190, wl, burned)
    nbr_h = (nir_h - swir_h) / (nir_h + swir_h)
    nbr_b = (nir_b - swir_b) / (nir_b + swir_b)
    ax.text(1330, 0.60, f"NBR = (B8 − B12) / (B8 + B12)\n\nhealthy:  ({nir_h:.2f} − {swir_h:.2f})/({nir_h:.2f} + {swir_h:.2f}) = {nbr_h:.2f}\n"
            f"burned:   ({nir_b:.2f} − {swir_b:.2f})/({nir_b:.2f} + {swir_b:.2f}) = {nbr_b:.2f}\n\n"
            f"dNBR = NBR_pre − NBR_post = {nbr_h - nbr_b:.2f}",
            fontsize=9, family="monospace", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("surface reflectance (fraction)")
    ax.set_title("Why NIR + SWIR reveal fire damage  (SCHEMATIC curves — illustrative, not measured data)")
    ax.set_xlim(400, 2300)
    ax.set_ylim(0, 0.65)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False)
    save(fig, "fig02_spectral_signature.png")


def fig_bands():
    info = [("B1", 443, 60), ("B2", 490, 10), ("B3", 560, 10), ("B4", 665, 10), ("B5", 705, 20),
            ("B6", 740, 20), ("B7", 783, 20), ("B8", 842, 10), ("B8A", 865, 20), ("B9", 945, 60),
            ("B11", 1610, 20), ("B12", 2190, 20)]
    colors = {10: "#1f77b4", 20: "#ff7f0e", 60: "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for name, wl, res in info:
        hi = name in ("B8", "B12")
        ax.bar(wl, 1, width=38, color=colors[res], alpha=0.95, edgecolor="#d62728" if hi else "none",
               linewidth=2.5 if hi else 0)
        ax.text(wl, 1.06, name, ha="center", fontsize=9, fontweight="bold" if hi else "normal")
        ax.text(wl, 0.5, f"{res} m", ha="center", va="center", color="white", fontsize=8, rotation=90)
    ax.set_yticks([])
    ax.set_xlabel("band centre wavelength (nm)")
    ax.set_ylim(0, 1.25)
    ax.set_title("The 12 Sentinel-2 L2A bands exported in this project (B10 is not in surface-reflectance products)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors[10], label="10 m native"), Patch(color=colors[20], label="20 m native (resampled to 10 m)"),
                       Patch(color=colors[60], label="60 m native (resampled to 10 m)"),
                       Patch(fc="white", ec="#d62728", lw=2, label="used for NBR (B8 NIR, B12 SWIR2)")],
              loc="upper left", frameon=False, fontsize=8, ncol=2)
    save(fig, "fig03_sentinel2_bands.png")


def fig_class_balance():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = ["unburned", "low", "moderate", "high", "not_applicable"]
    cols = CLASS_COLORS + ["#9e9e9e"]
    ax = axes[0]
    ax.bar(names, PIXEL_SHARE, color=cols)
    ax.set_yscale("log")
    ax.set_ylabel("% of all training-patch pixels (log scale)")
    ax.set_title("Pixel-level class distribution (Stage 3)")
    for i, v in enumerate(PIXEL_SHARE):
        ax.text(i, v * 1.15, f"{v:.2f}%", ha="center", fontsize=9)
    ax.set_ylim(0.1, 200)
    ax = axes[1]
    ax.bar(names[:4], CLASS_WEIGHTS, color=cols[:4])
    ax.set_yscale("log")
    ax.set_ylabel("loss weight (log scale)")
    ax.set_title("Inverse-frequency class weights (Stage 4)\n(not_applicable → ignore_index, no weight)")
    for i, v in enumerate(CLASS_WEIGHTS):
        ax.text(i, v * 1.15, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.1, 300)
    save(fig, "fig04_class_balance.png")


def fig_training():
    ep = np.arange(1, 51)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    ax = axes[0]
    ax.plot(ep, TRAIN_LOSS, label="train", color="#1f77b4", lw=2)
    ax.plot(ep, VAL_LOSS, label="validation", color="#d62728", lw=1.6)
    ax.axvline(47, color="k", ls="--", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted cross-entropy")
    ax.set_title("Loss")
    ax.legend(frameon=False)
    ax = axes[1]
    for series, name, c in zip([IOU_U, IOU_L, IOU_M, IOU_H], CLASS_NAMES, CLASS_COLORS):
        ax.plot(ep, series, label=name, color=c, lw=1.8)
    ax.axvline(47, color="k", ls="--", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation IoU")
    ax.set_title("Per-class validation IoU")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax = axes[2]
    ax.plot(ep, MEAN_IOU, color="#444", lw=1.8)
    ax.scatter([47], [MEAN_IOU[46]], s=140, marker="*", color="gold", ec="k", zorder=5,
               label=f"best checkpoint: epoch 47 ({MEAN_IOU[46]:.4f})")
    ax.scatter([50], [MEAN_IOU[49]], s=50, color="#d62728", zorder=5,
               label=f"final epoch 50 ({MEAN_IOU[49]:.4f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation mean IoU")
    ax.set_title("Mean IoU — noisy, so the last epoch ≠ the best epoch")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig05_training_curves.png")


def fig_in_domain():
    m = metrics(CM_SIMILIPAL)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    plot_cm(axes[0], CM_SIMILIPAL, "In-domain confusion matrix (row-normalised = recall)\nSimilipal validation patches")
    ax = axes[1]
    x = np.arange(4)
    w = 0.26
    ax.bar(x - w, m["prec"], w, label="precision", color="#4c78a8")
    ax.bar(x, m["rec"], w, label="recall", color="#f58518")
    ax.bar(x + w, m["iou"], w, label="IoU", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(["unburned", "low", "moderate", "high"])
    ax.set_ylim(0, 1.08)
    ax.set_title("Precision / recall / IoU per class\n(high_severity: recall 92 % but precision 21 %)")
    for i in range(4):
        for off, key in zip([-w, 0, w], ["prec", "rec", "iou"]):
            ax.text(i + off, m[key][i] + 0.015, f"{m[key][i]:.2f}", ha="center", fontsize=7)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    save(fig, "fig06_in_domain_evaluation.png")


def fig_domain_overview():
    labels = ["Similipal\n(in-domain)", "N. Evia", "C. Yakutia", "Pantanal", "Pantanal\n+ recalibration"]
    vals = [0.6659, 0.31241, 0.24552, 0.17238, PANTANAL_RECAL_MEAN]
    cols = [REG["Similipal"], REG["N. Evia"], REG["C. Yakutia"], REG["Pantanal"], REG["Pantanal"]]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, vals, color=cols)
    bars[-1].set_hatch("//")
    bars[-1].set_edgecolor("white")
    for b, v in zip(bars, vals):
        drop = (0.6659 - v) / 0.6659 * 100
        txt = f"{v:.3f}" if v == 0.6659 else f"{v:.3f}\n(−{drop:.1f} %)"
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, txt, ha="center", fontsize=9)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("mean IoU over 4 severity classes")
    ax.set_title("Zero-shot transfer: agreement with each region's reference labels")
    ax.text(0.5, -0.2, "Caveat: reference labels for the three zero-shot regions use literature-default RdNBR thresholds, whereas the\n"
            "training labels used calibrated (+380) thresholds — see 06_domain_gap_analysis/README.md §Limitations.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555")
    save(fig, "fig07_domain_gap_overview.png")


def fig_per_class():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    regions = list(PER_CLASS_IOU)
    w = 0.2
    x = np.arange(4)
    for i, r in enumerate(regions):
        ax.bar(x + (i - 1.5) * w, PER_CLASS_IOU[r], w, label=r, color=REG[r])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("IoU")
    ax.set_title("Per-class IoU by region (high IoU on a dominant class partly reflects class prevalence)")
    ax.legend(frameon=False, ncol=4)
    save(fig, "fig08_per_class_by_region.png")


def fig_domain_cms():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, name in zip(axes, ["Pantanal", "N. Evia", "C. Yakutia"]):
        plot_cm(ax, CMS[name], f"{name} (zero-shot)")
    save(fig, "fig09_domain_confusion_matrices.png")


def fig_baseline():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    names = list(NBR_STATS)
    means = [NBR_STATS[n][0] for n in names]
    stds = [NBR_STATS[n][1] for n in names]
    meds = [NBR_STATS[n][2] for n in names]
    ax.bar(names, means, yerr=stds, color=[REG[n] for n in names], capsize=5, alpha=0.9)
    ax.scatter(names, meds, marker="D", color="k", zorder=5, label="median")
    ax.set_ylabel("pre-fire NBR (all valid pixels)")
    ax.set_title("Pre-fire NBR distribution per region\n(bar = mean, whisker = ±1 SD of pixels)")
    ax.legend(frameon=False)
    ax = axes[1]
    for n in BASELINE_SHIFT:
        drop = (0.6659 - MEAN_IOU_REGION[n]) / 0.6659 * 100
        ax.scatter(BASELINE_SHIFT[n], drop, s=160, color=REG[n], ec="k", zorder=5)
        ax.annotate(n, (BASELINE_SHIFT[n], drop), textcoords="offset points", xytext=(8, 6))
    ax.set_xlabel("baseline NBR shift vs Similipal (mean pre-fire NBR difference)")
    ax.set_ylabel("mean-IoU drop vs in-domain (%)")
    ax.set_ylim(40, 85)
    ax.set_xlim(-0.05, 0.22)
    ax.set_title("Shift vs performance drop — only n = 3 regions,\ndescriptive only (no trend can be fitted)")
    save(fig, "fig10_baseline_shift.png")


def fig_recal():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    ax.plot(ALPHAS, SWEEP["Pantanal"], "o-", color=REG["Pantanal"], lw=2)
    ax.axhline(SWEEP["Pantanal"][0], color="#999", ls=":")
    ax.annotate("best α = 20 is the LAST grid value\n→ the optimum is not bracketed", (20, SWEEP["Pantanal"][-1]),
                xytext=(6.5, 0.185), arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set_xlabel("correction strength α")
    ax.set_ylabel("Pantanal mean IoU")
    ax.set_title("Pantanal: baseline-conditioned logit bias\n(0.1724 → 0.2203)")
    ax = axes[1]
    for n in ["N. Evia", "C. Yakutia"]:
        d = np.array(SWEEP[n]) - SWEEP[n][0]
        ax.plot(ALPHAS, d, "o-", color=REG[n], label=n, lw=2)
    ax.axhline(0, color="#999", ls=":")
    ax.set_xlabel("correction strength α")
    ax.set_ylabel("change in mean IoU vs α = 0")
    ax.set_title("Near-zero baseline shift ⇒ the correction is a no-op\n(tiny monotone decrease)")
    ax.legend(frameon=False)
    save(fig, "fig11_recalibration_sweep.png")


def fig_pred_vs_ref():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    for ax, n in zip(axes, ["Pantanal", "N. Evia", "C. Yakutia"]):
        ref = np.array(REF_SHARE[n])
        pred = np.array(PRED_SHARE[n])
        ref, pred = 100 * ref / ref.sum(), 100 * pred / pred.sum()
        x = np.arange(4)
        ax.bar(x - 0.2, ref, 0.4, label="reference (RdNBR defaults)", color="#9e9e9e")
        ax.bar(x + 0.2, pred, 0.4, label="model prediction", color=REG[n])
        ax.set_xticks(x)
        ax.set_xticklabels(["unb.", "low", "mod.", "high"])
        ax.set_title(n)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("% of classified (vegetated) pixels")
    fig.suptitle("Class shares: reference vs prediction, each normalised to classified pixels", y=1.02, fontweight="bold")
    save(fig, "fig12_pred_vs_reference_shares.png")


def fig_logit_bias():
    p = np.array([0.50, 0.30, 0.15, 0.05])
    shift = 0.179
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, alpha in zip(axes, [0, 3, 5, 20]):
        logit = np.log(p) + alpha * np.arange(4) * shift
        q = np.exp(logit - logit.max())
        q /= q.sum()
        ax.bar(range(4), q, color=CLASS_COLORS)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["unb.", "low", "mod.", "high"])
        ax.set_title(f"α = {alpha}   → argmax: {['unb.', 'low', 'mod.', 'high'][int(q.argmax())]}")
        for i, v in enumerate(q):
            ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("probability")
    fig.suptitle("ILLUSTRATIVE single pixel, model softmax = (0.50, 0.30, 0.15, 0.05), baseline shift = +0.179:  "
                 "adjusted logit = log p + α · class_index · shift", y=1.04, fontsize=10, fontweight="bold")
    save(fig, "fig13_logit_bias_illustration.png")


def fig_iou():
    fig, ax = plt.subplots(figsize=(8.8, 5))
    ax.add_patch(Rectangle((1, 1), 5, 4, fc="#1f77b4", alpha=0.35, ec="#1f77b4", lw=2))
    ax.add_patch(Rectangle((3, 2), 5, 4, fc="#ff7f0e", alpha=0.35, ec="#ff7f0e", lw=2))
    ax.add_patch(Rectangle((3, 2), 3, 3, fc="#2ca02c", alpha=0.55, ec="none"))
    ax.text(1.2, 1.25, "reference only\n= FN", fontsize=9, color="#0d3b66")
    ax.text(6.3, 5.4, "prediction only\n= FP", fontsize=9, color="#a04d00")
    ax.text(3.45, 3.4, "overlap = TP", fontsize=10, fontweight="bold", color="#1b5e20")
    ax.text(1, 6.3, "reference (blue)", color="#1f77b4", fontsize=10, fontweight="bold")
    ax.text(5.6, 1.4, "prediction (orange)", color="#ff7f0e", fontsize=10, fontweight="bold")
    tp, fn, fp = 9, 20 - 9, 20 - 9
    ax.text(8.6, 4.8, f"TP = {tp}\nFP = {fp}\nFN = {fn}\n\nIoU  = TP/(TP+FP+FN)\n     = {tp}/{tp + fp + fn} = {tp / (tp + fp + fn):.2f}\n"
            f"Dice = 2TP/(2TP+FP+FN)\n     = {2 * tp}/{2 * tp + fp + fn} = {2 * tp / (2 * tp + fp + fn):.2f}\n\n"
            "Dice = 2·IoU / (1 + IoU)", fontsize=9, family="monospace", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("IoU (Jaccard) and Dice for one class  (schematic, areas in grid cells)")
    save(fig, "fig14_iou_schematic.png")


def fig_tiling():
    fig, ax = plt.subplots(figsize=(10.5, 6))
    W, H = 1024, 768
    ax.add_patch(Rectangle((0, 0), W, H, fc="#f4f4f4", ec="#333", lw=2))
    for gx in range(0, W + 1, 128):
        ax.plot([gx, gx], [0, H], color="#ddd", lw=0.6)
    for gy in range(0, H + 1, 128):
        ax.plot([0, W], [gy, gy], color="#ddd", lw=0.6)
    ax.add_patch(Rectangle((128, 128), 256, 256, fc="#1f77b4", alpha=0.30, ec="#1f77b4", lw=2.5))
    ax.add_patch(Rectangle((256, 128), 256, 256, fc="#ff7f0e", alpha=0.30, ec="#ff7f0e", lw=2.5))
    ax.add_patch(Rectangle((256, 128), 128, 256, fc="#d62728", alpha=0.35, ec="none"))
    ax.text(150, 150, "patch A\n256×256", color="#0d3b66", fontsize=10, fontweight="bold")
    ax.text(395, 330, "patch B\n(shifted by 128 px)", color="#a04d00", fontsize=10, fontweight="bold")
    ax.text(262, 250, "shared\n50 %", color="#7f0000", fontsize=10, fontweight="bold")
    ax.annotate("", xy=(256, 80), xytext=(128, 80), arrowprops=dict(arrowstyle="<->"))
    ax.text(192, 55, "stride = 128 px", ha="center", fontsize=9)
    ax.set_xlim(-30, W + 30)
    ax.set_ylim(-30, H + 30)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Tiling: 256×256 patches, stride 128  →  every interior pixel appears in up to 4 patches\n"
                 "Randomly splitting such patches into train/val lets neighbours share pixels (spatial leakage risk)",
                 fontsize=10)
    save(fig, "fig15_patch_tiling.png")


def fig_unet():
    fig, ax = plt.subplots(figsize=(19, 7.4))
    ax.set_xlim(0, 20.5)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    specs = [  # label, h, colour
        ("input\n24 ch\n256²", 4.4, "#9e9e9e"),
        ("conv1\n64 ch\n128²", 3.8, "#4c78a8"),
        ("layer1\n64 ch\n64²", 3.2, "#4c78a8"),
        ("layer2\n128 ch\n32²", 2.6, "#4c78a8"),
        ("layer3\n256 ch\n16²", 2.0, "#4c78a8"),
        ("layer4\n512 ch\n8²", 1.4, "#2f4b7c"),
        ("dec 1\n256 ch\n16²", 2.0, "#f58518"),
        ("dec 2\n128 ch\n32²", 2.6, "#f58518"),
        ("dec 3\n64 ch\n64²", 3.2, "#f58518"),
        ("dec 4\n32 ch\n128²", 3.8, "#f58518"),
        ("dec 5\n16 ch\n256²", 4.4, "#f58518"),
        ("1×1 conv\n4 classes\n256²", 4.4, "#54a24b"),
    ]
    bw, gap = 1.1, 0.55
    xs, tops = [], []
    for i, (lab, h, c) in enumerate(specs):
        x = 0.3 + i * (bw + gap)
        y0 = 0.9 + (4.4 - h) / 2
        ax.add_patch(FancyBboxPatch((x, y0), bw, h, boxstyle="round,pad=0.01,rounding_size=0.08", fc=c, ec="none"))
        ax.text(x + bw / 2, y0 + h / 2, lab, color="white", ha="center", va="center", fontsize=8.5, linespacing=1.4)
        xs.append(x)
        tops.append(y0 + h)
        if i < len(specs) - 1:
            ax.add_patch(FancyArrowPatch((x + bw, 0.9 + 2.2), (x + bw + gap, 0.9 + 2.2), arrowstyle="-|>",
                                         mutation_scale=10, color="#777"))
    for src, dst in [(4, 6), (3, 7), (2, 8), (1, 9)]:
        ax.add_patch(FancyArrowPatch((xs[src] + bw / 2, tops[src]), (xs[dst] + bw / 2, tops[dst]),
                                     connectionstyle="arc3,rad=-0.35", arrowstyle="-|>", mutation_scale=11,
                                     color="#d62728", ls="--", lw=1.4))
    ax.text(xs[0], 7.3, "skip connections (feature maps concatenated into the decoder)", color="#d62728",
            fontsize=10, ha="left")
    ax.text(0.3, 0.45, "ENCODER = ResNet34 (random init, 24-ch first conv)",
            fontsize=10, color="#2f4b7c", fontweight="bold")
    ax.text(11.0, 0.45, "DECODER: upsample ×2 → concat skip → 2 convs (256-128-64-32-16)",
            fontsize=10, color="#b35c00", fontweight="bold")
    ax.set_title("U-Net with a ResNet34 encoder as built by segmentation_models_pytorch (schematic; ≈24 M parameters)",
                 fontsize=12, fontweight="bold", y=0.99)
    save(fig, "fig16_unet_schematic.png")


def fig_workflow_failures():
    rows = [
        ("Single scene for a multi-tile AOI", "black nodata blocks", "AOI > one MGRS tile", "median mosaic → later: smaller AOI, tile pinned"),
        ("Scene-wide cloud % filter", "cloud/haze still in AOI", "metadata is per scene, not per AOI", "visual QA of every tile"),
        ("Percentile stretch incl. nodata", "washed-out/black previews", "−32768 skews the 2nd percentile", "mask nodata before stretching"),
        ("geemap download", "0.5 MB file, 1 of 12 bands", "silent partial download", "verify bands/size after download"),
        ("Dry-season Sal leaf-drop", "≈91 % of forest 'burned'", "phenology looks like fire in NBR", "calibrated threshold offset (+380)"),
        ("MODIS/FIRMS reference mask", "real scar labelled unburned", "MCD64A1 misses canopy surface fires", "abandoned"),
        ("Multi-year anomaly baseline", "ridge-line speckle", "terrain illumination in composites", "abandoned"),
        ("RdNBR on all land", "≈84 % 'burned'", "denominator → 0 on bare land", "WorldCover vegetation mask"),
    ]
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.axis("off")
    tbl = ax.table(cellText=[list(r) for r in rows],
                   colLabels=["What was tried", "Symptom", "Root cause", "Outcome"],
                   loc="center", cellLoc="left", colWidths=[0.25, 0.22, 0.28, 0.25])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f1f5f7")
    ax.set_title("Debugging log in one picture: symptoms, root causes, outcomes", fontweight="bold")
    save(fig, "fig17_debugging_log.png")


def print_metric_tables():
    for name, cm in CMS.items():
        m = metrics(cm)
        print(f"\n### {name}   pixel accuracy {m['acc']:.4f}   Cohen kappa {m['kappa']:.4f}")
        print("| class | prevalence (reference) | precision | recall | F1/Dice | IoU |")
        print("|---|---|---|---|---|---|")
        for i, c in enumerate(CLASS_NAMES):
            print(f"| {c} | {100 * m['prevalence'][i]:.2f} % | {m['prec'][i]:.3f} | {m['rec'][i]:.3f} | "
                  f"{m['f1'][i]:.3f} | {m['iou'][i]:.3f} |")
    print("\nRelative improvement Pantanal recalibration:",
          f"{(PANTANAL_RECAL_MEAN - MEAN_IOU_REGION['Pantanal']) / MEAN_IOU_REGION['Pantanal'] * 100:.1f} %")
    gap = 0.6659 - MEAN_IOU_REGION["Pantanal"]
    print("Fraction of Pantanal gap to in-domain recovered:",
          f"{(PANTANAL_RECAL_MEAN - MEAN_IOU_REGION['Pantanal']) / gap * 100:.1f} %")
    print("Pantanal drop after recalibration:",
          f"{(0.6659 - PANTANAL_RECAL_MEAN) / 0.6659 * 100:.1f} %")


if __name__ == "__main__":
    fig_pipeline()
    fig_spectral()
    fig_bands()
    fig_class_balance()
    fig_training()
    fig_in_domain()
    fig_domain_overview()
    fig_per_class()
    fig_domain_cms()
    fig_baseline()
    fig_recal()
    fig_pred_vs_ref()
    fig_logit_bias()
    fig_iou()
    fig_tiling()
    fig_unet()
    fig_workflow_failures()
    print_metric_tables()