"""
generate_figures.py
===================
Generates all paper figures from experiment_results.csv.
Output: figures/*.pdf  (vector, ready for \\includegraphics in LaTeX)

Run:
    python generate_figures.py
"""

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH     = Path("simulation/results/experiment_results.csv")
FIGURES_DIR  = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ── IEEE-friendly style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "figure.dpi":        300,
    "pdf.fonttype":      42,   # embeds fonts for IEEE submission
})

NONIID_LABELS  = {"mild": r"Mild ($\alpha{=}1.0$)",
                   "moderate": r"Moderate ($\alpha{=}0.5$)",
                   "severe": r"Severe ($\alpha{=}0.1$)"}
MAIN_SEEDS     = ("42", "142", "242", "342", "442")

# Colorblind-friendly palette (Wong 2011)
COLORS  = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
HATCHES = [""] * 7


# ── Data helpers ─────────────────────────────────────────────────────────────
def load_rows():
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def collect(rows, method, noniid, seeds=MAIN_SEEDS):
    accs, f1s, eces = [], [], []
    for r in rows:
        if r["method"] == method and r["noniid"] == noniid and r["seed"] in seeds:
            accs.append(float(r["accuracy"]))
            f1s.append(float(r["f1_score"]))
            eces.append(float(r["ece"]))
    return accs, f1s, eces


def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


# ── Figure 1: Main accuracy comparison ───────────────────────────────────────
def fig_accuracy(rows):
    methods = ["Centralized", "Local-Best", "FedAvg", "FedProx",
               "FedMD", "EqualWt-Ens", "Ours"]
    noniids = ["mild", "moderate", "severe"]
    short   = ["Centralized", "Local-Best", "FedAvg", "FedProx",
               "FedMD", "EqualWt", "Ours"]

    n_methods = len(methods)
    n_groups  = len(noniids)
    width     = 0.11
    x         = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for i, (method, label) in enumerate(zip(methods, short)):
        means, errs = [], []
        for noniid in noniids:
            accs, _, _ = collect(rows, method, noniid)
            m, s = mean_std(accs)
            means.append(m)
            errs.append(s)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, means, width,
               yerr=errs, capsize=2,
               color=COLORS[i], hatch=HATCHES[i],
               edgecolor="black", linewidth=0.5,
               label=label, error_kw={"linewidth": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels([NONIID_LABELS[n] for n in noniids])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.35, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.legend(ncol=2, loc="upper right", framealpha=0.9,
              handlelength=1.2, handleheight=0.9)
    ax.set_title("Accuracy by method and non-IID level\n(mean ± std, 5 seeds)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIGURES_DIR / "fig1_accuracy.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 2: ECE comparison ──────────────────────────────────────────────────
def fig_ece(rows):
    methods = ["FedAvg", "FedProx", "FedMD", "EqualWt-Ens", "Ours"]
    short   = ["FedAvg", "FedProx", "FedMD", "EqualWt", "Ours"]
    noniids = ["mild", "moderate", "severe"]

    n_methods = len(methods)
    width     = 0.15
    x         = np.arange(len(noniids))

    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    for i, (method, label) in enumerate(zip(methods, short)):
        means, errs = [], []
        for noniid in noniids:
            _, _, eces = collect(rows, method, noniid)
            m, s = mean_std(eces)
            means.append(m)
            errs.append(s)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, means, width,
               yerr=errs, capsize=2,
               color=COLORS[i + 2], hatch=HATCHES[i + 2],
               edgecolor="black", linewidth=0.5,
               label=label, error_kw={"linewidth": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels([NONIID_LABELS[n] for n in noniids])
    ax.set_ylabel("ECE (lower is better)")
    ax.set_ylim(0, 0.50)
    ax.legend(ncol=2, loc="upper left", framealpha=0.9,
              handlelength=1.2, handleheight=0.9)
    ax.set_title("Calibration error by method and non-IID level\n(mean ± std, 5 seeds)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIGURES_DIR / "fig2_ece.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 3: Communication cost (log scale) ──────────────────────────────────
def fig_communication():
    RESNET50_PARAMS   = 25_557_032
    FEDMD_SAMPLES     = 500
    NUM_CLASSES       = 2
    BYTES_PER_PARAM   = 4

    fa_total    = 2 * RESNET50_PARAMS * BYTES_PER_PARAM          # bytes
    fedmd_total = 2 * FEDMD_SAMPLES * NUM_CLASSES * BYTES_PER_PARAM
    ours_total  = 128 + 3 * 32                                    # 224 bytes

    labels  = ["FedAvg /\nFedProx", "FedMD", "Ours"]
    totals  = [fa_total, fedmd_total, ours_total]
    colors  = [COLORS[0], COLORS[1], COLORS[2]]
    hatches = ["", "", ""]

    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    bars = ax.bar(labels, totals,
                  color=colors, hatch=hatches,
                  edgecolor="black", linewidth=0.7, width=0.5)

    # Annotate each bar
    def fmt_bytes(n):
        if n < 1_000:       return f"{n} B"
        elif n < 1_000_000: return f"{n/1000:.1f} KB"
        else:               return f"{n/1_000_000:.1f} MB"

    ax.set_yscale("log")

    # Place labels just above each bar with a small gap
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val * 1.4,
                fmt_bytes(val),
                ha="center", va="bottom", fontsize=7, color="black")

    ax.set_ylim(top=fa_total * 6)
    ax.set_ylabel("Payload (bytes, log scale)")
    ax.set_title("Communication cost comparison")
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "fig4_communication.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Figure 4: Ablation study ──────────────────────────────────────────────────
def fig_ablation(rows):
    components = ["Ours", "Abl:No CapMul", "Abl:No Conf",
                  "Abl:No ECE", "Abl:No Bonus", "Abl:No PoC"]
    short      = ["Full", "No CapMul", "No Conf",
                  "No ECE", "No Bonus", "No PoC"]
    noniids    = ["mild", "moderate", "severe"]

    n_comp  = len(components)
    width   = 0.13
    x       = np.arange(len(noniids))

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for i, (comp, label) in enumerate(zip(components, short)):
        means, errs = [], []
        for noniid in noniids:
            accs, _, _ = collect(rows, comp, noniid)
            m, s = mean_std(accs)
            means.append(m)
            errs.append(s)
        offset = (i - n_comp / 2 + 0.5) * width
        ax.bar(x + offset, means, width,
               yerr=errs, capsize=2,
               color=COLORS[i], hatch=HATCHES[i],
               edgecolor="black", linewidth=0.5,
               label=label, error_kw={"linewidth": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels([NONIID_LABELS[n] for n in noniids])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.45, 0.92)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.legend(ncol=2, loc="upper right", framealpha=0.9,
              handlelength=1.2, handleheight=0.9)
    ax.set_title("Ablation study — accuracy per component removed\n(mean ± std, 5 seeds)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIGURES_DIR / "fig3_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}\nRun experiment.py first.")

    rows = load_rows()
    print(f"Loaded {len(rows)} rows. Generating figures...")

    fig_accuracy(rows)
    fig_ece(rows)
    fig_communication()
    fig_ablation(rows)

    print(f"\nDone. All figures saved to {FIGURES_DIR}/")
    print("Include in LaTeX with: \\includegraphics[width=\\columnwidth]{figures/figN_name}")
