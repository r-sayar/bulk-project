"""
GSE279480 Null: 100 genes stratified by expression level, KDE distributions
with mode detection (mirrors gtex_ranked_gene_distributions.py).

Symbol categories: Housekeeping / Blood-Immune / Mitochondrial (MT-) / Other.
Bimodal genes flagged by KDE peak detection.
"""

from collections import Counter
from pathlib import Path
import gzip

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

# ── Config ──
HERE = Path(__file__).parent
COUNTS_CSV = HERE.parent / "data/GSE279480/GSE279480_P441_genecounts.csv.gz"
SERIES_MATRIX = HERE.parent / "data/GSE279480/GSE279480_series_matrix.txt.gz"
SYMBOL_MAP = HERE / "ensembl_to_symbol.tsv"

# ── Load Null subset ──
print("Loading GSE279480 Null...")
rows = {}
with gzip.open(SERIES_MATRIX, "rt") as fh:
    for line in fh:
        if line.startswith("!series_matrix_table_begin"):
            break
        if not line.startswith("!Sample_"):
            continue
        parts = line.rstrip("\n").split("\t")
        rows.setdefault(parts[0], []).append([p.strip('"') for p in parts[1:]])
meta = pd.DataFrame({"lib": rows["!Sample_description"][0]})
for r in rows.get("!Sample_characteristics_ch1", []):
    keys = [c.split(":", 1)[0].strip() for c in r if ":" in c]
    if not keys:
        continue
    key = Counter(keys).most_common(1)[0][0]
    meta[key] = [c.split(":", 1)[1].strip() if ":" in c else "" for c in r]

counts = pd.read_csv(COUNTS_CSV, index_col=0)
null_libs = [l for l in meta.loc[meta["stimulation"] == "Null", "lib"] if l in counts.columns]
expr = counts[null_libs].values.astype(np.float64)
gene_ids = np.array(counts.index)
n_samples = expr.shape[1]

lib_sizes = expr.sum(axis=0, keepdims=True)
log_expr = np.log2(expr / lib_sizes * 1e6 + 1)
gene_means = log_expr.mean(axis=1)
gene_stds = log_expr.std(axis=1)

sym_df = pd.read_csv(SYMBOL_MAP, sep="\t").drop_duplicates("ensembl_id")
ens_to_sym = dict(zip(sym_df["ensembl_id"], sym_df["symbol"]))
gene_names = np.array([ens_to_sym.get(g) if isinstance(ens_to_sym.get(g), str) else g
                       for g in gene_ids])

# ── Categories ──
HOUSEKEEPING = {
    "ACTB", "GAPDH", "B2M", "EEF1A1", "FTL", "FKBP8", "TMSB4X", "PSAP",
    "RPL13A", "RPL7", "RPL3", "RPS18", "RPS27A", "RPL11", "RPL4", "RPL8",
    "RPS3", "RPS4X", "RPL13", "RPL6", "RPL10", "RPL5", "RPS2", "RPS14",
    "EEF2", "UBC", "UBB", "PPIA", "HSP90AB1", "YWHAZ", "ALDOA", "ENO1",
    "LDHA", "PKM", "TPI1", "PGK1", "HNRNPA1", "NPM1", "CALM1", "ATP5F1B",
    "NDUFA4", "COX7C", "ATP5MC3", "VIM", "FLNA", "TPT1", "EIF4A1",
}
BLOOD_IMMUNE = {
    "HBB", "HBA1", "HBA2", "HBD", "HBG1", "HBG2",
    "S100A9", "S100A8", "S100A12", "S100A6", "S100A4",
    "LCP1", "CSF3R", "IFITM2", "IFITM3", "IFITM1",
    "HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-DRA", "HLA-DRB1",
    "SERPINA1", "LYZ", "MNDA", "FGL2", "AIF1", "TYROBP", "FCER1G",
    "CD74", "FCN1", "LST1", "CTSS", "CYBB", "NCF2", "SPI1",
    "IL1B", "CXCL8", "CCL3", "PTPRC", "CD14", "ITGB2",
}
MITOCHONDRIAL = {n for n in gene_names if isinstance(n, str) and n.startswith("MT-")}


def categorize(name):
    if name in HOUSEKEEPING:
        return "Housekeeping"
    if name in BLOOD_IMMUNE:
        return "Blood/Immune"
    if name in MITOCHONDRIAL:
        return "Mitochondrial"
    return "Other"


CAT_COLORS = {"Housekeeping": "#3fb950", "Blood/Immune": "#f78166",
              "Mitochondrial": "#d2a8ff", "Other": "#58a6ff"}

# ── Select 100 genes stratified by expression ──
expressed = np.where(gene_means > 1)[0]
sorted_by_mean = expressed[np.argsort(gene_means[expressed])[::-1]]
step = max(1, len(sorted_by_mean) // 100)
selected_idx = sorted_by_mean[::step][:100]


def detect_modes(vals):
    xs = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 1000)
    kde = gaussian_kde(vals, bw_method="scott")
    density = kde(xs)
    peaks, _ = find_peaks(density, prominence=density.max() * 0.08, distance=40)
    return xs, density, peaks, len(peaks) >= 2


# Find any bimodal among all expressed genes
print(f"Scanning {len(expressed):,} expressed genes for bimodality...")
all_bimodal = []
for gi in expressed:
    vals = log_expr[gi, :]
    if vals.std() < 0.3:
        continue
    _, _, _, is_bi = detect_modes(vals)
    if is_bi:
        all_bimodal.append(gi)
print(f"  Found {len(all_bimodal)} bimodal genes")

# Inject some bimodal genes into the 100-gene selection
selected_set = set(selected_idx)
extra = [g for g in all_bimodal if g not in selected_set]
np.random.seed(42)
if extra:
    np.random.shuffle(extra)
    n_add = min(20, len(extra))
    selected_idx = np.concatenate([selected_idx[:100 - n_add], np.array(extra[:n_add])])
selected_idx = selected_idx[np.argsort(gene_means[selected_idx])[::-1]]

# ── Theme ──
BG = "#0e1117"; CARD = "#1a1d23"; TEXT = "#e6edf3"; MUTED = "#7d8590"; GRID = "#21262d"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "text.color": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "grid.color": GRID, "font.family": "sans-serif",
    "font.size": 8,
})

NCOLS, NROWS = 10, 10
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(32, 28))
fig.suptitle(
    "GSE279480 Null — 100 Genes Stratified by Expression Level\n"
    "KDE distributions with mode detection  ·  "
    r"$\bf{Green}$=Housekeeping  $\bf{Orange}$=Blood/Immune  "
    r"$\bf{Purple}$=Mitochondrial  $\bf{Blue}$=Other  ·  "
    r"$\bigstar$ = Bimodal",
    fontsize=18, fontweight="bold", color=TEXT, y=0.998,
)

for idx in range(len(selected_idx)):
    row, col = divmod(idx, NCOLS)
    ax = axes[row, col]
    gi = selected_idx[idx]
    vals = log_expr[gi, :]
    name = gene_names[gi] if isinstance(gene_names[gi], str) else gene_ids[gi]
    cat = categorize(name)
    color = CAT_COLORS[cat]

    xs, density, peaks, is_bimodal = detect_modes(vals)
    ax.fill_between(xs, density, alpha=0.3, color=color)
    ax.plot(xs, density, color=color, lw=1.5)

    for pi in peaks:
        mode_val = xs[pi]
        ax.axvline(mode_val, color="#f0883e", ls="--", lw=1, alpha=0.7)
        ax.plot(mode_val, density[pi], "o", color="#f0883e", ms=5, zorder=5)

    bimodal_flag = " *" if is_bimodal else ""
    title_color = "#ff7b72" if is_bimodal else TEXT
    ax.set_title(f"{name}{bimodal_flag}\n[{cat}] μ={gene_means[gi]:.1f}",
                 fontsize=7, fontweight="bold", color=title_color, pad=3)

    if peaks is not None and len(peaks) > 0:
        lines = []
        for pi in peaks:
            mode_val = xs[pi]
            n_near = int(np.sum(np.abs(vals - mode_val) < 0.5))
            lines.append(f"{n_near}@{mode_val:.1f}")
        ax.text(0.97, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=6, color=MUTED, ha="right", va="top")

    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if is_bimodal:
        for spine in ax.spines.values():
            spine.set_edgecolor("#ff7b72"); spine.set_linewidth(2)

for idx in range(len(selected_idx), NROWS * NCOLS):
    r, c = divmod(idx, NCOLS)
    axes[r, c].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = HERE / "gse279480_null_ranked_distributions.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out}")

print(f"\n{'='*60}\nBIMODAL GENES FOUND: {len(all_bimodal)}\n{'='*60}")
for gi in all_bimodal[:30]:
    name = gene_names[gi] if isinstance(gene_names[gi], str) else gene_ids[gi]
    cat = categorize(name)
    vals = log_expr[gi, :]
    xs, density, peaks, _ = detect_modes(vals)
    modes_str = ", ".join(
        f"{xs[p]:.1f} ({int(np.sum(np.abs(vals - xs[p]) < 0.5))} samples)" for p in peaks
    )
    print(f"  {name:18s} [{cat:14s}]  modes: {modes_str}")
