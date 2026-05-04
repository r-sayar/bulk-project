#!/usr/bin/env python3
"""
geneplot — Quick gene expression visualization from terminal.

Usage:
    ./geneplot.py SOD2                          # ranked expression, single gene
    ./geneplot.py SOD2 HBB ACTB                 # overlay 3 genes
    ./geneplot.py SOD2 --kde                    # show KDE inset
    ./geneplot.py SOD2 --linear                 # CPM scale (no log)
    ./geneplot.py SOD2 --train-test             # 700/103 train/test split
    ./geneplot.py --top 20                      # top 20 by mean expression
    ./geneplot.py --top 20 --by variance        # top 20 by variance
    ./geneplot.py --random 10                   # 10 random genes
    ./geneplot.py --bimodal 10                  # top 10 bimodal genes
    ./geneplot.py --uniform 10                  # top 10 most uniform genes
    ./geneplot.py --similar SOD2 --n 5          # 5 genes most similar to SOD2
    ./geneplot.py --per-panel 3 --panels 4      # 3 genes overlaid, 4 panels
    ./geneplot.py SOD2 --sort pc1               # sort samples by PC1 instead of rank
    ./geneplot.py SOD2 --density                # dot opacity = KDE density
    ./geneplot.py --archetypes 12               # cluster into 12 shape archetypes
    ./geneplot.py SOD2 --sc                     # overlay scRNA pseudobulk donors
    ./geneplot.py --open                        # open output image after saving

Examples:
    ./geneplot.py HBB ACTB GAPDH --kde --density --open
    ./geneplot.py --top 50 --per-panel 10 --linear
    ./geneplot.py --bimodal 20 --kde --open
    ./geneplot.py --similar GSTM1 --n 3 --density --open
"""

import argparse
import sys
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# ── Paths ─────────────────────────────────────────────────────────────
BASEDIR = os.path.dirname(os.path.abspath(__file__))
GTEX_PATH = '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz'
HCA_PATH = os.path.join(BASEDIR, 'pseudobulk/hca_blood_pseudobulk.npz')
OUT_PATH = os.path.join(BASEDIR, 'geneplot_output.png')
GSE279480_COUNTS = os.path.join(BASEDIR, 'data/GSE279480/GSE279480_P441_genecounts.csv.gz')
GSE279480_MATRIX = os.path.join(BASEDIR, 'data/GSE279480/GSE279480_series_matrix.txt.gz')
GSE279480_SYMBOL_MAP = os.path.join(BASEDIR, 'gse279480_variance/ensembl_to_symbol.tsv')

# ── Dark theme ────────────────────────────────────────────────────────
BG    = '#0e1117'
CARD  = '#1a1d23'
TEXT  = '#e6edf3'
MUTED = '#7d8590'
GRID  = '#21262d'
COLORS = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#f0883e',
          '#a5d6ff', '#7ee787', '#ffa198', '#e2b6ff', '#ffc68a']


def setup_style():
    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': CARD,
        'axes.edgecolor': GRID, 'axes.labelcolor': TEXT,
        'text.color': TEXT, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'grid.color': GRID, 'font.family': 'sans-serif', 'font.size': 10,
    })


def load_gtex():
    """Load GTEx whole blood, return (log_expr, cpm, gene_names, n_samples)."""
    df = pd.read_csv(GTEX_PATH, sep='\t', skiprows=2, compression='gzip')
    raw = df.iloc[:, 2:].values.astype(np.float64)
    names = df['Description'].values.astype(str)
    lib = raw.sum(axis=0, keepdims=True)
    cpm = raw / lib * 1e6
    log_expr = np.log2(cpm + 1)
    return log_expr, cpm, names, raw.shape[1]


def load_gse279480(stimulation='Null'):
    """Load GSE279480 (Smithmyer 2025) bulk RNA-seq, filtered to one stimulation.

    Returns (log_expr, cpm, gene_names, n_samples) where gene_names are symbols
    (Ensembl IDs preserved when no symbol exists).
    """
    import gzip
    from collections import Counter
    rows = {}
    with gzip.open(GSE279480_MATRIX, 'rt') as fh:
        for line in fh:
            if line.startswith('!series_matrix_table_begin'):
                break
            if not line.startswith('!Sample_'):
                continue
            parts = line.rstrip('\n').split('\t')
            rows.setdefault(parts[0], []).append([p.strip('"') for p in parts[1:]])
    libs = rows['!Sample_description'][0]
    stim_col = None
    for r in rows.get('!Sample_characteristics_ch1', []):
        keys = [c.split(':', 1)[0].strip() for c in r if ':' in c]
        if not keys:
            continue
        key = Counter(keys).most_common(1)[0][0]
        if key == 'stimulation':
            stim_col = [c.split(':', 1)[1].strip() if ':' in c else '' for c in r]
            break
    if stim_col is None:
        raise RuntimeError('No stimulation column found in series matrix')
    keep_libs = [l for l, s in zip(libs, stim_col) if s == stimulation]

    counts = pd.read_csv(GSE279480_COUNTS, index_col=0)
    keep_libs = [l for l in keep_libs if l in counts.columns]
    sub = counts[keep_libs]
    raw = sub.values.astype(np.float64)
    ids = sub.index.values

    sym_df = pd.read_csv(GSE279480_SYMBOL_MAP, sep='\t').drop_duplicates('ensembl_id')
    ens_to_sym = dict(zip(sym_df['ensembl_id'], sym_df['symbol']))
    names = np.array([ens_to_sym[g] if isinstance(ens_to_sym.get(g), str) else g
                      for g in ids], dtype=object).astype(str)

    lib = raw.sum(axis=0, keepdims=True)
    cpm = raw / lib * 1e6
    log_expr = np.log2(cpm + 1)
    return log_expr, cpm, names, raw.shape[1]


def load_sc():
    """Load HCA pseudobulk, return (log_expr, gene_names, sample_ids)."""
    d = np.load(HCA_PATH, allow_pickle=True)
    raw = d['expr'].astype(np.float64)
    names = d['gene_names'].astype(str)
    ids = d['sample_ids'].astype(str)
    lib = raw.sum(axis=0, keepdims=True)
    cpm = raw / lib * 1e6
    log_expr = np.log2(cpm + 1)
    return log_expr, cpm, names, ids


def find_gene(name, gene_names):
    """Find gene index by name (case-insensitive)."""
    upper = name.upper()
    for i, n in enumerate(gene_names):
        if n.upper() == upper:
            return i
    return None


def select_genes(args, log_expr, cpm, gene_names):
    """Return list of gene indices based on args."""
    n_genes = len(gene_names)
    gene_means = log_expr.mean(axis=1)
    gene_stds = log_expr.std(axis=1)
    named_mask = ~np.array([n.startswith('ENSG') for n in gene_names])
    expressed = (gene_means > 1) & named_mask

    if args.genes:
        indices = []
        for g in args.genes:
            idx = find_gene(g, gene_names)
            if idx is None:
                print(f"Warning: gene '{g}' not found, skipping")
            else:
                indices.append(idx)
        return indices

    if args.top:
        pool = np.where(expressed)[0]
        if args.by == 'variance':
            order = np.argsort(gene_stds[pool])[::-1]
        elif args.by == 'cv':
            cvs = np.where(gene_means[pool] > 0, gene_stds[pool] / gene_means[pool], 0)
            order = np.argsort(cvs)[::-1]
        else:
            order = np.argsort(gene_means[pool])[::-1]
        return pool[order[:args.top]].tolist()

    if args.random:
        pool = np.where(expressed)[0]
        rng = np.random.default_rng(args.seed)
        return rng.choice(pool, min(args.random, len(pool)), replace=False).tolist()

    if args.bimodal:
        scores = []
        for gi in np.where(expressed)[0]:
            vals = log_expr[gi, :]
            if vals.std() < 0.3:
                continue
            xs = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 1000)
            kde = gaussian_kde(vals, bw_method='scott')
            density = kde(xs)
            peaks, props = find_peaks(density, prominence=density.max() * 0.05, distance=30)
            if len(peaks) < 2:
                continue
            ph = density[peaks]
            top2 = np.argsort(ph)[::-1][:2]
            sep = abs(xs[peaks[top2[0]]] - xs[peaks[top2[1]]])
            score = sep * min(props['prominences'][top2[0]], props['prominences'][top2[1]])
            scores.append((gi, score))
        scores.sort(key=lambda x: -x[1])
        return [s[0] for s in scores[:args.bimodal]]

    if args.uniform:
        from scipy.stats import kstest
        scores = []
        for gi in np.where(expressed)[0]:
            vals = cpm[gi, :]
            vmin, vmax = vals.min(), vals.max()
            if vmax - vmin < 1:
                continue
            normed = (vals - vmin) / (vmax - vmin)
            ks_stat, _ = kstest(normed, 'uniform')
            scores.append((gi, ks_stat))
        scores.sort(key=lambda x: x[1])
        return [s[0] for s in scores[:args.uniform]]

    if args.similar:
        center = find_gene(args.similar, gene_names)
        if center is None:
            print(f"Gene '{args.similar}' not found")
            sys.exit(1)
        center_curve = np.sort(log_expr[center, :])[::-1]
        pool = np.where(expressed)[0]
        dists = []
        for gi in pool:
            curve = np.sort(log_expr[gi, :])[::-1]
            mse = np.mean((curve - center_curve) ** 2)
            dists.append((gi, mse))
        dists.sort(key=lambda x: x[1])
        # First one is self (mse=0), include it
        return [d[0] for d in dists[:args.n]]

    return []


def plot_ranked(ax, gene_indices, log_expr, cpm, gene_names, n_samples, args):
    """Plot ranked expression for given genes on one axis."""
    use_log = not args.linear

    for j, gi in enumerate(gene_indices):
        color = COLORS[j % len(COLORS)]
        expr_data = log_expr if use_log else cpm
        vals_sorted = np.sort(expr_data[gi, :])[::-1]
        name = gene_names[gi]
        mean_val = expr_data[gi, :].mean()
        unit = 'log₂CPM' if use_log else 'CPM'
        label = f'{name} (μ={mean_val:.1f} {unit})'

        if args.density:
            kde = gaussian_kde(expr_data[gi, :], bw_method='scott')
            densities = kde(vals_sorted)
            d_min, d_max = densities.min(), densities.max()
            if d_max > d_min:
                alphas = 0.08 + 0.9 * (densities - d_min) / (d_max - d_min)
            else:
                alphas = np.full_like(densities, 0.5)
            alphas = np.clip(alphas, 0.0, 0.99)
            base = np.array(to_rgba(color))
            rgba = np.tile(base, (n_samples, 1))
            rgba[:, 3] = alphas
            ax.scatter(np.arange(n_samples), vals_sorted, s=4, c=rgba,
                       edgecolors='none', label=label, zorder=2 + j)
        elif args.train_test:
            rng = np.random.default_rng(args.seed)
            perm = rng.permutation(n_samples)
            train_idx, test_idx = perm[:700], perm[700:]
            train_vals = np.sort(expr_data[gi, train_idx])[::-1]
            test_vals = expr_data[gi, test_idx]
            test_ranks = np.searchsorted(-train_vals, -test_vals)
            ax.scatter(np.arange(700), train_vals, s=3, color=color,
                       alpha=0.4, edgecolors='none', label=f'{name} train')
            ax.scatter(test_ranks, test_vals, s=12, color='#f78166',
                       alpha=0.7, edgecolors='none', label=f'{name} test', zorder=5)
        else:
            ax.plot(np.arange(n_samples), vals_sorted, color=color, lw=1.5,
                    alpha=0.8, label=label)

        # KDE inset
        if args.kde and j == 0 and len(gene_indices) <= 3:
            inset = ax.inset_axes([0.55, 0.55, 0.42, 0.42])
            inset.set_facecolor(BG)
            for k, gi2 in enumerate(gene_indices):
                vals2 = (log_expr if use_log else cpm)[gi2, :]
                xs = np.linspace(max(0, vals2.min() - 0.3), vals2.max() + 0.3, 500)
                kde = gaussian_kde(vals2, bw_method='scott')
                c = COLORS[k % len(COLORS)]
                inset.fill_between(xs, kde(xs), alpha=0.2, color=c)
                inset.plot(xs, kde(xs), color=c, lw=1.5)
            inset.tick_params(labelsize=6, colors=MUTED)
            inset.set_xlabel(unit, fontsize=6, color=MUTED)
            for spine in inset.spines.values():
                spine.set_edgecolor(GRID)

    # sc overlay
    if args.sc:
        try:
            sc_log, sc_cpm, sc_names, sc_ids = load_sc()
            sc_data = sc_log if use_log else sc_cpm
            sc_idx_map = {n.upper(): i for i, n in enumerate(sc_names)}
            for gi in gene_indices:
                gname = gene_names[gi].upper()
                si = sc_idx_map.get(gname)
                if si is None:
                    continue
                for di in range(len(sc_ids)):
                    val = sc_data[si, di]
                    # Find rank position
                    vals_sorted = np.sort((log_expr if use_log else cpm)[gi, :])[::-1]
                    rank = np.searchsorted(-vals_sorted, -val)
                    ax.scatter(rank, val, s=40, color='#f78166', marker='*',
                               zorder=10, edgecolors='white', linewidths=0.5)
        except Exception as e:
            print(f"Warning: could not load sc data: {e}")

    ylabel = 'log₂(CPM + 1)' if use_log else 'CPM'
    ax.set_xlabel('Sample rank (high → low)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT,
              markerscale=2)
    ax.grid(alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description='Quick gene expression visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Gene selection
    parser.add_argument('genes', nargs='*', help='Gene names to plot')
    parser.add_argument('--top', type=int, help='Top N genes by expression/variance')
    parser.add_argument('--by', default='mean', choices=['mean', 'variance', 'cv'],
                        help='Sort criterion for --top')
    parser.add_argument('--random', type=int, help='N random expressed genes')
    parser.add_argument('--bimodal', type=int, help='Top N bimodal genes')
    parser.add_argument('--uniform', type=int, help='Top N most uniform genes')
    parser.add_argument('--similar', type=str, help='Find genes similar to this one')
    parser.add_argument('--n', type=int, default=5, help='Number of similar genes')
    parser.add_argument('--archetypes', type=int, help='Cluster into N shape archetypes')

    # Display options
    parser.add_argument('--linear', action='store_true', help='CPM scale (no log)')
    parser.add_argument('--kde', action='store_true', help='Show KDE inset')
    parser.add_argument('--density', action='store_true', help='Dot opacity = KDE density')
    parser.add_argument('--train-test', action='store_true', help='700/103 train/test split')
    parser.add_argument('--sc', action='store_true', help='Overlay scRNA pseudobulk donors')
    parser.add_argument('--sort', default='rank', choices=['rank', 'pc1', 'pc2'],
                        help='How to sort samples on x-axis')

    # Layout
    parser.add_argument('--per-panel', type=int, default=None,
                        help='Genes per panel (creates multi-panel grid)')
    parser.add_argument('--panels', type=int, default=None,
                        help='Number of panels (overrides auto)')

    # Output
    parser.add_argument('--out', default=OUT_PATH, help='Output path')
    parser.add_argument('--open', action='store_true', help='Open image after saving')
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--source', default='gtex',
                        choices=['gtex', 'gse279480'],
                        help='Data source (default: gtex whole blood)')
    parser.add_argument('--stimulation', default='Null',
                        help='For --source gse279480: which stimulation to load '
                             '(Null/LPS/Poly I:C/SEB)')

    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────
    if args.source == 'gse279480':
        print(f"Loading GSE279480 ({args.stimulation})...")
        log_expr, cpm, gene_names, n_samples = load_gse279480(args.stimulation)
    else:
        print("Loading GTEx data...")
        log_expr, cpm, gene_names, n_samples = load_gtex()

    # ── Archetypes mode ───────────────────────────────────────────────
    if args.archetypes:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        setup_style()
        mask = (log_expr.mean(axis=1) > 1) & ~np.array([n.startswith('ENSG') for n in gene_names])
        expressed = np.where(mask)[0]
        expr_data = cpm if args.linear else log_expr

        sorted_curves = np.zeros((len(expressed), n_samples))
        for i, gi in enumerate(expressed):
            sorted_curves[i] = np.sort(expr_data[gi, :])[::-1]

        scaler = StandardScaler()
        normed = scaler.fit_transform(sorted_curves.T).T
        ds = np.linspace(0, n_samples - 1, 100).astype(int)

        km = KMeans(n_clusters=args.archetypes, n_init=10, random_state=args.seed)
        labels = km.fit_predict(normed[:, ds])
        counts = np.bincount(labels, minlength=args.archetypes)
        order = np.argsort(counts)[::-1]

        ncols = min(5, args.archetypes)
        nrows = (args.archetypes + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)

        unit = 'CPM' if args.linear else 'log₂(CPM+1)'
        fig.suptitle(f'{args.archetypes} Shape Archetypes — {unit}',
                     fontsize=16, fontweight='bold', color=TEXT, y=0.995)
        cmap = plt.cm.viridis

        for rank, ci in enumerate(order):
            r, c = divmod(rank, ncols)
            ax = axes[r, c]
            members = np.where(labels == ci)[0]
            rng = np.random.default_rng(args.seed)
            show = members if len(members) <= 80 else rng.choice(members, 80, replace=False)
            for j, mi in enumerate(show):
                ax.plot(np.arange(n_samples), sorted_curves[mi],
                        color=cmap(j / len(show)), lw=0.5, alpha=0.35)
            closest = members[np.argsort(np.mean((normed[members] - normed[members].mean(axis=0))**2, axis=1))[:3]]
            rep = '\n'.join([gene_names[expressed[m]] for m in closest])
            ax.text(0.02, 0.95, rep, transform=ax.transAxes, fontsize=7, color=TEXT,
                    va='top', fontfamily='monospace',
                    bbox=dict(facecolor=BG, alpha=0.7, edgecolor=GRID, pad=2))
            ax.set_title(f'#{rank+1} ({len(members)} genes)', fontsize=10,
                         fontweight='bold', color=TEXT, pad=5)
            ax.set_xlabel('Rank', fontsize=8)
            ax.set_ylabel(unit, fontsize=8)
            ax.grid(alpha=0.3)

        for idx in range(len(order), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor=BG)
        plt.close(fig)
        print(f"Saved: {args.out}")
        if args.open:
            subprocess.run(['open', args.out])
        return

    # ── Select genes ──────────────────────────────────────────────────
    gene_indices = select_genes(args, log_expr, cpm, gene_names)
    if not gene_indices:
        print("No genes selected. Use gene names, --top, --random, --bimodal, --uniform, or --similar.")
        sys.exit(1)

    print(f"Selected {len(gene_indices)} genes")

    # ── PC sorting ────────────────────────────────────────────────────
    pc_order = None
    if args.sort in ('pc1', 'pc2'):
        from sklearn.decomposition import PCA
        gene_stds = log_expr.std(axis=1)
        mask = (log_expr.mean(axis=1) > 1)
        top_var = np.where(mask)[0][np.argsort(gene_stds[mask])[::-1][:5000]]
        pca = PCA(n_components=2)
        sample_pcs = pca.fit_transform(log_expr[top_var].T)
        pc_idx = 0 if args.sort == 'pc1' else 1
        pc_order = np.argsort(sample_pcs[:, pc_idx])
        print(f"Sorting by {args.sort} ({pca.explained_variance_ratio_[pc_idx]*100:.1f}% variance)")

    # ── Layout ────────────────────────────────────────────────────────
    setup_style()

    per_panel = args.per_panel or len(gene_indices)
    n_panels = args.panels or ((len(gene_indices) + per_panel - 1) // per_panel)
    groups = [gene_indices[i * per_panel:(i + 1) * per_panel] for i in range(n_panels)]

    if n_panels == 1:
        fig, ax = plt.subplots(figsize=(12, 7))
        axes_flat = [ax]
    else:
        ncols = min(5, n_panels)
        nrows = (n_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        axes_flat = axes.ravel() if n_panels > 1 else [axes]

    unit = 'CPM' if args.linear else 'log₂(CPM+1)'
    sort_label = args.sort.upper() if args.sort != 'rank' else 'rank'

    for panel_idx, group in enumerate(groups):
        ax = axes_flat[panel_idx]

        if pc_order is not None:
            # PC-sorted mode
            expr_data = log_expr if not args.linear else cpm
            from matplotlib.colors import to_rgba as _tr
            from sklearn.decomposition import PCA as _PCA
            pc_idx = 0 if args.sort == 'pc1' else 1
            for j, gi in enumerate(group):
                color = COLORS[j % len(COLORS)]
                vals = expr_data[gi, pc_order]
                ax.scatter(np.arange(n_samples), vals, s=4, color=color,
                           alpha=0.5, edgecolors='none',
                           label=f'{gene_names[gi]} (μ={expr_data[gi].mean():.1f})')
            ax.set_xlabel(f'Sample (sorted by {args.sort})', fontsize=10)
            ax.set_ylabel(unit, fontsize=10)
            ax.legend(fontsize=7, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
            ax.grid(alpha=0.3)
        else:
            plot_ranked(ax, group, log_expr, cpm, gene_names, n_samples, args)

    # Hide unused panels
    for idx in range(len(groups), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Saved: {args.out}")

    if args.open:
        subprocess.run(['open', args.out])


if __name__ == '__main__':
    main()
