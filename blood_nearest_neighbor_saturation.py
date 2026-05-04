"""
Nearest-neighbor saturation — whole transcriptome
=================================================
For each held-out sample h and each train-pool size n, find the train
sample (its full transcriptome vector) that minimises MSE to h:

    nn_mse(h, n) = min_{i in train[:n]}  mean_g (h_g - train_i_g)^2

If the training pool covers "what blood can be", every new blood
sample should eventually find a close twin already in the pool, and
nn_mse should plateau as n grows.

Operates on whole transcriptome: 70,821 genes (drop only all-zero),
log2(CPM+1).  Same shuffle / 50 holdouts as the earlier analyses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

GCT = Path('/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz')
HOLDOUT = 50
TRAIN_SIZES = [100, 200, 300, 400, 500, 600, 700]
SEED = 0

BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})


def load_whole_transcriptome():
    df = pd.read_csv(GCT, sep='\t', skiprows=2, compression='gzip')
    expr = df.iloc[:, 2:].values.astype(np.float64)
    n_genes, n_samples = expr.shape
    print(f"Raw GCT: {n_genes:,} genes x {n_samples} samples")
    nonzero = (expr > 0).any(axis=1)
    expr = expr[nonzero]
    print(f"After dropping all-zero genes: {expr.shape[0]:,} genes")
    cpm = expr / expr.sum(axis=0) * 1e6
    expr_log = np.log2(cpm + 1)
    return expr_log.T  # samples x genes


def split(X):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    test = X[-HOLDOUT:]
    train_pool = X[:-HOLDOUT]
    return train_pool, test


def pairwise_mse(H, T):
    """MSE between every pair (H_i, T_j) — H: holdout, T: train.  Returns (|H|, |T|)."""
    # ||H - T||^2 = ||H||^2 + ||T||^2 - 2 H·T   (sum over genes)
    n_genes = H.shape[1]
    H2 = (H ** 2).sum(axis=1, keepdims=True)         # (|H|, 1)
    T2 = (T ** 2).sum(axis=1, keepdims=True).T       # (1, |T|)
    cross = H @ T.T                                  # (|H|, |T|)
    sq = H2 + T2 - 2.0 * cross
    np.maximum(sq, 0.0, out=sq)                      # numerical safety
    return sq / n_genes                              # MSE


def main():
    X = load_whole_transcriptome()
    print(f"Working matrix: {X.shape[0]} samples x {X.shape[1]:,} genes")
    train_pool, test = split(X)
    print(f"Train pool: {train_pool.shape[0]} | Holdout: {test.shape[0]}")

    # One pass: pairwise MSE between every holdout and every train-pool sample.
    print("Computing 50 x 753 pairwise MSE matrix ...")
    D = pairwise_mse(test, train_pool)               # (50, 753)
    print(f"  D shape={D.shape}, min={D.min():.5f}, max={D.max():.5f}")

    rows = []
    nn_per_n = {}
    for n in TRAIN_SIZES:
        nn = D[:, :n].min(axis=1)                    # nearest-neighbor MSE per holdout
        nn_per_n[n] = nn
        rows.append((n, nn.mean(), nn.std(), nn.min(), nn.max(), np.median(nn)))
        print(f"  n={n:4d}  NN-MSE  mean={nn.mean():.5f}  median={np.median(nn):.5f}  "
              f"min={nn.min():.5f}  max={nn.max():.5f}  sd={nn.std():.5f}")

    res = pd.DataFrame(rows, columns=['n', 'nn_mse_mean', 'nn_mse_sd',
                                      'nn_mse_min', 'nn_mse_max',
                                      'nn_mse_median'])
    res['drop_pct'] = res['nn_mse_mean'].pct_change() * 100
    res.to_csv('blood_nn_saturation.csv', index=False)
    print("\nWrote blood_nn_saturation.csv")
    print(res.to_string(index=False))

    # ── PLOT ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.45, height_ratios=[3, 1.5, 2])
    fig.suptitle(f'Nearest-neighbor saturation — whole transcriptome '
                 f'({X.shape[1]:,} genes, holdout={HOLDOUT}, seed={SEED})',
                 color=TEXT, fontsize=13)

    # (1) Mean NN-MSE vs n with worst-case envelope
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(res['n'], res['nn_mse_min'], res['nn_mse_max'],
                     color='#79c0ff', alpha=0.18, label='min/max over 50 holdouts')
    ax1.errorbar(res['n'], res['nn_mse_mean'], yerr=res['nn_mse_sd'],
                 fmt='o-', color='#58a6ff', capsize=4, lw=2.5, markersize=9,
                 label='mean ± sd')
    ax1.plot(res['n'], res['nn_mse_median'], 's--', color='#3fb950',
             lw=1.8, markersize=7, label='median')
    asym = res['nn_mse_mean'].iloc[-1]
    ax1.axhline(asym, color=MUTED, ls=':', lw=1, alpha=0.7,
                label=f'mean at n=700 ({asym:.4f})')
    for _, row in res.iterrows():
        ax1.annotate(f"{row['nn_mse_mean']:.4f}",
                     (row['n'], row['nn_mse_mean']),
                     textcoords='offset points', xytext=(0, 12),
                     ha='center', color=TEXT, fontsize=9)
    ax1.set_ylabel('Nearest-neighbor MSE\n(log2-CPM units²)')
    ax1.set_title('Closest train transcriptome to each held-out sample, vs train size',
                  fontsize=12)
    ax1.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, loc='upper right',
               fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (2) Marginal drop
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    drops = res['drop_pct'].iloc[1:]
    ax2.bar(res['n'].iloc[1:], -drops, width=60, color='#3fb950', alpha=0.85,
            edgecolor='#1f6f33')
    for n, d in zip(res['n'].iloc[1:], -drops):
        ax2.annotate(f'-{d:.1f}%', (n, d),
                     textcoords='offset points', xytext=(0, 4),
                     ha='center', color=TEXT, fontsize=9)
    ax2.set_xlabel('Train size n')
    ax2.set_ylabel('Drop vs prev n (%)')
    ax2.set_title('Marginal benefit of +100 more samples', fontsize=12)
    ax2.set_xticks(res['n'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(-drops) * 1.25)

    # (3) Per-holdout NN-MSE distribution at each n (strip plot)
    ax3 = fig.add_subplot(gs[2])
    rng = np.random.default_rng(1)
    for n in TRAIN_SIZES:
        y = nn_per_n[n]
        x = n + rng.uniform(-15, 15, size=len(y))
        ax3.scatter(x, y, color='#f78166', alpha=0.6, s=22, edgecolor='none')
    ax3.plot(res['n'], res['nn_mse_mean'], 'o-', color='#58a6ff', lw=2,
             markersize=8, label='mean')
    ax3.set_xlabel('Train size n')
    ax3.set_ylabel('NN-MSE per held-out sample')
    ax3.set_title('Distribution across 50 held-out samples',
                  fontsize=12)
    ax3.set_xticks(res['n'])
    ax3.grid(True, alpha=0.3)
    ax3.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    plt.tight_layout()
    out = 'blood_nearest_neighbor_saturation.png'
    plt.savefig(out, dpi=140, facecolor=BG, bbox_inches='tight')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
