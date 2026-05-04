"""
PCA-MSE saturation curve — WHOLE TRANSCRIPTOME (no expression filter).

Same protocol as blood_sample_saturation.py, but instead of restricting
to expressed genes (CPM>1 in >=10% samples → 16,355 genes), we keep
every gene that has any signal at all (drop only all-zero genes).
This tells us whether the train pool covers the *whole transcriptome*
of new blood samples, including the long tail of sparse / sometimes-on
genes that the expressed-gene filter would discard.

Output: residual variance of held-out 50 samples vs train size n.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

GCT = Path('/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz')
HOLDOUT = 50
TRAIN_SIZES = [100, 200, 300, 400, 500, 600, 700]
N_PCS = 50
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
    expr = df.iloc[:, 2:].values.astype(np.float64)  # genes x samples
    n_genes, n_samples = expr.shape
    print(f"Raw GCT: {n_genes:,} genes x {n_samples} samples")

    # Drop only all-zero genes (no information). Everything else stays.
    nonzero = (expr > 0).any(axis=1)
    expr = expr[nonzero]
    print(f"After dropping all-zero genes: {expr.shape[0]:,} genes "
          f"({nonzero.sum() - (expr.sum(axis=1) > 0).sum()} sanity-check)")

    # CPM + log2(CPM+1) on the unfiltered matrix.
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


def pca_recon_mse(train, test, k):
    mu = train.mean(axis=0)
    Xc = train - mu
    k_eff = min(k, train.shape[0] - 1, train.shape[1])
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k_eff].T
    Tc = test - mu
    proj = Tc @ V
    recon = proj @ V.T
    resid = Tc - recon
    mse_per_sample = (resid ** 2).mean(axis=1)
    return mse_per_sample.mean(), mse_per_sample.std()


def main():
    X = load_whole_transcriptome()
    print(f"Working matrix: {X.shape[0]} samples x {X.shape[1]:,} genes")
    train_pool, test = split(X)
    print(f"Train pool: {train_pool.shape[0]} | Holdout: {test.shape[0]}")

    rows = []
    for n in TRAIN_SIZES:
        train = train_pool[:n]
        m, s = pca_recon_mse(train, test, N_PCS)
        rows.append((n, m, s))
        print(f"  n={n:4d}  PCA-MSE={m:.6f} (+/-{s:.6f})")

    res = pd.DataFrame(rows, columns=['n', 'pca_mse', 'pca_mse_sd'])
    res['drop_pct'] = res['pca_mse'].pct_change() * 100
    res.to_csv('blood_saturation_full_transcriptome.csv', index=False)
    print("\nWrote blood_saturation_full_transcriptome.csv")
    print(res.to_string(index=False))

    # ── PLOT — same layout as the filtered-gene version for direct comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1.5]})
    fig.suptitle(f'Whole-transcriptome saturation — holdout = last '
                 f'{HOLDOUT} samples ({X.shape[1]:,} genes, no expression '
                 f'filter, seed={SEED})', color=TEXT, fontsize=13)

    ax1.errorbar(res['n'], res['pca_mse'], yerr=res['pca_mse_sd'],
                 fmt='o-', color='#d2a8ff', capsize=4, lw=2.5, markersize=9,
                 label=f'PCA-{N_PCS} reconstruction MSE on holdout')
    asym = res['pca_mse'].iloc[-1]
    ax1.axhline(asym, color=MUTED, ls='--', lw=1, alpha=0.7,
                label=f'value at n=700 ({asym:.4f})')
    for _, row in res.iterrows():
        ax1.annotate(f"{row['pca_mse']:.4f}",
                     (row['n'], row['pca_mse']),
                     textcoords='offset points', xytext=(0, 12),
                     ha='center', color=TEXT, fontsize=9)
    ax1.set_ylabel('Best-fit residual variance\n(log2-CPM units²)')
    ax1.set_title('Residual variance of held-out samples vs train size '
                  '(whole transcriptome)', color=TEXT, fontsize=12)
    ax1.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, loc='upper right')
    ax1.grid(True, alpha=0.3)

    drops = res['drop_pct'].iloc[1:]
    ax2.bar(res['n'].iloc[1:], -drops, width=60, color='#3fb950', alpha=0.85,
            edgecolor='#1f6f33')
    for n, d in zip(res['n'].iloc[1:], -drops):
        ax2.annotate(f'-{d:.1f}%', (n, d),
                     textcoords='offset points', xytext=(0, 4),
                     ha='center', color=TEXT, fontsize=9)
    ax2.set_xlabel('Train size n')
    ax2.set_ylabel('Drop vs previous n (%)')
    ax2.set_title('Marginal benefit of +100 more samples',
                  color=TEXT, fontsize=12)
    ax2.set_xticks(res['n'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, max(-drops) * 1.25)

    plt.tight_layout()
    out = 'blood_saturation_full_transcriptome.png'
    plt.savefig(out, dpi=140, facecolor=BG, bbox_inches='tight')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
