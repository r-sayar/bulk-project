"""
Blood sample-space saturation
=============================
Question: do 750 GTEx whole-blood samples already cover "what blood can be",
or would more samples reveal new structure?

Test:
  - Hold out the last 50 samples.
  - For training sizes n in {100, 200, 300, 400, 500, 600, 700}:
      fit a model on n samples, then measure how well it reconstructs
      the held-out 50.
  - If the residual variance plateaus, the sample space is saturated.

Two complementary models, both run on log2(CPM+1):
  (1) PCA reconstruction with k=50 components — residual MSE per held-out
      sample after projecting onto the train PCs (centered on train mean).
  (2) Per-gene Gaussian — train mean/std per gene, test = mean squared
      z-score of held-out samples.  Saturates when train mean/std stop
      moving as n grows.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

GCT = Path('/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz')
HOLDOUT = 50
TRAIN_SIZES = [100, 200, 300, 400, 500, 600, 700]
N_PCS = 50
CPM_THRESHOLD = 1
MIN_SAMPLE_FRAC = 0.1
SEED = 0

BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})


def load_and_preprocess():
    df = pd.read_csv(GCT, sep='\t', skiprows=2, compression='gzip')
    expr = df.iloc[:, 2:].values.astype(np.float64)  # genes x samples
    n_genes, n_samples = expr.shape
    print(f"Raw: {n_genes:,} genes x {n_samples} samples")

    # CPM filter
    cpm = expr / expr.sum(axis=0) * 1e6
    keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_samples)
    expr = expr[keep]
    print(f"After filter: {expr.shape[0]:,} genes")

    # CPM + log2
    expr_cpm = expr / expr.sum(axis=0) * 1e6
    expr_log = np.log2(expr_cpm + 1)  # genes x samples
    return expr_log.T  # samples x genes  (rows = observations)


def split(X):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    if X.shape[0] < max(TRAIN_SIZES) + HOLDOUT:
        raise ValueError(f"Only {X.shape[0]} samples; need >= {max(TRAIN_SIZES) + HOLDOUT}")
    test = X[-HOLDOUT:]
    train_pool = X[:-HOLDOUT]
    return train_pool, test


def pca_recon_mse(train, test, k):
    """Mean squared reconstruction error of `test` using top-k PCs fit on `train`."""
    mu = train.mean(axis=0)
    Xc = train - mu
    # Use truncated SVD via numpy on Xc (samples x genes). Components = right
    # singular vectors. For n_samples << n_genes, do SVD on Xc directly.
    # Limit k to min(n-1, k).
    k_eff = min(k, train.shape[0] - 1, train.shape[1])
    # Economy SVD
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt[:k_eff].T  # genes x k_eff
    Tc = test - mu
    proj = Tc @ V                # holdout x k_eff
    recon = proj @ V.T           # holdout x genes
    resid = Tc - recon
    # Per-sample MSE, then average
    mse_per_sample = (resid ** 2).mean(axis=1)
    return mse_per_sample.mean(), mse_per_sample.std()


def gauss_zscore(train, test):
    """Mean squared z-score of test samples under per-gene Gaussian fit on train."""
    mu = train.mean(axis=0)
    sd = train.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-6, 1e-6, sd)  # guard against zero-variance genes
    z = (test - mu) / sd
    msz_per_sample = (z ** 2).mean(axis=1)
    return msz_per_sample.mean(), msz_per_sample.std()


def main():
    X = load_and_preprocess()
    print(f"Working matrix: {X.shape[0]} samples x {X.shape[1]:,} genes")
    train_pool, test = split(X)
    print(f"Train pool: {train_pool.shape[0]} | Holdout: {test.shape[0]}")

    rows = []
    for n in TRAIN_SIZES:
        train = train_pool[:n]
        pca_mean, pca_std = pca_recon_mse(train, test, N_PCS)
        z_mean, z_std = gauss_zscore(train, test)
        rows.append((n, pca_mean, pca_std, z_mean, z_std))
        print(f"  n={n:4d}  PCA-MSE={pca_mean:.5f} (+/-{pca_std:.5f})  "
              f"meanZ^2={z_mean:.4f} (+/-{z_std:.4f})")

    res = pd.DataFrame(rows, columns=['n', 'pca_mse', 'pca_mse_sd',
                                      'mean_z2', 'mean_z2_sd'])
    res.to_csv('blood_saturation.csv', index=False)
    print("\nWrote blood_saturation.csv")

    # Decay metric: relative drop between consecutive n
    res['pca_drop_pct'] = res['pca_mse'].pct_change() * 100
    print("\nRelative change in PCA-MSE between consecutive n:")
    print(res[['n', 'pca_mse', 'pca_drop_pct']].to_string(index=False))

    # Plot — saturation curve (top) + marginal drop (bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1.5]})
    fig.suptitle(f'Blood sample-space saturation — holdout = last {HOLDOUT} '
                 f'samples (n_pool=753, seed={SEED})',
                 color=TEXT, fontsize=14)

    # --- top: residual variance vs n ---
    ax1.errorbar(res['n'], res['pca_mse'], yerr=res['pca_mse_sd'],
                 fmt='o-', color='#f78166', capsize=4, lw=2.5, markersize=9,
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
    ax1.set_title('Residual variance of held-out samples vs train size',
                  color=TEXT, fontsize=12)
    ax1.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- bottom: marginal % drop between consecutive n ---
    drops = res['pca_drop_pct'].iloc[1:]
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
    out = 'blood_sample_saturation.png'
    plt.savefig(out, dpi=140, facecolor=BG, bbox_inches='tight')
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
