"""
KDE coverage check at n=700
===========================
For each gene we fit a 1D Gaussian KDE on the 700 training samples.
Question: do the 50 held-out samples land in high-probability regions
of those KDEs?

Headline metric — HDR-q coverage:
    For each gene, threshold = (1-q)-quantile of KDE density evaluated
    at training points. A held-out value is "in the q-HDR" if its
    density >= threshold. Under the null (holdout from the same
    distribution), expected coverage = q.

We compute:
  - per-gene HDR-90 coverage (mean across 50 holdouts)
  - per-sample HDR-90 coverage (mean across all genes) — answers
    "does this new blood sample have most genes in high-prob regions?"
  - density-rank distribution over all (gene, holdout) pairs — uniform
    if train captures the marginals
  - example gene panels with train KDE + holdout overlay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde
import time

GCT = Path('/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz')
HOLDOUT = 50
TRAIN = 700
HDR_Q = 0.9
SEED = 0
CPM_THRESHOLD = 1
MIN_SAMPLE_FRAC = 0.1

BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})


def load_and_preprocess():
    df = pd.read_csv(GCT, sep='\t', skiprows=2, compression='gzip')
    expr = df.iloc[:, 2:].values.astype(np.float64)
    gene_names = df['Description'].values
    n_genes, n_samples = expr.shape
    cpm = expr / expr.sum(axis=0) * 1e6
    keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= int(MIN_SAMPLE_FRAC * n_samples)
    expr = expr[keep]
    gene_names = gene_names[keep]
    expr_cpm = expr / expr.sum(axis=0) * 1e6
    expr_log = np.log2(expr_cpm + 1)
    print(f"After filter: {expr.shape[0]:,} genes x {n_samples} samples")
    return expr_log.T, gene_names  # samples x genes


def split(X):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    test = X[-HOLDOUT:]
    train = X[:TRAIN]
    return train, test


def per_gene_kde_check(train, test, q):
    n_genes = train.shape[1]
    coverage = np.zeros(n_genes)
    rank_matrix = np.zeros((HOLDOUT, n_genes))
    in_hdr = np.zeros((HOLDOUT, n_genes), dtype=bool)
    flat_genes = 0
    threshold_q = 1.0 - q

    for g in range(n_genes):
        x_train = train[:, g]
        x_test = test[:, g]

        if x_train.std() < 1e-9:
            # Degenerate — gene constant in train. Coverage trivially 1
            # if test also matches, else 0. Cheap check.
            same = np.isclose(x_test, x_train[0])
            coverage[g] = same.mean()
            rank_matrix[:, g] = same.astype(float)
            in_hdr[:, g] = same
            flat_genes += 1
            continue

        kde = gaussian_kde(x_train)
        d_train = kde(x_train)
        d_test = kde(x_test)
        thresh = np.quantile(d_train, threshold_q)
        in_hdr[:, g] = d_test >= thresh
        coverage[g] = in_hdr[:, g].mean()
        sorted_d_train = np.sort(d_train)
        rank_matrix[:, g] = np.searchsorted(sorted_d_train, d_test) / len(d_train)

        if g % 2000 == 0 and g > 0:
            print(f"  gene {g}/{n_genes}")

    print(f"  {flat_genes} flat genes (std<1e-9 in train)")
    return coverage, rank_matrix, in_hdr


def main():
    X, gene_names = load_and_preprocess()
    train, test = split(X)
    print(f"Train: {train.shape}, Test: {test.shape}")

    t0 = time.time()
    coverage, rank_matrix, in_hdr = per_gene_kde_check(train, test, HDR_Q)
    print(f"Per-gene KDE done in {time.time() - t0:.1f}s")

    sample_coverage = in_hdr.mean(axis=1)  # one value per held-out sample

    print(f"\n=== HDR-{int(HDR_Q*100)} coverage ===")
    print(f"Per-gene coverage (avg over 50 holdouts):")
    print(f"  mean   = {coverage.mean():.4f}  (expected ~{HDR_Q})")
    print(f"  median = {np.median(coverage):.4f}")
    print(f"  std    = {coverage.std():.4f}")
    print(f"  fraction genes >= 0.80: {(coverage >= 0.80).mean():.4f}")
    print(f"  fraction genes >= 0.70: {(coverage >= 0.70).mean():.4f}")

    print(f"\nPer-sample coverage (avg over all genes):")
    print(f"  mean   = {sample_coverage.mean():.4f}")
    print(f"  median = {np.median(sample_coverage):.4f}")
    print(f"  min    = {sample_coverage.min():.4f}")
    print(f"  max    = {sample_coverage.max():.4f}")

    print(f"\nDensity-rank distribution over all {rank_matrix.size:,} "
          f"(gene, holdout) pairs:")
    print(f"  mean = {rank_matrix.mean():.4f}  (uniform => 0.5)")
    print(f"  fraction with rank < 0.10 (low-density tails): "
          f"{(rank_matrix < 0.10).mean():.4f}  (uniform => 0.10)")

    # ── PLOT ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)
    fig.suptitle(f'Per-gene KDE coverage at n={TRAIN} — holdout = {HOLDOUT} '
                 f'samples (HDR-{int(HDR_Q*100)})', fontsize=14)

    # (1) Per-sample coverage — answers the headline question directly
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(sample_coverage, bins=20, color='#3fb950', edgecolor='#1f6f33')
    ax.axvline(HDR_Q, color='#f78166', ls='--', lw=2,
               label=f'expected = {HDR_Q}')
    ax.axvline(sample_coverage.mean(), color='w', ls=':', lw=1.5,
               label=f'observed mean = {sample_coverage.mean():.3f}')
    ax.set_xlabel('Fraction of genes in 90%-HDR')
    ax.set_ylabel('# held-out samples')
    ax.set_title('Per-sample HDR-90 coverage', fontsize=11)
    ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # (2) Per-gene coverage
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(coverage, bins=40, color='#58a6ff', edgecolor='#1f4e8f')
    ax.axvline(HDR_Q, color='#f78166', ls='--', lw=2, label=f'expected = {HDR_Q}')
    ax.axvline(coverage.mean(), color='w', ls=':', lw=1.5,
               label=f'observed = {coverage.mean():.3f}')
    ax.set_xlabel('Per-gene HDR-90 coverage')
    ax.set_ylabel('# genes')
    ax.set_title('Per-gene HDR-90 coverage', fontsize=11)
    ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # (3) Density-rank distribution (should be ~uniform if blood captured)
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(rank_matrix.flatten(), bins=50, color='#d2a8ff',
            edgecolor='#7c5fbf', density=True)
    ax.axhline(1.0, color='#f78166', ls='--', lw=2, label='uniform')
    ax.set_xlabel('Density rank of holdout in train KDE')
    ax.set_ylabel('density')
    ax.set_title('PIT-style rank histogram (all gene×holdout pairs)',
                 fontsize=11)
    ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    # (4-9) Example gene panels — pick a spread by per-gene std on train
    train_stds = train.std(axis=0)
    pick_idx = []
    pick_idx += list(np.argsort(train_stds)[-2:])             # most variable
    pick_idx += list(np.argsort(train_stds)[len(train_stds) // 2 - 1:
                                            len(train_stds) // 2 + 1])  # median
    pick_idx += list(np.argsort(train_stds)[:2])              # least variable
    pick_idx = pick_idx[:6]

    for i, g in enumerate(pick_idx):
        row, col = 1 + i // 3, i % 3
        ax = fig.add_subplot(gs[row, col])
        x_train = train[:, g]
        x_test = test[:, g]
        if x_train.std() < 1e-9:
            ax.set_title(f'{gene_names[g]} (constant)', fontsize=10)
            continue
        kde = gaussian_kde(x_train)
        x_grid = np.linspace(min(x_train.min(), x_test.min()) - 0.5,
                             max(x_train.max(), x_test.max()) + 0.5, 400)
        d_grid = kde(x_grid)
        # HDR-90 region = density >= threshold
        thresh = np.quantile(kde(x_train), 1 - HDR_Q)
        ax.fill_between(x_grid, 0, d_grid, where=(d_grid >= thresh),
                        color='#3fb950', alpha=0.25, label='90% HDR')
        ax.fill_between(x_grid, 0, d_grid, where=(d_grid < thresh),
                        color='#f78166', alpha=0.18, label='outside HDR')
        ax.plot(x_grid, d_grid, color='#58a6ff', lw=1.5)
        # train rug
        ax.plot(x_train, np.full_like(x_train, -d_grid.max() * 0.04),
                '|', color=MUTED, alpha=0.4, markersize=6)
        # holdout markers
        d_test = kde(x_test)
        in_hdr_g = d_test >= thresh
        ax.scatter(x_test[in_hdr_g], np.full(in_hdr_g.sum(), d_grid.max() * 0.05),
                   color='#3fb950', s=40, edgecolor='w', linewidth=0.5,
                   zorder=5, label=f'holdout in HDR ({in_hdr_g.sum()})')
        ax.scatter(x_test[~in_hdr_g],
                   np.full((~in_hdr_g).sum(), d_grid.max() * 0.05),
                   color='#f78166', s=40, edgecolor='w', linewidth=0.5,
                   zorder=5, label=f'outside ({(~in_hdr_g).sum()})')
        ax.set_xlabel('log2(CPM+1)')
        ax.set_ylabel('density')
        ax.set_title(f'{gene_names[g]} (std={train_stds[g]:.2f}, '
                     f'cov={coverage[g]:.2f})', fontsize=10)
        if i == 0:
            ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT,
                      fontsize=8, loc='upper right')

    out = 'blood_kde_coverage.png'
    plt.savefig(out, dpi=140, facecolor=BG, bbox_inches='tight')
    print(f"\nWrote {out}")

    pd.DataFrame({
        'gene': gene_names,
        'train_std': train_stds,
        'hdr90_coverage': coverage,
    }).to_csv('blood_kde_coverage_per_gene.csv', index=False)
    pd.DataFrame({
        'sample_idx': np.arange(HOLDOUT),
        'hdr90_coverage': sample_coverage,
    }).to_csv('blood_kde_coverage_per_sample.csv', index=False)
    print('Wrote blood_kde_coverage_per_gene.csv and per_sample.csv')


if __name__ == '__main__':
    main()
