"""
Housekeeping-gene variation across modalities
=============================================
Step 1: Benchmark normalization factors using HK-gene stability.
        Lower HK CV after normalization ⇒ better normalization.

Step 2: Using the winner, quantify HK variation for:
          (a) bulk ↔ bulk        — GTEx whole blood split in half
          (b) bulk ↔ pseudobulk  — GTEx vs HCA pseudobulk
          (c) bulk ↔ single-cell — GTEx vs Tabula Sapiens 10X blood cells

Reports median/mean CV of HK panel, cross-modality CV-correlation, and
log-ratio scatter of HK-gene means.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, gmean
import anndata as ad
import warnings
warnings.filterwarnings('ignore')

BASEDIR  = '/Users/rls/Desktop/programming-projects/single-cell/bulk-project'
GTEX_PATH = '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz'
PB_PATH   = f'{BASEDIR}/pseudobulk/hca_blood_pseudobulk.npz'
SC_PATH   = f'{BASEDIR}/data/downloaded_sc/tabula_sapiens/blood.h5ad'
OUT_FIG   = f'{BASEDIR}/hk_variation_normfactor.png'

RNG       = np.random.default_rng(0)
SC_N      = 3000            # subsample of single cells
PB_COLS_OK = 8

# ── visual style ──────────────────────────────────────────────────────
BG='#0e1117'; CARD='#1a1d23'; TEXT='#e6edf3'; MUTED='#7d8590'
GRID='#21262d'; C_G='#f78166'; C_H='#3fb950'; C_P='#d2a8ff'; C_B='#58a6ff'
plt.rcParams.update({
    'figure.facecolor': BG,'axes.facecolor': CARD,'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT,'text.color': TEXT,'xtick.color': MUTED,
    'ytick.color': MUTED,'grid.color': GRID,'grid.alpha': 0.5,
    'font.family': 'sans-serif','font.size': 10,
})

# ── housekeeping panel (Eisenberg & Levanon + standards) ──────────────
HK_GENES = [
    'ACTB','GAPDH','B2M','HPRT1','TBP','PPIA','YWHAZ','SDHA','HMBS','TFRC',
    'RPL13A','RPL32','RPS18','RPS27A','EEF1A1','EEF2','PGK1','TPI1','ENO1','PKM',
    'PGAM1','LDHA','MDH2','CS','ATP5F1B','RPL4','RPL7','RPL11','RPL27','RPL30',
    'RPS3','RPS8','RPS14','RPS19','ARF1','GNB1','UBC','UBB',
]

# ══════════════════════════════════════════════════════════════════════
# 1. DATA LOADERS  →  raw count matrix (genes × samples)
# ══════════════════════════════════════════════════════════════════════

def load_gtex():
    print('Loading GTEx whole blood ...')
    df = pd.read_csv(GTEX_PATH, sep='\t', skiprows=2, compression='gzip')
    counts = df.iloc[:, 2:].values.astype(np.float64)       # genes × samples
    names  = df['Description'].astype(str).values
    # collapse duplicate gene symbols by summing
    u, inv = np.unique(names, return_inverse=True)
    out = np.zeros((len(u), counts.shape[1]))
    np.add.at(out, inv, counts)
    print(f'  GTEx: {out.shape[0]:,} genes × {out.shape[1]} samples')
    return out, u

def load_pb():
    print('Loading HCA pseudobulk ...')
    d = np.load(PB_PATH, allow_pickle=True)
    counts = d['expr'].astype(np.float64)                   # genes × donors
    names  = d['gene_names'].astype(str)
    u, inv = np.unique(names, return_inverse=True)
    out = np.zeros((len(u), counts.shape[1]))
    np.add.at(out, inv, counts)
    print(f'  PB: {out.shape[0]:,} genes × {out.shape[1]} donors')
    return out, u

def load_sc(n_cells=SC_N):
    print(f'Loading Tabula Sapiens blood (sampling {n_cells} 10X cells)...')
    a = ad.read_h5ad(SC_PATH, backed='r')
    mask_10x = (a.obs['method'] == '10X').values
    idx = np.where(mask_10x)[0]
    sample_idx = RNG.choice(idx, size=min(n_cells, len(idx)), replace=False)
    sample_idx = np.sort(sample_idx)
    X = a.raw.X[sample_idx, :].toarray().T.astype(np.float64)  # genes × cells
    names = a.raw.var['feature_name'].astype(str).values
    u, inv = np.unique(names, return_inverse=True)
    out = np.zeros((len(u), X.shape[1]))
    np.add.at(out, inv, X)
    print(f'  SC: {out.shape[0]:,} genes × {out.shape[1]} cells')
    return out, u

# ══════════════════════════════════════════════════════════════════════
# 2. NORMALIZATION FACTORS
# Each returns a log2(normalized+1) genes×samples matrix.
# ══════════════════════════════════════════════════════════════════════

def n_cpm(M):
    lib = M.sum(axis=0) + 1e-9
    return np.log2(M / lib * 1e6 + 1)

def n_median_ratio(M):
    # DESeq2-style: size factor = median_gene( x_gi / geomean_i(x_g.) )
    pseudo = M + 1
    log_geo = np.log(pseudo).mean(axis=1)            # per-gene log-geo mean
    ratios  = np.log(pseudo) - log_geo[:, None]      # per (gene,sample)
    # use only genes finite and expressed in all samples
    mask = np.all(M > 0, axis=1)
    sf = np.exp(np.median(ratios[mask], axis=0))
    sf = sf / gmean(sf)                              # center
    scaled = M / sf[None, :]
    return np.log2(scaled + 1)

def n_upper_quartile(M):
    # UQ: scale so 75th percentile of non-zero counts is constant
    sf = np.array([np.percentile(col[col > 0], 75) if (col > 0).any() else 1.
                    for col in M.T])
    sf = sf / gmean(sf)
    return np.log2(M / sf[None, :] + 1)

def n_hk_geomean(M, hk_idx):
    # scale each sample so HK-gene geomean = constant
    sub = M[hk_idx, :] + 1
    sf  = np.exp(np.log(sub).mean(axis=0))
    sf  = sf / gmean(sf)
    return np.log2(M / sf[None, :] + 1)

def n_quantile(M):
    # classic quantile normalization per-sample
    ranks = np.argsort(np.argsort(M, axis=0), axis=0)
    sort  = np.sort(M, axis=0)
    ref   = sort.mean(axis=1)
    out   = ref[ranks]
    return np.log2(out + 1)

NORMS = {
    'CPM':           n_cpm,
    'median-ratio':  n_median_ratio,
    'upper-quart.':  n_upper_quartile,
    'HK-geomean':    n_hk_geomean,
    'quantile':      n_quantile,
}

# ══════════════════════════════════════════════════════════════════════
# 3. HK PANEL INTERSECT + CV
# ══════════════════════════════════════════════════════════════════════

def hk_indices(names):
    idx_map = {n.upper(): i for i, n in enumerate(names)}
    return np.array([idx_map[g] for g in HK_GENES if g in idx_map])

def cv_per_gene(log_mat):
    mu  = log_mat.mean(axis=1)
    sd  = log_mat.std(axis=1)
    cv  = np.where(mu > 0.1, sd / mu, np.nan)
    return mu, sd, cv

# ══════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    # load
    gM, gN = load_gtex()
    pM, pN = load_pb()
    sM, sN = load_sc()

    # HK indices per dataset
    g_hk = hk_indices(gN); p_hk = hk_indices(pN); s_hk = hk_indices(sN)
    print(f'HK genes matched: GTEx {len(g_hk)}  PB {len(p_hk)}  SC {len(s_hk)}')

    # ─── Step 1: normalization benchmark (HK median CV) ─────────────
    print('\n' + '='*78)
    print('STEP 1 — NORMALIZATION BENCHMARK (HK median CV; lower = more stable)')
    print('='*78)
    bench = {}
    for label, fn in NORMS.items():
        row = {}
        for ds, M, hk in [('GTEx', gM, g_hk), ('PB', pM, p_hk), ('SC', sM, s_hk)]:
            if label == 'HK-geomean':
                Ln = fn(M, hk)
            else:
                Ln = fn(M)
            _, _, cv = cv_per_gene(Ln[hk, :])
            row[ds] = np.nanmedian(cv)
        row['mean'] = np.nanmean(list(row.values()))
        bench[label] = row
        vals = row
        sc_str = f'{vals["SC"]:.4f}' if not np.isnan(vals["SC"]) else '   NaN'
        mean_str = f'{vals["mean"]:.4f}' if not np.isnan(vals["mean"]) else '   NaN'
        print(f'  {label:<14}  GTEx={vals["GTEx"]:.4f}  PB={vals["PB"]:.4f}  '
              f'SC={sc_str}  mean={mean_str}')
    # Winner: only consider methods that gave a finite value on all 3 modalities
    feasible = {k: v for k, v in bench.items()
                if all(np.isfinite([v['GTEx'], v['PB'], v['SC']]))}
    best = min(feasible, key=lambda k: feasible[k]['mean'])
    print(f'\n  → best normalization: {best}')

    # ─── Step 2: apply winner and compute modality variation ────────
    print('\n' + '='*78)
    print(f'STEP 2 — VARIATION VIA "{best}"')
    print('='*78)

    def apply(fn_name, M, hk):
        fn = NORMS[fn_name]
        return fn(M, hk) if fn_name == 'HK-geomean' else fn(M)

    gL = apply(best, gM, g_hk)
    pL = apply(best, pM, p_hk)
    sL = apply(best, sM, s_hk)

    # Bulk-to-bulk: random split of GTEx into two halves
    n_g = gL.shape[1]
    perm = RNG.permutation(n_g)
    A, B = perm[:n_g // 2], perm[n_g // 2:]
    gA, gB = gL[:, A], gL[:, B]

    # sub-dicts
    def stats(hk_mat, label):
        mu = hk_mat.mean(axis=1); sd = hk_mat.std(axis=1)
        cv = np.where(mu > 0.1, sd/mu, np.nan)
        return {'label': label, 'mu': mu, 'sd': sd, 'cv': cv}

    S = {
        'BulkA':  stats(gA[g_hk, :], 'GTEx bulk (half A)'),
        'BulkB':  stats(gB[g_hk, :], 'GTEx bulk (half B)'),
        'Bulk':   stats(gL[g_hk, :], 'GTEx bulk (all)'),
        'PB':     stats(pL[p_hk, :], 'HCA pseudobulk'),
        'SC':     stats(sL[s_hk, :], 'Tabula Sap. single-cell'),
    }

    # shared HK-gene list (labels aligned)
    gN_hk = [gN[i].upper() for i in g_hk]
    pN_hk = [pN[i].upper() for i in p_hk]
    sN_hk = [sN[i].upper() for i in s_hk]

    # Align by gene symbol — use the smallest common set
    common = sorted(set(gN_hk) & set(pN_hk) & set(sN_hk))
    print(f'\nHK genes present in all 3 modalities: {len(common)}')
    g_map = {n: i for i, n in enumerate(gN_hk)}
    p_map = {n: i for i, n in enumerate(pN_hk)}
    s_map = {n: i for i, n in enumerate(sN_hk)}
    g_pos = [g_map[n] for n in common]
    p_pos = [p_map[n] for n in common]
    s_pos = [s_map[n] for n in common]

    def pair_report(label, xcv, ycv, xmu, ymu):
        n0 = len(xcv)
        m = ~(np.isnan(xcv) | np.isnan(ycv) | np.isnan(xmu) | np.isnan(ymu))
        n = int(m.sum())
        if n < 2:
            print(f'  {label:<28}  <2 valid pairs ({n})')
            return {'r_cv': np.nan, 'r_mu': np.nan, 'mad_cv': np.nan,
                    'mad_mu': np.nan, 'medx': np.nan, 'medy': np.nan, 'n': n}
        r_cv, _ = pearsonr(xcv[m], ycv[m])
        r_mu, _ = pearsonr(xmu[m], ymu[m])
        mad_cv  = np.median(np.abs(xcv[m] - ycv[m]))
        mad_mu  = np.median(np.abs(xmu[m] - ymu[m]))
        medx    = np.nanmedian(xcv); medy = np.nanmedian(ycv)
        print(f'  {label:<28}  r_CV={r_cv:+.4f}  r_μ={r_mu:+.4f}  '
              f'MAD_CV={mad_cv:.4f}  MAD_μ={mad_mu:.4f}  '
              f'medCV: X={medx:.4f} Y={medy:.4f}  n={n}')
        return {'r_cv': r_cv, 'r_mu': r_mu, 'mad_cv': mad_cv,
                'mad_mu': mad_mu, 'medx': medx, 'medy': medy, 'n': n}

    def col(S_, pos): return S_['cv'][pos], S_['mu'][pos]

    print('\n  PAIRWISE COMPARISONS (HK genes shared across all 3):')
    bulkA_cv = S['BulkA']['cv'][g_pos]; bulkA_mu = S['BulkA']['mu'][g_pos]
    bulkB_cv = S['BulkB']['cv'][g_pos]; bulkB_mu = S['BulkB']['mu'][g_pos]
    bulk_cv  = S['Bulk']['cv'][g_pos];  bulk_mu  = S['Bulk']['mu'][g_pos]
    pb_cv    = S['PB']['cv'][p_pos];    pb_mu    = S['PB']['mu'][p_pos]
    sc_cv    = S['SC']['cv'][s_pos];    sc_mu    = S['SC']['mu'][s_pos]

    m_bb = pair_report('bulk  ↔ bulk   (split)', bulkA_cv, bulkB_cv, bulkA_mu, bulkB_mu)
    m_bp = pair_report('bulk  ↔ pseudobulk',     bulk_cv,  pb_cv,    bulk_mu,  pb_mu)
    m_bs = pair_report('bulk  ↔ single-cell',    bulk_cv,  sc_cv,    bulk_mu,  sc_mu)

    # per-gene table
    table = pd.DataFrame({
        'gene':    common,
        'bulkA_CV':bulkA_cv, 'bulkB_CV':bulkB_cv,
        'bulk_CV': bulk_cv,  'PB_CV':   pb_cv,    'SC_CV': sc_cv,
        'bulk_μ':  bulk_mu,  'PB_μ':    pb_mu,    'SC_μ':  sc_mu,
    }).sort_values('bulk_CV')
    print('\n  Per-gene HK variation (sorted by bulk CV):')
    print(table.to_string(float_format=lambda v: f'{v:7.4f}', index=False))

    summary = pd.DataFrame({
        'Dataset':   ['bulk-A','bulk-B','bulk-all','pseudobulk','single-cell'],
        'med_CV':    [np.nanmedian(bulkA_cv), np.nanmedian(bulkB_cv),
                      np.nanmedian(bulk_cv),  np.nanmedian(pb_cv),
                      np.nanmedian(sc_cv)],
        'mean_CV':   [np.nanmean(bulkA_cv), np.nanmean(bulkB_cv),
                      np.nanmean(bulk_cv),  np.nanmean(pb_cv),
                      np.nanmean(sc_cv)],
    })
    print('\n  Summary HK CV:')
    print(summary.to_string(float_format=lambda v: f'{v:.4f}', index=False))

    # ─── Figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32,
                   left=0.06, right=0.97, top=0.92, bottom=0.06)
    fig.suptitle(
        f'Housekeeping-Gene Variation Across Modalities   (norm = {best})',
        fontsize=16, fontweight='bold', color=TEXT, y=0.97)

    # panel A — normalization benchmark bars
    ax = fig.add_subplot(gs[0, :])
    labels = list(bench.keys())
    gtex_v = [bench[l]['GTEx'] for l in labels]
    pb_v   = [bench[l]['PB']   for l in labels]
    sc_v   = [bench[l]['SC']   for l in labels]
    x = np.arange(len(labels)); w = 0.26
    ax.bar(x - w,     gtex_v, w, color=C_G, alpha=0.85, label='GTEx bulk')
    ax.bar(x,         pb_v,   w, color=C_H, alpha=0.85, label='HCA pseudobulk')
    ax.bar(x + w,     sc_v,   w, color=C_P, alpha=0.85, label='Tab. Sap. SC')
    for xi, l in enumerate(labels):
        ax.text(xi, max(gtex_v[xi], pb_v[xi], sc_v[xi]) + 0.01,
                f'mean={bench[l]["mean"]:.3f}', ha='center', fontsize=9, color=TEXT)
    best_i = labels.index(best)
    ax.axvspan(best_i - 0.4, best_i + 0.4, color=C_B, alpha=0.12)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Median HK-gene CV'); ax.set_title('Normalization benchmark (lower = more stable HK panel)', fontweight='bold', pad=10)
    ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # panel B — bulk ↔ bulk CV scatter
    def scatter(ax, x, y, xl, yl, title, color, m):
        lim = max(np.nanmax(x), np.nanmax(y)) * 1.15
        ax.scatter(x, y, s=50, c=color, alpha=0.8, edgecolor=TEXT, linewidth=0.3)
        ax.plot([0, lim], [0, lim], '--', color=MUTED, lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title, fontweight='bold', fontsize=12, pad=8)
        ax.text(0.03, 0.97, f'r = {m["r_cv"]:+.3f}\nMAD = {m["mad_cv"]:.3f}\nn = {m["n"]}',
                transform=ax.transAxes, va='top', fontsize=9, color=TEXT,
                bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.85, pad=4))
        ax.grid(alpha=0.3)

    scatter(fig.add_subplot(gs[1, 0]), bulkA_cv, bulkB_cv,
            'Bulk A (GTEx ½) CV', 'Bulk B (GTEx ½) CV',
            'bulk ↔ bulk', C_G, m_bb)
    scatter(fig.add_subplot(gs[1, 1]), bulk_cv,  pb_cv,
            'GTEx bulk CV', 'HCA pseudobulk CV',
            'bulk ↔ pseudobulk', C_H, m_bp)
    scatter(fig.add_subplot(gs[1, 2]), bulk_cv,  sc_cv,
            'GTEx bulk CV', 'Tab. Sap. single-cell CV',
            'bulk ↔ single-cell', C_P, m_bs)

    # panel C — summary of CV magnitudes per dataset
    ax = fig.add_subplot(gs[2, 0])
    datasets = ['bulk-A','bulk-B','bulk-all','pseudo','SC']
    cvs = [bulkA_cv, bulkB_cv, bulk_cv, pb_cv, sc_cv]
    colors = [C_G, C_G, C_G, C_H, C_P]
    bp = ax.boxplot(cvs, labels=datasets, patch_artist=True, widths=0.55,
                     medianprops=dict(color=TEXT, lw=1.6))
    for box, c in zip(bp['boxes'], colors):
        box.set(facecolor=c, alpha=0.5, edgecolor=c)
    for w2 in bp['whiskers'] + bp['caps']:
        w2.set(color=MUTED)
    ax.set_ylabel('HK-gene CV per dataset')
    ax.set_title('HK-gene CV — absolute magnitudes', fontweight='bold', fontsize=12, pad=8)
    ax.grid(axis='y', alpha=0.3)

    # panel D — per-gene CV bars, three-way
    ax = fig.add_subplot(gs[2, 1:])
    x = np.arange(len(common)); w = 0.22
    ax.bar(x - 1.5*w, bulk_cv, w, color=C_G, alpha=0.85, label='GTEx bulk')
    ax.bar(x - 0.5*w, pb_cv,   w, color=C_H, alpha=0.85, label='HCA pseudobulk')
    ax.bar(x + 0.5*w, sc_cv,   w, color=C_P, alpha=0.85, label='Tab. Sap. SC')
    ax.bar(x + 1.5*w, bulkA_cv, w, color=C_B, alpha=0.6, label='GTEx ½ (bulk-A)')
    ax.set_xticks(x); ax.set_xticklabels(common, rotation=70, ha='right', fontsize=8)
    ax.set_ylabel('CV')
    ax.set_title('Per-gene HK CV across modalities', fontweight='bold', fontsize=12, pad=8)
    ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(OUT_FIG, dpi=170, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'\nSaved: {OUT_FIG}')

if __name__ == '__main__':
    main()
