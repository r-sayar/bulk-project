"""
Stress-Gene Removal and Housekeeping-Gene Comparison
=====================================================
1. Compute baseline similarity between GTEx bulk blood and HCA pseudobulk blood.
2. Remove a curated stress-gene panel, recompute similarity → quantify improvement.
3. Compare housekeeping-gene profiles (CV, mean, std, KDE) across both datasets.

Similarity metrics used throughout:
  - Pearson r of per-gene CV (shared genes, log-scale for CV)
  - Pearson r of per-gene mean expression (log2 CPM)
  - Median absolute difference in CV
  - KS-test p-value between CV distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, ks_2samp, pearsonr
import warnings
warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────
BASEDIR  = '/Users/rls/Desktop/programming-projects/single-cell/bulk-project'
PSEUDO   = f'{BASEDIR}/pseudobulk/hca_blood_pseudobulk.npz'
GTEX_PATH = '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz'
OUT_STRESS = f'{BASEDIR}/stress_removal_comparison.png'
OUT_HK     = f'{BASEDIR}/housekeeping_comparison.png'

# ── visual style ────────────────────────────────────────────────────────
BG    = '#0e1117'; CARD  = '#1a1d23'; TEXT  = '#e6edf3'; MUTED = '#7d8590'
GRID  = '#21262d'; C_G   = '#f78166'; C_H   = '#3fb950'; C_P   = '#d2a8ff'
ACCENT1 = '#58a6ff'; ACCENT4 = '#d2a8ff'; ACCENT5 = '#f0883e'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED,  'grid.color': GRID,  'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

# ══════════════════════════════════════════════════════════════════════
# GENE PANELS
# ══════════════════════════════════════════════════════════════════════

# Stress genes: heat shock proteins, co-chaperones, immediate-early
# transcription factors, and key inflammatory cytokines. These genes
# are prone to ex-vivo induction (sample handling stress) or
# inter-donor inflammatory state variation unrelated to steady-state.
STRESS_GENES = {
    # Heat shock proteins (HSP70 family)
    'HSPA1A', 'HSPA1B', 'HSPA1L', 'HSPA2', 'HSPA4', 'HSPA4L',
    'HSPA5', 'HSPA6', 'HSPA7', 'HSPA8', 'HSPA9', 'HSPA12A', 'HSPA12B',
    'HSPA13', 'HSPA14',
    # Small HSPs (HSP27 family)
    'HSPB1', 'HSPB2', 'HSPB3', 'HSPB6', 'HSPB7', 'HSPB8', 'HSPB9', 'HSPB11',
    # HSP90 family
    'HSP90AA1', 'HSP90AA2P', 'HSP90AB1', 'HSP90B1', 'HSP90B2P',
    # Large HSPs / chaperonins
    'HSPH1', 'HSPD1', 'HSPE1', 'HSPC',
    # DnaJ co-chaperones
    'DNAJB1', 'DNAJB2', 'DNAJB4', 'DNAJB5', 'DNAJB6', 'DNAJB9',
    'DNAJB11', 'DNAJB12', 'DNAJC1', 'DNAJC3', 'DNAJC5',
    # Immediate-early response TFs (strongly induced by stress/handling)
    'FOS', 'FOSB', 'FOSL1', 'FOSL2',
    'JUN', 'JUNB', 'JUND',
    'EGR1', 'EGR2', 'EGR3',
    'ATF3', 'ATF4',
    'DUSP1', 'DUSP2', 'DUSP5', 'DUSP6',
    'NR4A1', 'NR4A2', 'NR4A3',
    'IER2', 'IER3', 'IER5',
    'ZFP36', 'ZFP36L1', 'ZFP36L2',
    # Ubiquitin / proteasome stress markers
    'HSPA5', 'UBB', 'UBC',
    # Inflammatory cytokines / chemokines (highly variable inflammatory state)
    'IL1A', 'IL1B', 'IL2', 'IL6', 'IL8', 'CXCL8',
    'TNF', 'TNFA',
    'CCL2', 'CCL3', 'CCL4', 'CCL5', 'CCL7', 'CCL8',
    'CXCL1', 'CXCL2', 'CXCL3', 'CXCL6',
    'LIF', 'MAFF', 'AREG', 'PTGS2', 'PLAUR',
    # Hypoxia / metabolic stress
    'HIF1A', 'VEGFA', 'LDHA',
}

# Housekeeping genes: consistently and highly expressed in virtually all
# human tissues and cell types; used as internal controls in RT-qPCR.
# Selection based on Eisenberg & Levanon (2013) and common practice.
HOUSEKEEPING_GENES = {
    # Cytoskeletal / structural
    'ACTB', 'ACTG1',
    # Ribosomal proteins (large subunit)
    'RPL3', 'RPL4', 'RPL5', 'RPL6', 'RPL7', 'RPL7A', 'RPL8',
    'RPL9', 'RPL10', 'RPL10A', 'RPL11', 'RPL12', 'RPL13', 'RPL13A',
    'RPL14', 'RPL15', 'RPL17', 'RPL18', 'RPL18A', 'RPL19', 'RPL21',
    'RPL22', 'RPL23', 'RPL23A', 'RPL24', 'RPL26', 'RPL27', 'RPL27A',
    'RPL28', 'RPL29', 'RPL30', 'RPL31', 'RPL32', 'RPL34', 'RPL35',
    'RPL35A', 'RPL36', 'RPL37', 'RPL37A', 'RPL38', 'RPL39', 'RPL41',
    # Ribosomal proteins (small subunit)
    'RPS2', 'RPS3', 'RPS3A', 'RPS4X', 'RPS4Y1', 'RPS5', 'RPS6',
    'RPS7', 'RPS8', 'RPS9', 'RPS10', 'RPS11', 'RPS12', 'RPS13',
    'RPS14', 'RPS15', 'RPS15A', 'RPS16', 'RPS17', 'RPS18', 'RPS19',
    'RPS20', 'RPS21', 'RPS23', 'RPS24', 'RPS25', 'RPS26', 'RPS27',
    'RPS27A', 'RPS28', 'RPS29',
    # Translation / elongation factors
    'EEF1A1', 'EEF1B2', 'EEF2', 'EIF3E',
    # Metabolic housekeeping
    'GAPDH', 'PGAM1', 'ENO1', 'TPI1', 'PKM', 'PGK1',
    'LDHA', 'MDH2', 'CS', 'ATP5F1B',
    # Cell biology essentials
    'B2M', 'HPRT1', 'TBP', 'TFRC',
    # ARF / GTPases (from our stable-gene lists)
    'ARF1', 'GNB1',
    # Other commonly used reference genes
    'SDHA', 'YWHAZ', 'IPO8', 'HMBS', 'PPIA',
}

# ══════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

CPM_THRESHOLD    = 1
MIN_SAMPLE_FRAC  = 0.1


def preprocess(expr_raw, gene_names):
    """CPM filter → CPM normalize → log2(CPM+1). Returns (expr_log, expr_cpm, mask)."""
    n_samples = expr_raw.shape[1]
    lib = expr_raw.sum(axis=0)
    cpm = expr_raw / lib * 1e6
    min_s = max(1, int(MIN_SAMPLE_FRAC * n_samples))
    keep = (cpm > CPM_THRESHOLD).sum(axis=1) >= min_s
    e_filt = expr_raw[keep]
    n_filt = gene_names[keep]
    e_cpm  = e_filt / e_filt.sum(axis=0) * 1e6
    e_log  = np.log2(e_cpm + 1)
    return e_log, e_cpm, n_filt


def compute_cv(expr_log):
    """Per-gene CV on log2-scale for expressed genes (mean > 0.5)."""
    means  = expr_log.mean(axis=1)
    stds   = expr_log.std(axis=1)
    expressed = means > 0.5
    cvs = np.full(len(means), np.nan)
    cvs[expressed] = stds[expressed] / means[expressed]
    return means, stds, cvs, expressed


def load_hca(path):
    print("Loading HCA pseudobulk ...")
    d = np.load(path, allow_pickle=True)
    expr_raw   = d['expr'].astype(np.float64)
    gene_names = d['gene_names'].astype(str)
    e_log, e_cpm, names = preprocess(expr_raw, gene_names)
    means, stds, cvs, expressed = compute_cv(e_log)
    print(f"  {len(names):,} genes after filter,  {int(expressed.sum()):,} expressed")
    return {'expr_log': e_log, 'expr_cpm': e_cpm, 'names': names,
            'means': means, 'stds': stds, 'cvs': cvs, 'expressed': expressed}


def load_gtex(path):
    print("Loading GTEx whole blood ...")
    df = pd.read_csv(path, sep='\t', skiprows=2, compression='gzip')
    expr_raw   = df.iloc[:, 2:].values.astype(np.float64)
    gene_names = df['Description'].values.astype(str)
    e_log, e_cpm, names = preprocess(expr_raw, gene_names)
    means, stds, cvs, expressed = compute_cv(e_log)
    print(f"  {len(names):,} genes after filter,  {int(expressed.sum()):,} expressed")
    return {'expr_log': e_log, 'expr_cpm': e_cpm, 'names': names,
            'means': means, 'stds': stds, 'cvs': cvs, 'expressed': expressed}


# ══════════════════════════════════════════════════════════════════════
# SIMILARITY METRICS (on shared expressed genes)
# ══════════════════════════════════════════════════════════════════════

def shared_stats(g, h, exclude_genes=None):
    """
    Return (gcvs, hcvs, gmeans, hmeans, gene_list) for shared expressed genes.
    Optionally exclude a set of gene names.
    """
    exclude = set() if exclude_genes is None else {x.upper() for x in exclude_genes}

    g_idx = {n.upper(): i for i, n in enumerate(g['names'])}
    h_idx = {n.upper(): i for i, n in enumerate(h['names'])}

    gcvs_l, hcvs_l, gm_l, hm_l, gs_l, hs_l, names_l = [], [], [], [], [], [], []

    for name_u, gi in g_idx.items():
        if name_u in exclude:
            continue
        if name_u not in h_idx:
            continue
        hi = h_idx[name_u]
        gcv = g['cvs'][gi];  hcv = h['cvs'][hi]
        if np.isnan(gcv) or np.isnan(hcv):
            continue
        gcvs_l.append(gcv);  hcvs_l.append(hcv)
        gm_l.append(g['means'][gi]); hm_l.append(h['means'][hi])
        gs_l.append(g['stds'][gi]);  hs_l.append(h['stds'][hi])
        names_l.append(name_u)

    return (np.array(gcvs_l), np.array(hcvs_l),
            np.array(gm_l),   np.array(hm_l),
            np.array(gs_l),   np.array(hs_l),
            names_l)


def similarity_report(label, gcvs, hcvs, gmeans, hmeans):
    """Print a row of similarity metrics."""
    r_cv,  _ = pearsonr(np.log1p(gcvs), np.log1p(hcvs))
    r_mean,_ = pearsonr(gmeans, hmeans)
    mad_cv   = np.median(np.abs(gcvs - hcvs))
    ks_stat, ks_p = ks_2samp(gcvs, hcvs)
    med_g = np.median(gcvs); med_h = np.median(hcvs)
    print(f"  {label:<25}  n={len(gcvs):>6,}  "
          f"r_CV={r_cv:+.4f}  r_mean={r_mean:+.4f}  "
          f"MAD_CV={mad_cv:.4f}  "
          f"med_CV: GTEx={med_g:.4f} HCA={med_h:.4f}  KS_p={ks_p:.2e}")
    return {'r_cv': r_cv, 'r_mean': r_mean, 'mad_cv': mad_cv,
            'ks_stat': ks_stat, 'ks_p': ks_p,
            'med_gtex': med_g, 'med_hca': med_h,
            'n': len(gcvs)}


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 – STRESS-GENE REMOVAL
# Panels:
#   1. CV scatter BEFORE removal (shared genes)
#   2. CV scatter AFTER removal
#   3. Bar chart: how each metric changes
#   4. Volcano-style: stress gene positions on CV scatter
#   5. CV distribution overlay before vs after (GTEx)
#   6. CV distribution overlay before vs after (HCA)
# ══════════════════════════════════════════════════════════════════════

def fig_stress(g, h, out):
    # ── collect data ──────────────────────────────────────────────────
    gcvs_b, hcvs_b, gm_b, hm_b, gs_b, hs_b, names_b = shared_stats(g, h)
    gcvs_a, hcvs_a, gm_a, hm_a, gs_a, hs_a, names_a = shared_stats(g, h, STRESS_GENES)

    print("\n" + "="*75)
    print("STRESS-GENE REMOVAL — SIMILARITY COMPARISON")
    print("="*75)
    m_before = similarity_report("All shared genes",    gcvs_b, hcvs_b, gm_b, hm_b)
    m_after  = similarity_report("Stress genes removed", gcvs_a, hcvs_a, gm_a, hm_a)

    # which shared genes are stress genes?
    stress_set_u = {x.upper() for x in STRESS_GENES}
    is_stress = np.array([n in stress_set_u for n in names_b])
    n_stress_found = is_stress.sum()
    print(f"\n  Stress genes in shared expressed set: {n_stress_found}")
    # names of the stress genes actually present
    stress_present = [names_b[i] for i in range(len(names_b)) if is_stress[i]]
    print(f"  Genes: {', '.join(sorted(stress_present))}")

    # ── figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 16))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.3,
                   left=0.07, right=0.96, top=0.91, bottom=0.06)
    fig.suptitle(
        'Effect of Stress-Gene Removal on GTEx Bulk vs HCA Pseudobulk Similarity',
        fontsize=17, fontweight='bold', color=TEXT, y=0.97)

    lim = min(max(gcvs_b.max(), hcvs_b.max()) * 1.05, 3.0)
    diag = [0, lim]

    # ── panel 1: scatter BEFORE ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(gcvs_b[~is_stress], hcvs_b[~is_stress],
               s=2.5, alpha=0.22, c=C_P, rasterized=True, label='Other genes')
    ax.scatter(gcvs_b[is_stress],  hcvs_b[is_stress],
               s=30,  alpha=0.85, c=C_G, marker='*', zorder=5, label='Stress genes')
    ax.plot(diag, diag, '--', color=MUTED, lw=1.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    r_b = m_before['r_cv']
    ax.text(0.05, 0.95, f'r = {r_b:.4f}\nMAD = {m_before["mad_cv"]:.4f}\nn = {m_before["n"]:,}',
            transform=ax.transAxes, fontsize=11, color=TEXT, va='top',
            bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.8, pad=4))
    ax.set_xlabel('GTEx Bulk CV'); ax.set_ylabel('HCA Pseudobulk CV')
    ax.set_title('CV Scatter — All Shared Genes', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT, markerscale=2)
    ax.grid(alpha=0.3)

    # ── panel 2: scatter AFTER ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(gcvs_a, hcvs_a, s=2.5, alpha=0.22, c=C_H, rasterized=True)
    ax.plot(diag, diag, '--', color=MUTED, lw=1.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    r_a = m_after['r_cv']
    ax.text(0.05, 0.95, f'r = {r_a:.4f}\nMAD = {m_after["mad_cv"]:.4f}\nn = {m_after["n"]:,}',
            transform=ax.transAxes, fontsize=11, color=TEXT, va='top',
            bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.8, pad=4))
    ax.set_xlabel('GTEx Bulk CV'); ax.set_ylabel('HCA Pseudobulk CV')
    ax.set_title('CV Scatter — Stress Genes Removed', fontsize=14, fontweight='bold', pad=10)
    ax.grid(alpha=0.3)

    # ── panel 3: metric delta bar chart ──────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    metrics = ['r_CV', 'r_mean', 'MAD_CV↓\n(×10)', 'Δmed_CV↓\n(×10)']
    # positive = improvement; for MAD and med-gap we want them lower → flip sign
    delta_mad   = (m_before['mad_cv']  - m_after['mad_cv'])  * 10  # lower is better
    delta_medgap= (abs(m_before['med_gtex'] - m_before['med_hca']) -
                   abs(m_after['med_gtex']  - m_after['med_hca']))  * 10
    vals_b = [m_before['r_cv'], m_before['r_mean'], 0, 0]
    vals_a = [m_after['r_cv'],  m_after['r_mean'],  0, 0]
    x = np.arange(len(metrics))
    w = 0.32
    bars_b = ax.bar(x - w/2, vals_b, w, color=C_G, alpha=0.8, label='Before removal')
    bars_a = ax.bar(x + w/2, vals_a, w, color=C_H, alpha=0.8, label='After removal')
    # For MAD and med-gap show improvement bars separately
    ax.bar([2 - w/2], [0], w, color=C_G, alpha=0.8)
    ax.bar([2 + w/2], [delta_mad], w, color=C_H, alpha=0.8)
    ax.bar([3 - w/2], [0], w, color=C_G, alpha=0.8)
    ax.bar([3 + w/2], [delta_medgap], w, color=C_H, alpha=0.8)
    # annotate delta on top
    for i, (b, a) in enumerate([(m_before['r_cv'],   m_after['r_cv']),
                                  (m_before['r_mean'], m_after['r_mean'])]):
        delta = a - b
        col = C_H if delta > 0 else C_G
        ax.text(i + w/2, max(a, b) + 0.005,
                f'Δ{delta:+.4f}', ha='center', fontsize=9, color=col, fontweight='bold')
    # annotate delta on MAD / med columns
    ax.text(2 + w/2, delta_mad + 0.002,
            f'+{delta_mad:.3f}', ha='center', fontsize=9, color=C_H, fontweight='bold')
    ax.text(3 + w/2, delta_medgap + 0.002,
            f'+{delta_medgap:.3f}', ha='center', fontsize=9, color=C_H, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel('Metric value (higher = more similar)')
    ax.set_title('Similarity Metric Changes\nafter Stress Removal', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # ── panel 4: stress gene positions labelled ───────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(gcvs_b[~is_stress], hcvs_b[~is_stress],
               s=2, alpha=0.15, c=MUTED, rasterized=True)
    ax.scatter(gcvs_b[is_stress],  hcvs_b[is_stress],
               s=45,  alpha=0.95, c=C_G, marker='D', zorder=6)
    # label top-CV stress genes
    top_stress = sorted(
        [(gcvs_b[i], hcvs_b[i], names_b[i]) for i in range(len(names_b)) if is_stress[i]],
        key=lambda x: x[0], reverse=True)[:12]
    for gv, hv, nm in top_stress:
        ax.annotate(nm, (gv, hv), xytext=(6, 3), textcoords='offset points',
                    fontsize=7.5, color=C_G, fontweight='bold')
    ax.plot(diag, diag, '--', color=MUTED, lw=1.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('GTEx Bulk CV'); ax.set_ylabel('HCA Pseudobulk CV')
    ax.set_title(f'Stress Gene Positions on CV Scatter\n({n_stress_found} genes found in both)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.grid(alpha=0.3)

    # ── panel 5: CV distribution GTEx before/after ────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(gcvs_b, bins=200, range=(0, 2), density=True, alpha=0.5,
            color=C_G, edgecolor='none', label=f'All  (med={np.median(gcvs_b):.3f})')
    ax.hist(gcvs_a, bins=200, range=(0, 2), density=True, alpha=0.5,
            color=ACCENT1, edgecolor='none', label=f'No stress (med={np.median(gcvs_a):.3f})')
    ax.axvline(np.median(gcvs_b), color=C_G,    ls='--', lw=1.5)
    ax.axvline(np.median(gcvs_a), color=ACCENT1, ls='--', lw=1.5)
    ax.set_xlabel('CV'); ax.set_ylabel('Density')
    ax.set_title('GTEx CV Distribution\nBefore vs After Stress Removal', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # ── panel 6: CV distribution HCA before/after ─────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(hcvs_b, bins=200, range=(0, 2), density=True, alpha=0.5,
            color=C_H, edgecolor='none', label=f'All  (med={np.median(hcvs_b):.3f})')
    ax.hist(hcvs_a, bins=200, range=(0, 2), density=True, alpha=0.5,
            color=ACCENT1, edgecolor='none', label=f'No stress (med={np.median(hcvs_a):.3f})')
    ax.axvline(np.median(hcvs_b), color=C_H,    ls='--', lw=1.5)
    ax.axvline(np.median(hcvs_a), color=ACCENT1, ls='--', lw=1.5)
    ax.set_xlabel('CV'); ax.set_ylabel('Density')
    ax.set_title('HCA CV Distribution\nBefore vs After Stress Removal', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"\nSaved: {out}")
    return m_before, m_after, stress_present


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 – HOUSEKEEPING GENE COMPARISON
# Panels (2 rows × 3 cols):
#   1. CV side-by-side bar chart (per housekeeping gene)
#   2. Mean expression side-by-side bar chart
#   3. Std side-by-side bar chart
#   4. CV KDE — GTEx vs HCA (housekeeping genes)
#   5. Mean KDE
#   6. Std KDE
# ══════════════════════════════════════════════════════════════════════

def fig_housekeeping(g, h, out):
    # Find housekeeping genes expressed in both
    hk_upper = {x.upper() for x in HOUSEKEEPING_GENES}
    g_idx = {n.upper(): i for i, n in enumerate(g['names'])}
    h_idx = {n.upper(): i for i, n in enumerate(h['names'])}

    records = []
    for gene_u in sorted(hk_upper):
        if gene_u not in g_idx or gene_u not in h_idx:
            continue
        gi = g_idx[gene_u]; hi = h_idx[gene_u]
        gcv = g['cvs'][gi];  hcv = h['cvs'][hi]
        if np.isnan(gcv) or np.isnan(hcv):
            continue
        records.append({
            'gene':   gene_u,
            'g_cv':   gcv,             'h_cv':   hcv,
            'g_mean': g['means'][gi],  'h_mean': h['means'][hi],
            'g_std':  g['stds'][gi],   'h_std':  h['stds'][hi],
            'g_cpm':  g['expr_cpm'][gi].mean(), 'h_cpm': h['expr_cpm'][hi].mean(),
        })

    df = pd.DataFrame(records).sort_values('g_cv')
    n_hk = len(df)
    print(f"\n{'='*75}")
    print(f"HOUSEKEEPING GENE ANALYSIS  ({n_hk} genes found in both datasets)")
    print(f"{'='*75}")

    # Summary table
    print(f"\n  {'Gene':<12} {'GTEx CV':>8} {'HCA CV':>8} {'GTEx mean':>10} {'HCA mean':>10} "
          f"{'GTEx std':>9} {'HCA std':>9} {'GTEx CPM':>10} {'HCA CPM':>10}")
    print(f"  {'-'*88}")
    for _, r in df.iterrows():
        print(f"  {r.gene:<12} {r.g_cv:>8.4f} {r.h_cv:>8.4f} {r.g_mean:>10.3f} {r.h_mean:>10.3f} "
              f"{r.g_std:>9.4f} {r.h_std:>9.4f} {r.g_cpm:>10.1f} {r.h_cpm:>10.1f}")

    print(f"\n  Summary stats across {n_hk} housekeeping genes:")
    for metric, gc, hc in [('CV',   df.g_cv,   df.h_cv),
                            ('Mean', df.g_mean, df.h_mean),
                            ('Std',  df.g_std,  df.h_std)]:
        print(f"    {metric:5s}  GTEx: mean={gc.mean():.4f}  med={gc.median():.4f}  "
              f"HCA:  mean={hc.mean():.4f}  med={hc.median():.4f}  "
              f"Δmed={hc.median()-gc.median():+.4f}")

    r_cv,  _ = pearsonr(df.g_cv,   df.h_cv)
    r_mean,_ = pearsonr(df.g_mean, df.h_mean)
    r_std, _ = pearsonr(df.g_std,  df.h_std)
    print(f"\n    Pearson r (GTEx vs HCA)  CV={r_cv:.4f}   Mean={r_mean:.4f}   Std={r_std:.4f}")

    # ── figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 18))
    gs2 = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28,
                   left=0.06, right=0.97, top=0.91, bottom=0.05)
    fig.suptitle(
        f'Housekeeping Gene Comparison — GTEx Bulk Blood vs HCA Pseudobulk  ({n_hk} genes)',
        fontsize=17, fontweight='bold', color=TEXT, y=0.97)

    gene_labels = df.gene.tolist()
    x = np.arange(n_hk); w = 0.38

    # ── panel 1: CV bars ──────────────────────────────────────────────
    ax = fig.add_subplot(gs2[0, :])
    ax.bar(x - w/2, df.g_cv, w, color=C_G, alpha=0.8, label='GTEx Bulk')
    ax.bar(x + w/2, df.h_cv, w, color=C_H, alpha=0.8, label='HCA Pseudo')
    ax.set_xticks(x); ax.set_xticklabels(gene_labels, rotation=75, ha='right', fontsize=7.5)
    ax.set_ylabel('CV')
    ax.set_title(f'Per-Gene CV  (Pearson r = {r_cv:.4f})', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # ── panel 2: mean bars ────────────────────────────────────────────
    ax = fig.add_subplot(gs2[1, 0])
    ax.bar(x - w/2, df.g_mean, w, color=C_G, alpha=0.8, label='GTEx')
    ax.bar(x + w/2, df.h_mean, w, color=C_H, alpha=0.8, label='HCA')
    ax.set_xticks(x); ax.set_xticklabels(gene_labels, rotation=75, ha='right', fontsize=7)
    ax.set_ylabel('Mean log₂(CPM+1)')
    ax.set_title(f'Per-Gene Mean Expression  (r = {r_mean:.4f})', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # ── panel 3: std bars ─────────────────────────────────────────────
    ax = fig.add_subplot(gs2[1, 1])
    ax.bar(x - w/2, df.g_std, w, color=C_G, alpha=0.8, label='GTEx')
    ax.bar(x + w/2, df.h_std, w, color=C_H, alpha=0.8, label='HCA')
    ax.set_xticks(x); ax.set_xticklabels(gene_labels, rotation=75, ha='right', fontsize=7)
    ax.set_ylabel('Std (log₂ CPM+1)')
    ax.set_title(f'Per-Gene Std  (r = {r_std:.4f})', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    # ── KDE helper ────────────────────────────────────────────────────
    def kde_panel(ax, g_vals, h_vals, xlabel, title):
        bw = 'scott'
        xmin = min(g_vals.min(), h_vals.min()) * 0.9
        xmax = max(g_vals.max(), h_vals.max()) * 1.1
        xs   = np.linspace(xmin, xmax, 400)
        kg = gaussian_kde(g_vals, bw_method=bw)
        kh = gaussian_kde(h_vals, bw_method=bw)
        ax.fill_between(xs, kg(xs), alpha=0.25, color=C_G)
        ax.plot(xs, kg(xs), color=C_G, lw=2,
                label=f'GTEx  med={np.median(g_vals):.4f}')
        ax.fill_between(xs, kh(xs), alpha=0.25, color=C_H)
        ax.plot(xs, kh(xs), color=C_H, lw=2,
                label=f'HCA   med={np.median(h_vals):.4f}')
        ax.axvline(np.median(g_vals), color=C_G, ls='--', lw=1.5)
        ax.axvline(np.median(h_vals), color=C_H, ls='--', lw=1.5)
        # individual rug marks
        ax.scatter(g_vals, np.zeros_like(g_vals) - kg(g_vals).max()*0.04,
                   s=20, color=C_G, marker='|', alpha=0.6, zorder=5)
        ax.scatter(h_vals, np.zeros_like(h_vals) - kh(h_vals).max()*0.08,
                   s=20, color=C_H, marker='|', alpha=0.6, zorder=5)
        ax.set_xlabel(xlabel); ax.set_ylabel('Density')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
        ax.grid(axis='y', alpha=0.3)

    # ── panels 4–6: KDEs ──────────────────────────────────────────────
    kde_panel(fig.add_subplot(gs2[2, 0]),
              df.g_cv.values,   df.h_cv.values,
              'CV', 'CV Distribution (KDE) — Housekeeping Genes')
    kde_panel(fig.add_subplot(gs2[2, 1]),
              df.g_mean.values, df.h_mean.values,
              'Mean log₂(CPM+1)', 'Mean Expression Distribution (KDE) — Housekeeping Genes')

    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out}")
    return df


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 – HOUSEKEEPING KDE GRID (std + mean + CV per gene)
# One 3-col row per gene: (A) per-sample CV context, (B) per-sample mean,
# (C) expression KDE across samples
# → shows within-gene spread for each dataset side-by-side
# ══════════════════════════════════════════════════════════════════════

def fig_hk_kde_grid(g, h, df_hk, out):
    """
    For each housekeeping gene, show KDE of per-sample log2(CPM+1) values
    (one value per sample/donor) for GTEx and HCA side-by-side.
    Laid out as a grid: rows = genes (sorted by GTEx median expression, top 20).
    """
    top_genes = df_hk.sort_values('g_mean', ascending=False).head(20).gene.tolist()

    g_name_idx = {n.upper(): i for i, n in enumerate(g['names'])}
    h_name_idx = {n.upper(): i for i, n in enumerate(h['names'])}

    NCOLS = 4
    NROWS = (len(top_genes) + NCOLS - 1) // NCOLS
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(6 * NCOLS, 4 * NROWS),
                              facecolor=BG)
    fig.suptitle(
        'Housekeeping Gene Expression KDE — Top 20 by Mean\n'
        'GTEx Bulk Blood (803 samples) vs HCA Pseudobulk (8 donors)',
        fontsize=16, fontweight='bold', color=TEXT, y=0.995)

    for ax in axes.ravel():
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values(): spine.set_edgecolor(GRID)

    for idx, gene_u in enumerate(top_genes):
        row, col = divmod(idx, NCOLS)
        ax = axes[row, col]

        gi = g_name_idx.get(gene_u)
        hi = h_name_idx.get(gene_u)

        g_vals = g['expr_log'][gi, :]   # per-sample expression
        h_vals = h['expr_log'][hi, :]

        xmin = min(g_vals.min(), h_vals.min()) * 0.9
        xmax = max(g_vals.max(), h_vals.max()) * 1.1
        xs   = np.linspace(max(0, xmin), xmax, 400)

        # KDE with tight bandwidth to reveal structure.
        # gaussian_kde bw_method accepts a scalar as bandwidth factor (silverman's rule × scale).
        # We compute it as std/3 / (n^(1/5)) so the effective smoothing is tighter than default.
        n_g = len(g_vals); n_h = len(h_vals)
        bw_g = max(g_vals.std() / 3, 0.05) / (n_g ** 0.2)
        bw_h = max(h_vals.std() / 3, 0.05) / (n_h ** 0.2)
        kg = gaussian_kde(g_vals, bw_method=bw_g)
        kh = gaussian_kde(h_vals, bw_method=bw_h)

        ax.fill_between(xs, kg(xs), alpha=0.3, color=C_G)
        ax.plot(xs, kg(xs), color=C_G, lw=2,
                label=f'GTEx  CV={g["cvs"][gi]:.3f}  μ={g["means"][gi]:.2f}')
        ax.fill_between(xs, kh(xs), alpha=0.3, color=C_H)
        ax.plot(xs, kh(xs), color=C_H, lw=2,
                label=f'HCA   CV={h["cvs"][hi]:.3f}  μ={h["means"][hi]:.2f}')

        ax.axvline(g_vals.mean(), color=C_G, ls='--', lw=1.2)
        ax.axvline(h_vals.mean(), color=C_H, ls='--', lw=1.2)

        ax.set_title(gene_u, fontsize=13, fontweight='bold', color=TEXT, pad=6)
        ax.set_xlabel('log₂(CPM+1)', color=MUTED, fontsize=9)
        ax.set_ylabel('Density', color=MUTED, fontsize=9)
        ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
        ax.grid(axis='y', alpha=0.3)

    # hide unused axes
    for idx in range(len(top_genes), NROWS * NCOLS):
        row, col = divmod(idx, NCOLS)
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Load both datasets
    hca   = load_hca(PSEUDO)
    gtex  = load_gtex(GTEX_PATH)

    # Figure 1: stress gene removal effect on similarity
    m_before, m_after, stress_present = fig_stress(
        gtex, hca, OUT_STRESS)

    # Figure 2: housekeeping gene bar + KDE summary
    df_hk = fig_housekeeping(gtex, hca, OUT_HK)

    # Figure 3: per-gene KDE grid for top housekeeping genes
    fig_hk_kde_grid(gtex, hca, df_hk,
                    f'{BASEDIR}/housekeeping_kde_grid.png')

    print("\nAll done.")
