"""
GTEx liver–blood interaction via the surfaceome and secretome
==============================================================
The hypothesis:
  - Liver sinusoidal cells sit in direct contact with blood. Evidence should be
    visible as (a) massive expression of the SECRETOME (plasma proteins dumped
    into blood — ALB, fibrinogen, clotting factors, apolipoproteins) and
    (b) the SURFACEOME receptors/transporters that pull things out of blood
    (ASGR1/2, SLC/SLCO transporters, LRP1, Kupffer-cell scavenger receptors).
  - Whole blood (which is the other side of the interface) should be dominated
    by blood-cell machinery (hemoglobin, HLA, immune receptors) and should NOT
    itself manufacture plasma proteins — blood cells carry them, they don't
    make them.

Inputs:
  data/annotations/surfaceome_bausch_fluck_2018.xlsx  (Bausch-Fluck 2018 SURFY)
  data/annotations/hpa_secretome.tsv                  (HPA predicted secreted)
  /Users/rls/Downloads/gene_reads_v11_liver.gct.gz    (GTEx v11 bulk liver)
  /Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz (GTEx v11 bulk blood)
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

PROJECT = Path('/Users/rls/Desktop/programming-projects/single-cell/bulk-project')
TISSUES = {
    'Liver':       Path('/Users/rls/Downloads/gene_reads_v11_liver.gct.gz'),
    'Whole Blood': Path('/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz'),
}
SURFACEOME_XLSX = PROJECT / 'data/annotations/surfaceome_bausch_fluck_2018.xlsx'
SECRETOME_TSV   = PROJECT / 'data/annotations/hpa_secretome.tsv'
OUT_PNG         = PROJECT / 'gtex_liver_blood_interaction.png'

MIN_CPM     = 1.0     # gene detected if ≥1 CPM
MIN_FRAC    = 0.10    # ... in at least 10% of samples
TOP_N_RANK  = 20      # top-N shown in bar panels
TOP_N_LABEL = 18      # max labels in scatter

# Dark theme (matches existing gtex_* figures)
BG, CARD, TEXT, MUTED, GRID = '#0e1117', '#1a1d23', '#e6edf3', '#7d8590', '#21262d'
C_LIVER  = '#58a6ff'
C_BLOOD  = '#f78166'
C_SECR   = '#3fb950'   # plasma-protein secretome
C_SURF   = '#d2a8ff'   # surfaceome
C_BOTH   = '#ffa657'   # surface + secreted (rare but real: shed receptors etc.)
C_OTHER  = '#30363d'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.4,
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.titleweight': 'bold',
})


# ══════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════

def load_gct(path: Path) -> pd.DataFrame:
    """GCT → DataFrame indexed by gene SYMBOL (dedup by summing), columns = samples."""
    df = pd.read_csv(path, sep='\t', skiprows=2, compression='gzip')
    df = df.drop(columns=['Name'])                           # keep Description = symbol
    df = df.rename(columns={'Description': 'gene'})
    df = df.groupby('gene', as_index=True).sum(numeric_only=True)
    return df


def cpm_log(counts: pd.DataFrame) -> pd.DataFrame:
    lib = counts.sum(axis=0)
    cpm = counts.div(lib, axis=1) * 1e6
    return np.log10(cpm + 1.0)


def detected(counts: pd.DataFrame) -> pd.Series:
    lib = counts.sum(axis=0)
    cpm = counts.div(lib, axis=1) * 1e6
    return (cpm >= MIN_CPM).mean(axis=1) >= MIN_FRAC


def load_surfaceome() -> set:
    df = pd.read_excel(SURFACEOME_XLSX, sheet_name='in silico surfaceome only', header=1)
    return set(df['UniProt gene'].dropna().astype(str))


def load_secretome() -> pd.DataFrame:
    """Returns gene → {'Secreted to blood', 'Secreted to other', ...}."""
    df = pd.read_csv(SECRETOME_TSV, sep='\t')
    df = df[['Gene', 'Secretome location']].copy()
    df['Secretome location'] = df['Secretome location'].fillna('Secreted (unclassified)')
    df = df.drop_duplicates('Gene').set_index('Gene')
    return df


# ══════════════════════════════════════════════════════════════════════
# ANNOTATION
# ══════════════════════════════════════════════════════════════════════

def annotate(genes: pd.Index, surf: set, secr: pd.DataFrame) -> pd.DataFrame:
    """One row per gene with boolean flags + a combined category label."""
    a = pd.DataFrame(index=genes)
    a['surfaceome']       = a.index.isin(surf)
    a['secretome']        = a.index.isin(secr.index)
    a['secreted_to_blood'] = a.index.map(
        lambda g: 'Secreted to blood' in str(secr.loc[g, 'Secretome location'])
        if g in secr.index else False
    )
    def cat(row):
        if row['surfaceome'] and row['secretome']:     return 'both'
        if row['secreted_to_blood']:                    return 'plasma'      # secreted → blood
        if row['secretome']:                            return 'secreted_other'
        if row['surfaceome']:                           return 'surfaceome'
        return 'other'
    a['category'] = a.apply(cat, axis=1)
    return a


# ══════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════

CAT_COLOR = {
    'plasma':         C_SECR,
    'surfaceome':     C_SURF,
    'both':           C_BOTH,
    'secreted_other': '#56d364',
    'other':          C_OTHER,
}
CAT_LABEL = {
    'plasma':         'Secreted → blood',
    'surfaceome':     'Surfaceome',
    'both':           'Surface + secreted',
    'secreted_other': 'Secreted (other)',
    'other':          'Other',
}

# Curated "story" genes — labeled in the scatter to make the biology readable
LIVER_STORY = {
    'plasma': ['ALB','SERPINA1','FGA','FGB','FGG','F2','APOB','APOA1','APOA2',
               'APOC3','APOH','TF','HP','APCS','AHSG','TTR','C3','SERPINC1',
               'F9','F10','F7','KNG1','PROC','PROS1','RBP4','AFP'],
    'surfaceome': ['ASGR1','ASGR2','SLC10A1','SLCO1B1','SLCO1B3','SLC2A2',
                   'LRP1','TFR2','LIFR','CD163','VSIG4','STAB1','STAB2',
                   'CLEC4F','MARCO','CLEC4M','FCGRT','ABCC2','ABCB11'],
}
BLOOD_STORY = {
    'surfaceome': ['CD3D','CD4','CD8A','CD19','CD14','CD79A','FCGR3A','FCGR3B',
                   'ITGAM','HLA-A','HLA-DRA','CXCR4','IL7R','SELL'],
    'other':      ['HBB','HBA1','HBA2','HBD'],
}


def main():
    print('Loading GTEx GCTs …')
    counts = {t: load_gct(p) for t, p in TISSUES.items()}
    for t, c in counts.items():
        print(f'  {t}: {c.shape[0]} genes × {c.shape[1]} samples')

    # Keep genes detected in BOTH tissues (so comparison is fair)
    common = counts['Liver'].index.intersection(counts['Whole Blood'].index)
    det = detected(counts['Liver'].loc[common]) | detected(counts['Whole Blood'].loc[common])
    keep = common[det]
    counts = {t: c.loc[keep] for t, c in counts.items()}
    print(f'  kept {len(keep)} detected genes')

    logcpm = {t: cpm_log(c) for t, c in counts.items()}
    median = pd.DataFrame({t: lc.median(axis=1) for t, lc in logcpm.items()})

    # Per-tissue share of total reads (before log) — for the "what fraction of
    # the liver transcriptome IS plasma proteins" panel.
    share = pd.DataFrame({
        t: counts[t].sum(axis=1) / counts[t].sum().sum() for t in TISSUES
    })

    print('Loading surfaceome + secretome annotations …')
    surf = load_surfaceome()
    secr = load_secretome()
    ann = annotate(median.index, surf, secr)
    print(f'  surfaceome genes in data: {ann["surfaceome"].sum()}')
    print(f'  secretome genes in data:  {ann["secretome"].sum()}')
    print(f'  secreted-to-blood genes:  {ann["secreted_to_blood"].sum()}')

    df = median.join(ann).join(share.rename(columns=lambda t: f'share_{t}'))
    df['liver_minus_blood'] = df['Liver'] - df['Whole Blood']

    # ───────────────────── figure ─────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                  height_ratios=[1.2, 1.0, 1.0])

    # Panel A: the main scatter — liver vs blood median log-CPM, colored by category
    axA = fig.add_subplot(gs[0, :2])
    order = ['other', 'secreted_other', 'surfaceome', 'both', 'plasma']  # back → front
    for cat in order:
        sub = df[df['category'] == cat]
        axA.scatter(sub['Whole Blood'], sub['Liver'],
                    s=6 if cat == 'other' else 12,
                    c=CAT_COLOR[cat], alpha=0.35 if cat == 'other' else 0.75,
                    edgecolors='none', label=f'{CAT_LABEL[cat]} (n={len(sub)})',
                    zorder=order.index(cat))
    lim = (df[['Liver', 'Whole Blood']].min().min() - 0.2,
           df[['Liver', 'Whole Blood']].max().max() + 0.2)
    axA.plot(lim, lim, ls='--', color=MUTED, lw=0.8, alpha=0.6, zorder=0)
    axA.set_xlim(lim); axA.set_ylim(lim)
    axA.set_xlabel('log₁₀(CPM+1) — Whole Blood median')
    axA.set_ylabel('log₁₀(CPM+1) — Liver median')
    axA.set_title('A. Where does each tissue live on the surfaceome / secretome map?')
    axA.legend(loc='lower right', frameon=False, fontsize=8)

    # Label the story genes
    labeled = set()
    for gene in LIVER_STORY['plasma'] + LIVER_STORY['surfaceome']:
        if gene in df.index and gene not in labeled:
            row = df.loc[gene]
            axA.annotate(gene, (row['Whole Blood'], row['Liver']),
                         xytext=(4, 4), textcoords='offset points',
                         color=TEXT, fontsize=7.5, fontweight='bold', alpha=0.9)
            labeled.add(gene)
    for gene in BLOOD_STORY['surfaceome'] + BLOOD_STORY['other']:
        if gene in df.index and gene not in labeled:
            row = df.loc[gene]
            axA.annotate(gene, (row['Whole Blood'], row['Liver']),
                         xytext=(4, -8), textcoords='offset points',
                         color=MUTED, fontsize=7, alpha=0.8)
            labeled.add(gene)

    # Panel B: enrichment in top-K — what fraction of the top-K highest-
    # expressed genes in each tissue belong to each category?
    axB = fig.add_subplot(gs[0, 2])
    ks = [50, 100, 250, 500, 1000, 2500]
    rows = []
    for tissue in ['Liver', 'Whole Blood']:
        ranked = df.sort_values(tissue, ascending=False)
        for k in ks:
            top = ranked.head(k)
            rows.append({
                'tissue': tissue, 'K': k,
                'plasma': (top['category'] == 'plasma').mean(),
                'surfaceome': (top['category'] == 'surfaceome').mean(),
                'both': (top['category'] == 'both').mean(),
            })
    enr = pd.DataFrame(rows)
    x = np.arange(len(ks))
    w = 0.38
    for i, tis in enumerate(['Liver', 'Whole Blood']):
        sub = enr[enr['tissue'] == tis].sort_values('K')
        plasma = sub['plasma'].values
        surface = sub['surfaceome'].values + sub['both'].values
        axB.bar(x + (i - 0.5) * w, plasma, w,
                color=C_SECR, alpha=0.9, edgecolor='none',
                label=f'{tis}: plasma' if i == 0 else None)
        axB.bar(x + (i - 0.5) * w, surface, w, bottom=plasma,
                color=C_SURF, alpha=0.9, edgecolor='none',
                label=f'{tis}: surface' if i == 0 else None)
        # tissue label at the top of each stack
        for xi, p, s in zip(x + (i - 0.5) * w, plasma, surface):
            axB.text(xi, p + s + 0.01, 'L' if tis == 'Liver' else 'B',
                     ha='center', va='bottom', fontsize=7,
                     color=C_LIVER if tis == 'Liver' else C_BLOOD,
                     fontweight='bold')
    axB.set_xticks(x); axB.set_xticklabels([str(k) for k in ks])
    axB.set_xlabel('Top-K genes by expression')
    axB.set_ylabel('Fraction in category')
    axB.set_title('B. Top-K enrichment: plasma (green) + surface (purple)')
    axB.legend(loc='upper right', frameon=False, fontsize=8)

    # Panel C: share of total reads going to each category (the big reveal —
    # liver devotes enormous transcriptional budget to plasma proteins)
    axC = fig.add_subplot(gs[1, 0])
    cats_show = ['plasma', 'both', 'surfaceome', 'secreted_other', 'other']
    shares = {
        t: [df[df['category'] == c][f'share_{t}'].sum() for c in cats_show]
        for t in TISSUES
    }
    bottom_l = np.zeros(1); bottom_b = np.zeros(1)
    for c in cats_show:
        sl = df[df['category'] == c]['share_Liver'].sum()
        sb = df[df['category'] == c]['share_Whole Blood'].sum()
        axC.bar(['Liver'], [sl], bottom=bottom_l,
                color=CAT_COLOR[c], label=CAT_LABEL[c], edgecolor='none')
        axC.bar(['Whole Blood'], [sb], bottom=bottom_b,
                color=CAT_COLOR[c], edgecolor='none')
        # percentage labels for dominant categories
        if sl > 0.03:
            axC.text(0, bottom_l[0] + sl / 2, f'{sl*100:.0f}%',
                     ha='center', va='center', fontsize=9, color=BG, fontweight='bold')
        if sb > 0.03:
            axC.text(1, bottom_b[0] + sb / 2, f'{sb*100:.0f}%',
                     ha='center', va='center', fontsize=9, color=BG, fontweight='bold')
        bottom_l = bottom_l + sl; bottom_b = bottom_b + sb
    axC.set_ylim(0, 1.02)
    axC.set_ylabel('Share of total reads')
    axC.set_title('C. What fraction of the transcriptome\nis each category?')
    axC.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
               frameon=False, fontsize=8)

    # Panel D: top-20 "Secreted to blood" genes in liver — the plasma proteins
    axD = fig.add_subplot(gs[1, 1])
    plasma_df = df[df['category'] == 'plasma'].sort_values('Liver', ascending=True).tail(TOP_N_RANK)
    y = np.arange(len(plasma_df))
    axD.barh(y, plasma_df['Liver'], color=C_LIVER, alpha=0.85, label='Liver')
    axD.barh(y, plasma_df['Whole Blood'], color=C_BLOOD, alpha=0.75, label='Whole Blood')
    axD.set_yticks(y); axD.set_yticklabels(plasma_df.index, fontsize=8)
    axD.set_xlabel('log₁₀(CPM+1)')
    axD.set_title('D. Top plasma-protein genes in liver\n(HPA "Secreted to blood")')
    axD.legend(loc='lower right', frameon=False, fontsize=8)

    # Panel E: top-20 surfaceome genes ENRICHED in liver (liver - blood)
    axE = fig.add_subplot(gs[1, 2])
    surf_df = df[df['category'].isin(['surfaceome', 'both'])]
    surf_df = surf_df[surf_df['Liver'] > np.log10(10 + 1)]   # reasonably expressed
    surf_df = surf_df.sort_values('liver_minus_blood', ascending=True).tail(TOP_N_RANK)
    y = np.arange(len(surf_df))
    axE.barh(y, surf_df['liver_minus_blood'], color=C_SURF, alpha=0.9)
    axE.axvline(0, color=MUTED, lw=0.6)
    axE.set_yticks(y); axE.set_yticklabels(surf_df.index, fontsize=8)
    axE.set_xlabel('log₁₀ FC  (liver − blood)')
    axE.set_title('E. Liver-enriched surfaceome\n(uptake / scavenger receptors)')

    # Panel F: mirror — top surfaceome genes ENRICHED in blood (the other side
    # of the interface, i.e. lymphocyte/granulocyte surface receptors)
    axF = fig.add_subplot(gs[2, 0])
    surf_df_b = df[df['category'].isin(['surfaceome', 'both'])]
    surf_df_b = surf_df_b[surf_df_b['Whole Blood'] > np.log10(10 + 1)]
    surf_df_b = surf_df_b.sort_values('liver_minus_blood', ascending=False).tail(TOP_N_RANK)
    y = np.arange(len(surf_df_b))
    axF.barh(y, -surf_df_b['liver_minus_blood'], color=C_BLOOD, alpha=0.85)
    axF.axvline(0, color=MUTED, lw=0.6)
    axF.set_yticks(y); axF.set_yticklabels(surf_df_b.index, fontsize=8)
    axF.set_xlabel('log₁₀ FC  (blood − liver)')
    axF.set_title('F. Blood-enriched surfaceome\n(lymphocyte / myeloid receptors)')

    # Panel G: cumulative expression share — how few plasma-protein genes
    # capture most of the liver transcriptome vs blood
    axG = fig.add_subplot(gs[2, 1])
    for tis, col in [('Liver', C_LIVER), ('Whole Blood', C_BLOOD)]:
        plasma_in_tis = df[df['category'] == 'plasma'].sort_values(f'share_{tis}', ascending=False)
        cum = plasma_in_tis[f'share_{tis}'].cumsum().values * 100
        axG.plot(np.arange(1, len(cum) + 1), cum, color=col, lw=2,
                 label=f'{tis} (total {cum[-1]:.1f}%)')
    axG.set_xscale('log')
    axG.set_xlabel('Top-K plasma-protein genes (log)')
    axG.set_ylabel('Cumulative % of total reads')
    axG.set_title('G. Cumulative plasma-protein share\nof the transcriptome')
    axG.legend(loc='lower right', frameon=False, fontsize=8)
    axG.grid(True, alpha=0.3)

    # Panel H: hypergeometric enrichment p-values in top-500
    axH = fig.add_subplot(gs[2, 2])
    from scipy.stats import hypergeom
    N = len(df)
    results = []
    for tis in ['Liver', 'Whole Blood']:
        top500 = df.sort_values(tis, ascending=False).head(500).index
        for cat, label in [('plasma', 'Plasma'), ('surfaceome', 'Surface'),
                           ('both', 'Surf+secr')]:
            K = (df['category'] == cat).sum()
            k = df.loc[top500, 'category'].eq(cat).sum()
            # P(X >= k) under hypergeometric (K successes in N, draw 500)
            pval = hypergeom.sf(k - 1, N, K, 500) if K > 0 else 1.0
            fold = (k / 500) / (K / N) if K > 0 else 0
            results.append({'tissue': tis, 'cat': label, 'k': k, 'K': K,
                            'fold': fold, 'log10p': -np.log10(max(pval, 1e-300))})
    enr_df = pd.DataFrame(results)
    cats = ['Plasma', 'Surface', 'Surf+secr']
    x = np.arange(len(cats)); w = 0.38
    for i, tis in enumerate(['Liver', 'Whole Blood']):
        sub = enr_df[enr_df['tissue'] == tis].set_index('cat').loc[cats]
        color = C_LIVER if tis == 'Liver' else C_BLOOD
        axH.bar(x + (i - 0.5) * w, sub['fold'], w, color=color, alpha=0.85, label=tis)
        for xi, f, lp, kk in zip(x + (i - 0.5) * w, sub['fold'], sub['log10p'], sub['k']):
            axH.text(xi, f + 0.2, f'n={kk}\n–log₁₀p={lp:.0f}',
                     ha='center', va='bottom', fontsize=7, color=MUTED)
    axH.axhline(1, color=MUTED, lw=0.6, ls='--')
    axH.set_xticks(x); axH.set_xticklabels(cats)
    axH.set_ylabel('Fold-enrichment in top-500')
    axH.set_title('H. Category enrichment in top-500\n(hypergeometric)')
    axH.legend(frameon=False, fontsize=8)

    fig.suptitle(
        'GTEx v11 — Liver ↔ Blood interaction via the surfaceome and secretome',
        fontsize=14, fontweight='bold', y=0.995
    )
    fig.savefig(OUT_PNG, dpi=140, bbox_inches='tight', facecolor=BG)
    print(f'\n✓ Wrote {OUT_PNG}')

    # ───────────────────── text summary ─────────────────────
    print('\n─── Top 15 plasma-protein genes by liver expression ───')
    top_plasma = df[df['category'] == 'plasma'].sort_values('Liver', ascending=False).head(15)
    print(top_plasma[['Liver', 'Whole Blood', 'share_Liver']].round(3).to_string())

    print('\n─── Top 15 liver-enriched surfaceome genes ───')
    print(surf_df.sort_values('liver_minus_blood', ascending=False).head(15)
          [['Liver', 'Whole Blood', 'liver_minus_blood']].round(3).to_string())

    print('\n─── Transcriptome share by category ───')
    for t in TISSUES:
        print(f'  {t}:')
        for c in cats_show:
            s = df[df['category'] == c][f'share_{t}'].sum()
            print(f'    {CAT_LABEL[c]:25s} {s*100:6.2f}%')


if __name__ == '__main__':
    main()
