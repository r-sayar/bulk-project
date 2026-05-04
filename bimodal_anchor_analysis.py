"""
Bimodal anchor gene analysis for GTEx whole blood.
Identifies anchor genes that reliably distinguish two transcriptomic states.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# 1. Load GTEx whole blood data
print("Loading GTEx whole blood data...")
df = pd.read_csv(
    '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz',
    sep='\t', skiprows=2, compression='gzip'
)
gene_names = df['Name'].values if 'Name' in df.columns else df.iloc[:, 0].values
gene_desc = df['Description'].values if 'Description' in df.columns else df.iloc[:, 1].values
expr = df.iloc[:, 2:].values.astype(np.float64)
sample_ids = df.columns[2:].tolist()
print(f"Loaded {expr.shape[0]} genes x {expr.shape[1]} samples")

# 2. Compute log2(CPM+1)
print("Computing log2(CPM+1)...")
lib_sizes = expr.sum(axis=0)
cpm = expr / lib_sizes * 1e6
log_cpm = np.log2(cpm + 1)

# 3. Find bimodal genes using KDE + peak detection
print("Finding bimodal genes...")
gene_means = log_cpm.mean(axis=1)
gene_stds = log_cpm.std(axis=1)

candidate_mask = (gene_means > 1) & (gene_stds > 0.3)
candidate_idx = np.where(candidate_mask)[0]
print(f"Candidate genes (mean>1, std>0.3): {len(candidate_idx)}")

bimodal_genes = []
bimodal_peaks = {}

for i in candidate_idx:
    vals = log_cpm[i, :]
    try:
        kde = gaussian_kde(vals, bw_method='scott')
    except Exception:
        continue
    x_grid = np.linspace(vals.min(), vals.max(), 500)
    density = kde(x_grid)

    prominence_thresh = 0.08 * density.max()
    peaks, properties = find_peaks(density, prominence=prominence_thresh, distance=40)

    if len(peaks) >= 2:
        # Take the two tallest peaks
        peak_heights = density[peaks]
        top2 = np.argsort(peak_heights)[-2:]
        top2_sorted = sorted(top2, key=lambda idx: x_grid[peaks[idx]])
        p1, p2 = peaks[top2_sorted[0]], peaks[top2_sorted[1]]
        bimodal_genes.append(i)
        bimodal_peaks[i] = (x_grid[p1], x_grid[p2])

print(f"Bimodal genes found: {len(bimodal_genes)}")

# 4. Build binary matrix (threshold = midpoint between two tallest modes)
print("Building binary matrix...")
binary_matrix = np.zeros((len(bimodal_genes), log_cpm.shape[1]), dtype=np.int8)

for row_idx, gene_idx in enumerate(bimodal_genes):
    p1, p2 = bimodal_peaks[gene_idx]
    threshold = (p1 + p2) / 2.0
    binary_matrix[row_idx, :] = (log_cpm[gene_idx, :] > threshold).astype(np.int8)

# 5. PCA on binary matrix, split by PC1
print("Running PCA on binary matrix...")
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(binary_matrix.T)  # samples x components

state_a_mask = pca_result[:, 0] > 0
state_b_mask = pca_result[:, 0] <= 0

n_a = state_a_mask.sum()
n_b = state_b_mask.sum()
print(f"PCA split: State A = {n_a} samples, State B = {n_b} samples")

# 6. Find anchor genes at 98% threshold
print("Finding anchor genes...")

anchor_genes_98 = []
anchor_genes_95 = []
anchor_info = {}

for row_idx, gene_idx in enumerate(bimodal_genes):
    vals_a = binary_matrix[row_idx, state_a_mask]
    vals_b = binary_matrix[row_idx, state_b_mask]

    frac_a_high = vals_a.mean()
    frac_a_low = 1 - frac_a_high
    frac_b_high = vals_b.mean()
    frac_b_low = 1 - frac_b_high

    # Check if >threshold% of State A is in one mode AND >threshold% of State B is in the other
    # Case 1: A is mostly HIGH, B is mostly LOW
    # Case 2: A is mostly LOW, B is mostly HIGH

    is_anchor_98 = False
    is_anchor_95 = False
    a_mode = None
    b_mode = None

    if frac_a_high > 0.98 and frac_b_low > 0.98:
        is_anchor_98 = True
        a_mode = "HIGH"
        b_mode = "LOW"
    elif frac_a_low > 0.98 and frac_b_high > 0.98:
        is_anchor_98 = True
        a_mode = "LOW"
        b_mode = "HIGH"

    if frac_a_high > 0.95 and frac_b_low > 0.95:
        is_anchor_95 = True
        if a_mode is None:
            a_mode = "HIGH"
            b_mode = "LOW"
    elif frac_a_low > 0.95 and frac_b_high > 0.95:
        is_anchor_95 = True
        if a_mode is None:
            a_mode = "LOW"
            b_mode = "HIGH"

    if is_anchor_98:
        anchor_genes_98.append(row_idx)
        anchor_info[row_idx] = {
            'gene_idx': gene_idx,
            'gene_name': gene_names[gene_idx],
            'gene_desc': gene_desc[gene_idx],
            'a_mode': a_mode,
            'b_mode': b_mode,
            'frac_a_high': frac_a_high,
            'frac_b_high': frac_b_high,
            'threshold': 0.98
        }

    if is_anchor_95:
        anchor_genes_95.append(row_idx)
        if row_idx not in anchor_info:
            anchor_info[row_idx] = {
                'gene_idx': gene_idx,
                'gene_name': gene_names[gene_idx],
                'gene_desc': gene_desc[gene_idx],
                'a_mode': a_mode,
                'b_mode': b_mode,
                'frac_a_high': frac_a_high,
                'frac_b_high': frac_b_high,
                'threshold': 0.95
            }

print(f"\nAnchor genes at 98% threshold: {len(anchor_genes_98)}")
print(f"Anchor genes at 95% threshold: {len(anchor_genes_95)}")

# 7. Print anchor gene details
print("\n=== ANCHOR GENES (98% threshold) ===")
for row_idx in anchor_genes_98:
    info = anchor_info[row_idx]
    print(f"  {info['gene_name']:20s} | {info['gene_desc'][:40]:40s} | A={info['a_mode']:4s} B={info['b_mode']:4s} | "
          f"frac_A_high={info['frac_a_high']:.3f} frac_B_high={info['frac_b_high']:.3f}")

only_95 = [r for r in anchor_genes_95 if r not in anchor_genes_98]
if only_95:
    print(f"\n=== ADDITIONAL ANCHOR GENES (95% but not 98%) ===")
    for row_idx in only_95:
        info = anchor_info[row_idx]
        print(f"  {info['gene_name']:20s} | {info['gene_desc'][:40]:40s} | A={info['a_mode']:4s} B={info['b_mode']:4s} | "
              f"frac_A_high={info['frac_a_high']:.3f} frac_B_high={info['frac_b_high']:.3f}")

# 8. Filter ambiguous samples using anchor genes (98%)
print("\n=== FILTERING AMBIGUOUS SAMPLES ===")

if len(anchor_genes_98) > 0:
    anchor_binary = binary_matrix[anchor_genes_98, :]

    # For each anchor, determine which binary value corresponds to State A
    # State A expected value for each anchor gene
    expected_a = np.array([1 if anchor_info[r]['a_mode'] == 'HIGH' else 0 for r in anchor_genes_98])

    n_samples = binary_matrix.shape[1]
    agreement_a = np.zeros(n_samples)
    agreement_b = np.zeros(n_samples)

    for s in range(n_samples):
        sample_vals = anchor_binary[:, s]
        # Agreement with State A pattern
        agreement_a[s] = np.mean(sample_vals == expected_a)
        # Agreement with State B pattern (opposite of A)
        agreement_b[s] = np.mean(sample_vals != expected_a)

    # A sample is ambiguous if it doesn't agree >= 80% with either state
    clear_a = agreement_a >= 0.80
    clear_b = agreement_b >= 0.80
    ambiguous = ~(clear_a | clear_b)

    n_ambiguous = ambiguous.sum()
    n_clear_a = (clear_a & ~ambiguous).sum()
    n_clear_b = (clear_b & ~ambiguous).sum()

    print(f"Samples removed as ambiguous (agreement < 80%): {n_ambiguous}")
    print(f"Final State A samples: {n_clear_a}")
    print(f"Final State B samples: {n_clear_b}")
    print(f"Total remaining: {n_clear_a + n_clear_b}")

    # Some samples might agree with both at 80% -- check overlap
    both = (clear_a & clear_b).sum()
    if both > 0:
        print(f"  (Note: {both} samples agree >=80% with both states)")
else:
    print("No anchor genes found at 98% threshold -- cannot filter.")
    n_ambiguous = 0
    n_clear_a = n_a
    n_clear_b = n_b
    ambiguous = np.zeros(binary_matrix.shape[1], dtype=bool)

# 9. Save summary
print("\nSaving summary...")
output_path = '/Users/rls/Desktop/programming-projects/single-cell/bulk-project/bimodal_anchor_summary.txt'

with open(output_path, 'w') as f:
    f.write("BIMODAL ANCHOR GENE ANALYSIS - GTEx Whole Blood\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total genes: {expr.shape[0]}\n")
    f.write(f"Total samples: {expr.shape[1]}\n")
    f.write(f"Bimodal genes detected: {len(bimodal_genes)}\n\n")

    f.write(f"PCA split (before filtering):\n")
    f.write(f"  State A (PC1 > 0):  {n_a} samples\n")
    f.write(f"  State B (PC1 <= 0): {n_b} samples\n\n")

    f.write(f"Anchor genes (98% threshold): {len(anchor_genes_98)}\n")
    f.write(f"Anchor genes (95% threshold): {len(anchor_genes_95)}\n\n")

    f.write("ANCHOR GENES (98% threshold):\n")
    f.write("-" * 60 + "\n")
    for row_idx in anchor_genes_98:
        info = anchor_info[row_idx]
        f.write(f"  {info['gene_name']:20s} | {info['gene_desc'][:50]:50s}\n")
        f.write(f"    State A mode: {info['a_mode']}, State B mode: {info['b_mode']}\n")
        f.write(f"    frac_A_high={info['frac_a_high']:.4f}, frac_B_high={info['frac_b_high']:.4f}\n")
        p1, p2 = bimodal_peaks[info['gene_idx']]
        f.write(f"    Peaks at log2CPM: {p1:.2f}, {p2:.2f} (threshold: {(p1+p2)/2:.2f})\n\n")

    if only_95:
        f.write("\nADDITIONAL ANCHOR GENES (95% but not 98%):\n")
        f.write("-" * 60 + "\n")
        for row_idx in only_95:
            info = anchor_info[row_idx]
            f.write(f"  {info['gene_name']:20s} | {info['gene_desc'][:50]:50s}\n")
            f.write(f"    State A mode: {info['a_mode']}, State B mode: {info['b_mode']}\n")
            f.write(f"    frac_A_high={info['frac_a_high']:.4f}, frac_B_high={info['frac_b_high']:.4f}\n\n")

    f.write("\nSAMPLE FILTERING:\n")
    f.write("-" * 60 + "\n")
    f.write(f"  Ambiguous samples removed: {n_ambiguous}\n")
    if len(anchor_genes_98) > 0:
        f.write(f"  Final State A: {n_clear_a}\n")
        f.write(f"  Final State B: {n_clear_b}\n")
        f.write(f"  Total remaining: {n_clear_a + n_clear_b}\n")
    else:
        f.write(f"  Final State A: {n_a}\n")
        f.write(f"  Final State B: {n_b}\n")

print(f"Summary saved to: {output_path}")
print("Done!")
