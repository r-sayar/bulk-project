#!/Library/Developer/CommandLineTools/usr/bin/python3
"""
Cross-Modality RNA-seq Latent Space Model
==========================================
Projects bulk RNA-seq and scRNA-seq into a shared latent space so that the
same biology (cell type, functional state) maps to nearby points regardless
of which measurement technology produced it.

WHY THIS IS HARD
----------------
Bulk RNA-seq: one number per gene per donor = population-average signal.
  803 GTEx donors × 16k genes → each point is a mixture of ~10 cell types.
scRNA-seq:    one number per gene per cell = single-cell signal.
  Millions of cells × 33k genes → sparse, noisy, but cell-type resolved.

The distributions are very different even after identical normalization:
  • Bulk is smooth (law of large numbers across ~10^4 cells per donor)
  • Single cells are sparse (~15% dropout) and bimodal (0 or expressed)

ARCHITECTURE
------------

    ┌─────────┐     ┌─────────────────┐
    │  Bulk   │────▶│  BulkEncoder    │──┐
    │ (GTEx)  │     │  FC 1024→512    │  │    ┌──────────────┐
    └─────────┘     └─────────────────┘  │    │              │
                                         ├───▶│  Shared      │───▶ Decoder ──▶ recon
    ┌─────────┐     ┌─────────────────┐  │    │  Latent z    │
    │  SC     │────▶│  SCEncoder      │──┘    │  (dim = 64)  │───▶ Domain
    │  cells  │     │  FC 1024→512    │       │              │     Discriminator
    └─────────┘     └─────────────────┘       └──────────────┘     (bulk vs sc)

Both encoders output (μ, log σ²) → reparameterize → z.

LOSS FUNCTIONS
--------------
L_recon  = MSE(x̂, x)              reconstruction quality
L_KL     = KL(q(z|x) ‖ N(0,I))   regularize latent space to a sphere
L_adv    = CE(disc(z), modality)   discriminator learns to separate bulk/sc
                                   encoder is trained to FOOL discriminator
                                   → forces latent space to be modality-agnostic

Total encoder loss: L_recon + β·L_KL − λ·L_adv
Discriminator loss: L_adv  (maximized separately)

β anneals from 0→1 over training (KL annealing avoids posterior collapse).
λ anneals from 0→0.1 over training (give reconstruction time to stabilise first).

USAGE
-----
  python cross_modality_vae.py              # train and evaluate
  python cross_modality_vae.py --epochs 50  # quick test run
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────
BASEDIR   = '/Users/rls/Desktop/programming-projects/single-cell/bulk-project'
GTEX_PATH = '/Users/rls/Downloads/gene_reads_v11_whole_blood.gct.gz'
PSEUDO    = os.path.join(BASEDIR, 'pseudobulk/hca_blood_pseudobulk.npz')
OUT_DIR   = BASEDIR

# ── style ──────────────────────────────────────────────────────────────
BG = '#0e1117'; CARD = '#1a1d23'; TEXT = '#e6edf3'; MUTED = '#7d8590'; GRID = '#21262d'
C_G = '#f78166'; C_H = '#3fb950'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD, 'axes.edgecolor': GRID,
    'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'grid.color': GRID, 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'font.size': 11,
})

# ══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════

CPM_THRESHOLD   = 1
MIN_SAMPLE_FRAC = 0.1


def _cpm_log(expr_raw):
    """CPM normalize then log2(CPM+1)."""
    lib = expr_raw.sum(axis=0, keepdims=True)
    cpm = expr_raw / lib * 1e6
    return np.log2(cpm + 1)


def _filter_genes(expr_raw, gene_names, n_samples=None):
    if n_samples is None:
        n_samples = expr_raw.shape[1]
    lib  = expr_raw.sum(axis=0)
    cpm  = expr_raw / lib * 1e6
    min_s = max(1, int(MIN_SAMPLE_FRAC * n_samples))
    keep  = (cpm > CPM_THRESHOLD).sum(axis=1) >= min_s
    return expr_raw[keep], gene_names[keep]


def load_gtex(path):
    """Load GTEx bulk blood → (samples × genes) float32."""
    print("Loading GTEx bulk blood ...")
    df        = pd.read_csv(path, sep='\t', skiprows=2, compression='gzip')
    expr_raw  = df.iloc[:, 2:].values.astype(np.float64)    # (genes, samples)
    gene_names = df['Description'].values.astype(str)

    expr_filt, names_filt = _filter_genes(expr_raw, gene_names)
    expr_log = _cpm_log(expr_filt)   # (genes, samples)

    print(f"  {expr_log.shape[1]} samples × {expr_log.shape[0]:,} genes")
    # Transpose to (samples × genes) for PyTorch
    return expr_log.T.astype(np.float32), names_filt


def load_hca(path):
    """
    Load HCA pseudobulk → (donors × genes) float32.
    Falls back to pseudobulk .npz if path ends in .npz.
    """
    print("Loading HCA pseudobulk ...")
    d         = np.load(path, allow_pickle=True)
    expr_raw  = d['expr'].astype(np.float64)   # (genes, donors)
    gene_names = d['gene_names'].astype(str)

    expr_filt, names_filt = _filter_genes(expr_raw, gene_names)
    expr_log = _cpm_log(expr_filt)

    print(f"  {expr_log.shape[1]} donors × {expr_log.shape[0]:,} genes")
    return expr_log.T.astype(np.float32), names_filt


def load_h5ad_cells(h5ad_path, max_cells=None, seed=42):
    """
    Load individual cells from a CellxGene h5ad file.
    Returns (cells × genes) log2(CPM+1) float32 and gene names.
    Optionally subsample to max_cells for memory/speed.
    """
    import anndata
    import scipy.sparse as sp

    print(f"Loading {os.path.basename(h5ad_path)} ...")
    adata = anndata.read_h5ad(h5ad_path)
    print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    print(f"  Donors: {adata.obs['donor_id'].nunique()}")

    if max_cells and adata.shape[0] > max_cells:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(adata.shape[0], max_cells, replace=False))
        # Subset BEFORE densifying — critical for memory efficiency.
        # anndata slicing on a sparse matrix selects rows without full densification.
        adata = adata[idx].copy()
        print(f"  Subsampled to {max_cells:,} cells")

    # Densify after subsetting (now only max_cells rows, not all 389k)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float64)   # (cells × genes)

    # CellxGene h5ad files use Ensembl IDs as var_names;
    # gene symbols are in 'feature_name' or 'gene_symbols' column.
    if 'feature_name' in adata.var.columns:
        gene_names = np.array(adata.var['feature_name'].astype(str))
    elif 'gene_symbols' in adata.var.columns:
        gene_names = np.array(adata.var['gene_symbols'].astype(str))
    else:
        gene_names = np.array([str(g) for g in adata.var_names])

    # Filter genes expressed in ≥10% of cells
    expr_filt, names_filt = _filter_genes(X.T, gene_names, n_samples=X.shape[0])
    # CPM + log2
    expr_log = _cpm_log(expr_filt).T   # back to (cells × genes)

    print(f"  After filter: {expr_log.shape[0]:,} cells × {expr_log.shape[1]:,} genes")
    return expr_log.astype(np.float32), names_filt


def get_shared_genes(bulk_expr, bulk_names, sc_expr, sc_names):
    """
    Intersect gene sets; return aligned (n_shared,) arrays.
    Duplicate gene names are resolved by keeping the first occurrence.
    """
    bulk_map = {}
    for i, n in enumerate(bulk_names):
        if n.upper() not in bulk_map:
            bulk_map[n.upper()] = i

    sc_map = {}
    for i, n in enumerate(sc_names):
        if n.upper() not in sc_map:
            sc_map[n.upper()] = i

    shared = sorted(set(bulk_map.keys()) & set(sc_map.keys()))
    bulk_idx = np.array([bulk_map[g] for g in shared])
    sc_idx   = np.array([sc_map[g]   for g in shared])

    print(f"  Shared genes: {len(shared):,}")
    return (bulk_expr[:, bulk_idx],
            sc_expr[:, sc_idx],
            np.array(shared))


# ══════════════════════════════════════════════════════════════════════
# 2. MODEL
# ══════════════════════════════════════════════════════════════════════

def _fc_block(in_dim, out_dim, dropout=0.1, bn=True):
    """Fully-connected block: Linear → BatchNorm → LeakyReLU → Dropout."""
    layers = [nn.Linear(in_dim, out_dim)]
    if bn:
        layers.append(nn.BatchNorm1d(out_dim))
    layers += [nn.LeakyReLU(0.2), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class ModalityEncoder(nn.Module):
    """
    Modality-specific encoder.
    Maps (n_genes,) → (latent_dim,) via μ and log σ².
    Using separate encoders per modality lets each one learn to handle
    its own noise structure (bulk smooth vs sc sparse).
    """
    def __init__(self, n_genes, hidden=(1024, 512), latent_dim=64, dropout=0.1):
        super().__init__()
        dims = [n_genes] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            # Only use BatchNorm if we're likely to have > 1 sample per batch
            layers.append(_fc_block(dims[i], dims[i+1], dropout=dropout, bn=True))
        self.net  = nn.Sequential(*layers)
        self.mu   = nn.Linear(hidden[-1], latent_dim)
        self.logv = nn.Linear(hidden[-1], latent_dim)

    def forward(self, x):
        h    = self.net(x)
        mu   = self.mu(h)
        logv = self.logv(h).clamp(-10, 4)   # clamp for numerical stability
        return mu, logv


class Decoder(nn.Module):
    """
    Shared decoder: z → reconstructed gene expression.
    A single decoder for both modalities forces the latent space to capture
    expression patterns that are meaningful regardless of measurement platform.
    """
    def __init__(self, latent_dim, n_genes, hidden=(512, 1024), dropout=0.1):
        super().__init__()
        dims = [latent_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(_fc_block(dims[i], dims[i+1], dropout=dropout, bn=True))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden[-1], n_genes)

    def forward(self, z):
        return self.out(self.net(z))


class DomainDiscriminator(nn.Module):
    """
    Binary classifier: is this latent code from bulk or single-cell?
    Trained adversarially against the encoders — the discriminator tries
    to tell them apart, the encoders try to fool it.
    When the discriminator can only guess at 50% accuracy, the latent
    space has been successfully aligned across modalities.
    """
    def __init__(self, latent_dim, hidden=(128, 64)):
        super().__init__()
        dims = [latent_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.LeakyReLU(0.2), nn.Dropout(0.1)]
        layers.append(nn.Linear(hidden[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class CrossModalityVAE(nn.Module):
    """
    Full cross-modality VAE.

    Forward pass returns a dict with all intermediate quantities needed
    to compute the combined loss.
    """
    def __init__(self, n_genes, latent_dim=64,
                 enc_hidden=(1024, 512), dec_hidden=(512, 1024),
                 disc_hidden=(128, 64), dropout=0.1):
        super().__init__()
        self.n_genes    = n_genes
        self.latent_dim = latent_dim

        # Modality-specific encoders (same architecture, separate weights)
        self.enc_bulk = ModalityEncoder(n_genes, enc_hidden, latent_dim, dropout)
        self.enc_sc   = ModalityEncoder(n_genes, enc_hidden, latent_dim, dropout)

        # Shared decoder
        self.decoder = Decoder(latent_dim, n_genes, dec_hidden, dropout)

        # Domain discriminator (separate optimizer, not in main model params)
        self.discriminator = DomainDiscriminator(latent_dim, disc_hidden)

    @staticmethod
    def reparameterize(mu, logv):
        """Sample z ~ N(μ, σ²) via the reparameterization trick."""
        if not CrossModalityVAE.training:  # eval mode → use mean
            return mu
        std = (0.5 * logv).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    # Make reparameterize work as an instance method too
    def _reparam(self, mu, logv):
        if not self.training:
            return mu
        std = (0.5 * logv).exp()
        return mu + std * torch.randn_like(std)

    def encode(self, x, modality):
        """modality: 'bulk' or 'sc'"""
        enc = self.enc_bulk if modality == 'bulk' else self.enc_sc
        mu, logv = enc(x)
        z = self._reparam(mu, logv)
        return z, mu, logv

    def forward(self, x_bulk, x_sc):
        z_bulk, mu_bulk, logv_bulk = self.encode(x_bulk, 'bulk')
        z_sc,   mu_sc,   logv_sc   = self.encode(x_sc,   'sc')

        recon_bulk = self.decoder(z_bulk)
        recon_sc   = self.decoder(z_sc)

        disc_bulk = self.discriminator(z_bulk)
        disc_sc   = self.discriminator(z_sc)

        return {
            'z_bulk': z_bulk,   'z_sc': z_sc,
            'mu_bulk': mu_bulk, 'mu_sc': mu_sc,
            'logv_bulk': logv_bulk, 'logv_sc': logv_sc,
            'recon_bulk': recon_bulk, 'recon_sc': recon_sc,
            'disc_bulk': disc_bulk, 'disc_sc': disc_sc,
        }


def kl_divergence(mu, logv):
    """KL(N(μ,σ²) ‖ N(0,1)) per sample, summed over latent dims."""
    return -0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum(dim=-1).mean()


def compute_losses(out, x_bulk, x_sc, beta, lam):
    """
    Compute all losses and return a dict.

    beta: KL annealing weight (0→1 during warmup)
    lam:  adversarial weight  (0→0.1 during warmup)
    """
    # Reconstruction: how well can we recover expression from z?
    l_recon_bulk = F.mse_loss(out['recon_bulk'], x_bulk)
    l_recon_sc   = F.mse_loss(out['recon_sc'],   x_sc)
    l_recon      = (l_recon_bulk + l_recon_sc) / 2

    # KL divergence: regularise z towards N(0,I)
    l_kl_bulk = kl_divergence(out['mu_bulk'], out['logv_bulk'])
    l_kl_sc   = kl_divergence(out['mu_sc'],   out['logv_sc'])
    l_kl      = (l_kl_bulk + l_kl_sc) / 2

    # Adversarial: discriminator distinguishes bulk (label=0) vs sc (label=1)
    # labels
    lbl_bulk = torch.zeros(out['z_bulk'].size(0), dtype=torch.long,
                           device=out['z_bulk'].device)
    lbl_sc   = torch.ones( out['z_sc'].size(0),   dtype=torch.long,
                           device=out['z_sc'].device)

    l_disc = (F.cross_entropy(out['disc_bulk'], lbl_bulk) +
              F.cross_entropy(out['disc_sc'],   lbl_sc)) / 2

    # Encoder adversarial loss: fool the discriminator (swap labels)
    # We want the encoder to produce z that the discriminator CANNOT classify
    l_adv_enc = (F.cross_entropy(out['disc_bulk'], lbl_sc) +
                 F.cross_entropy(out['disc_sc'],   lbl_bulk)) / 2

    # Discriminator accuracy (for monitoring)
    with torch.no_grad():
        pred_bulk = out['disc_bulk'].argmax(1)
        pred_sc   = out['disc_sc'].argmax(1)
        disc_acc  = ((pred_bulk == 0).float().mean() +
                     (pred_sc   == 1).float().mean()) / 2

    # Total encoder loss
    l_encoder = l_recon + beta * l_kl - lam * l_adv_enc

    return {
        'encoder':   l_encoder,
        'disc':      l_disc,
        'recon':     l_recon.item(),
        'kl':        l_kl.item(),
        'adv_enc':   l_adv_enc.item(),
        'disc_acc':  disc_acc.item(),
    }


# ══════════════════════════════════════════════════════════════════════
# 3. TRAINING
# ══════════════════════════════════════════════════════════════════════

def make_loaders(bulk_t, sc_t, batch_size=64):
    """
    Create DataLoaders for bulk and sc tensors.
    Handles both cases:
      SC << bulk: oversample SC so each epoch is balanced.
      SC >> bulk: both get their own DataLoader, zip stops at shorter one.
    """
    bulk_ds = TensorDataset(bulk_t)
    sc_ds   = TensorDataset(sc_t)

    n_bulk_batches = max(1, len(bulk_t) // batch_size)

    if len(sc_t) < len(bulk_t):
        # SC is smaller — repeat it to match bulk epoch length
        n_needed = n_bulk_batches * batch_size
        repeats  = max(1, -(-n_needed // len(sc_t)))          # ceiling division
        sc_rep   = sc_t.repeat(repeats, 1)[:n_needed]
        sc_rep   = sc_rep[torch.randperm(len(sc_rep))]
        sc_ds    = TensorDataset(sc_rep)

    bulk_loader = DataLoader(bulk_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    sc_loader   = DataLoader(sc_ds,   batch_size=batch_size, shuffle=True,  drop_last=True)
    return bulk_loader, sc_loader


def train(model, bulk_t, sc_t, epochs=150, lr=1e-3, batch_size=64,
          beta_max=1.0, lam_max=0.1, warmup_frac=0.3, device='cpu'):

    model = model.to(device)
    bulk_t, sc_t = bulk_t.to(device), sc_t.to(device)

    # Separate optimizers: encoder/decoder vs discriminator
    enc_dec_params = (list(model.enc_bulk.parameters()) +
                      list(model.enc_sc.parameters()) +
                      list(model.decoder.parameters()))
    opt_enc  = torch.optim.Adam(enc_dec_params, lr=lr, weight_decay=1e-5)
    opt_disc = torch.optim.Adam(model.discriminator.parameters(), lr=lr * 0.5)

    sched_enc  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_enc,  T_max=epochs)
    sched_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=epochs)

    warmup_epochs = max(1, int(epochs * warmup_frac))
    history = {k: [] for k in ('recon', 'kl', 'adv_enc', 'disc_acc', 'beta', 'lam')}

    print(f"\nTraining on {device}  |  {epochs} epochs  |  batch={batch_size}")
    print(f"  Bulk: {bulk_t.shape[0]} samples × {bulk_t.shape[1]} genes")
    print(f"  SC:   {sc_t.shape[0]} donors   × {sc_t.shape[1]} genes")
    print(f"  Latent dim: {model.latent_dim}")
    print(f"  KL warmup: {warmup_epochs} epochs, adv warmup: {warmup_epochs} epochs")
    print()

    for epoch in range(1, epochs + 1):
        model.train()
        bulk_loader, sc_loader = make_loaders(bulk_t, sc_t, batch_size)

        # Anneal β and λ linearly during warmup
        beta = min(beta_max, beta_max * epoch / warmup_epochs)
        lam  = min(lam_max,  lam_max  * epoch / warmup_epochs)

        epoch_stats = {k: 0.0 for k in ('recon', 'kl', 'adv_enc', 'disc_acc')}
        n_batches = 0

        for (xb,), (xs,) in zip(bulk_loader, sc_loader):
            # ── Step 1: Update discriminator (keep encoder frozen) ───────
            # Detach z from the computation graph so gradients only flow
            # through the discriminator, not the encoders.
            with torch.no_grad():
                z_bulk = model.enc_bulk(xb)[0]  # just mu for speed
                z_sc   = model.enc_sc(xs)[0]
            disc_bulk = model.discriminator(z_bulk)
            disc_sc   = model.discriminator(z_sc)
            lbl_bulk  = torch.zeros(xb.size(0), dtype=torch.long, device=device)
            lbl_sc    = torch.ones( xs.size(0), dtype=torch.long, device=device)
            l_disc    = (F.cross_entropy(disc_bulk, lbl_bulk) +
                         F.cross_entropy(disc_sc,   lbl_sc)) / 2

            opt_disc.zero_grad()
            l_disc.backward()
            opt_disc.step()

            # ── Step 2: Update encoders + decoder ───────────────────────
            out    = model(xb, xs)
            losses = compute_losses(out, xb, xs, beta, lam)

            opt_enc.zero_grad()
            losses['encoder'].backward()
            nn.utils.clip_grad_norm_(enc_dec_params, max_norm=5.0)
            opt_enc.step()

            for k in epoch_stats:
                epoch_stats[k] += losses[k]
            n_batches += 1

        sched_enc.step(); sched_disc.step()

        for k in epoch_stats:
            history[k].append(epoch_stats[k] / n_batches)
        history['beta'].append(beta)
        history['lam'].append(lam)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{epochs}  "
                  f"recon={history['recon'][-1]:.4f}  "
                  f"KL={history['kl'][-1]:.4f}  "
                  f"adv={history['adv_enc'][-1]:.4f}  "
                  f"disc_acc={history['disc_acc'][-1]:.2%}  "
                  f"β={beta:.3f}  λ={lam:.4f}")

    return history


# ══════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def get_latent(model, bulk_t, sc_t, device='cpu'):
    """Extract latent codes for both datasets."""
    model.eval()
    model = model.to(device)
    bulk_t, sc_t = bulk_t.to(device), sc_t.to(device)
    z_bulk = model.enc_bulk(bulk_t)[0].cpu().numpy()   # use μ (mean)
    z_sc   = model.enc_sc(sc_t)[0].cpu().numpy()
    return z_bulk, z_sc


def pca_baseline(bulk_expr, sc_expr, n_components=50):
    """
    Naïve baseline: concatenate and PCA together.
    Does NOT account for the domain gap — bulk and sc will cluster
    apart unless the signal overwhelms the batch effect.
    """
    combined = np.vstack([bulk_expr, sc_expr])
    pca = PCA(n_components=min(n_components, combined.shape[0] - 1, combined.shape[1]))
    pca.fit(combined)
    z_all = pca.transform(combined)
    return z_all[:len(bulk_expr)], z_all[len(bulk_expr):]


def alignment_score(z_bulk, z_sc, k=5):
    """
    kNN mixing score: for each sc point, what fraction of its k nearest
    neighbours (in the combined space) are bulk?
    1.0 = perfectly mixed; ~(n_bulk/n_total) = random mixing.
    """
    from sklearn.neighbors import NearestNeighbors
    combined = np.vstack([z_bulk, z_sc])
    labels   = np.array([0] * len(z_bulk) + [1] * len(z_sc))
    nn  = NearestNeighbors(n_neighbors=k + 1).fit(combined)
    idx = nn.kneighbors(combined, return_distance=False)[:, 1:]   # exclude self
    # For each sc point, fraction of neighbours that are bulk
    sc_indices = np.where(labels == 1)[0]
    fracs = [(labels[idx[i]] == 0).mean() for i in sc_indices]
    return float(np.mean(fracs))


# ══════════════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ══════════════════════════════════════════════════════════════════════

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def _reduce_2d(z_bulk, z_sc, method='umap'):
    """Reduce to 2D for plotting using PCA or UMAP (falls back to t-SNE)."""
    combined = np.vstack([z_bulk, z_sc])
    n_pca = min(50, combined.shape[0] - 1, combined.shape[1])
    pca_coords = PCA(n_components=n_pca).fit_transform(combined)

    if method == 'umap' and HAS_UMAP:
        n_neighbors = min(15, combined.shape[0] - 1)
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                                min_dist=0.3, random_state=42, verbose=False)
        coords = reducer.fit_transform(pca_coords)
    elif method in ('tsne', 'umap'):   # fallback
        perp = min(30, combined.shape[0] // 3)
        coords = TSNE(n_components=2, perplexity=max(5, perp),
                      random_state=42).fit_transform(pca_coords)
    else:
        coords = pca_coords[:, :2]
    return coords[:len(z_bulk)], coords[len(z_bulk):]


def fig_results(z_bulk_vae, z_sc_vae,
                z_bulk_pca, z_sc_pca,
                bulk_expr, sc_expr,
                history, out_path):
    """
    6-panel figure:
      1. Training curves (recon, KL, adversarial)
      2. Discriminator accuracy over training (should approach 50%)
      3. PCA of VAE latent space (PCA on z, coloured by modality)
      4. t-SNE of VAE latent space
      5. Naïve PCA baseline (no alignment)
      6. Reconstruction quality: bulk predicted vs actual for top genes
    """
    n_bulk = len(z_bulk_vae)
    n_sc   = len(z_sc_vae)

    fig = plt.figure(figsize=(26, 18))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30,
                   left=0.06, right=0.97, top=0.91, bottom=0.07)
    fig.suptitle('Cross-Modality VAE: Bulk + scRNA-seq in Shared Latent Space',
                 fontsize=17, fontweight='bold', color=TEXT, y=0.97)

    # ── Panel 1: training curves ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['recon']) + 1)
    ax2 = ax.twinx()
    ax.plot(epochs, history['recon'],   color='#f78166', lw=2, label='Recon (MSE)')
    ax.plot(epochs, history['kl'],      color='#58a6ff', lw=2, label='KL')
    ax.plot(epochs, history['adv_enc'], color='#d2a8ff', lw=2, label='Adv (encoder)')
    ax2.plot(epochs, history['beta'], color=MUTED, lw=1.5, ls='--', label='β (KL weight)')
    ax2.plot(epochs, history['lam'],  color='#f0883e', lw=1.5, ls=':', label='λ (adv weight)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax2.set_ylabel('Annealing weights', color=MUTED)
    ax2.tick_params(colors=MUTED)
    ax.set_title('Training Curves', fontsize=14, fontweight='bold', pad=10)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
              facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(alpha=0.3)

    # ── Panel 2: discriminator accuracy ──────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, [a * 100 for a in history['disc_acc']],
            color='#3fb950', lw=2, label='Disc accuracy')
    ax.axhline(50, color=MUTED, ls='--', lw=1.5, label='50% = perfectly aligned')
    ax.fill_between(epochs, 45, 55, alpha=0.15, color=MUTED)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Discriminator Accuracy (%)')
    ax.set_title('Domain Alignment Progress\n(50% = bulk and sc indistinguishable)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    # ── Panel 3: PCA of VAE latent space ─────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    pb, ps = _reduce_2d(z_bulk_vae, z_sc_vae, method='pca')
    ax.scatter(pb[:, 0], pb[:, 1], s=10, alpha=0.4, c=C_G,
               label=f'GTEx Bulk (n={n_bulk})', rasterized=True)
    ax.scatter(ps[:, 0], ps[:, 1], s=80, alpha=0.9, c=C_H,
               label=f'HCA SC (n={n_sc})', zorder=5, edgecolors='white', lw=0.5)
    score = alignment_score(z_bulk_vae, z_sc_vae)
    ax.text(0.05, 0.95, f'kNN mixing score = {score:.3f}',
            transform=ax.transAxes, fontsize=10, color=TEXT, va='top',
            bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.85, pad=4))
    ax.set_title('VAE Latent Space — PCA projection', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(alpha=0.3)

    # ── Panel 4: t-SNE of VAE latent space ───────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    tb, ts = _reduce_2d(z_bulk_vae, z_sc_vae, method='tsne')
    ax.scatter(tb[:, 0], tb[:, 1], s=10, alpha=0.4, c=C_G,
               label=f'GTEx Bulk (n={n_bulk})', rasterized=True)
    ax.scatter(ts[:, 0], ts[:, 1], s=80, alpha=0.9, c=C_H,
               label=f'HCA SC (n={n_sc})', zorder=5, edgecolors='white', lw=0.5)
    ax.set_title('VAE Latent Space — UMAP projection', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(alpha=0.3)

    # ── Panel 5: Naïve PCA baseline (no alignment) ───────────────────
    ax = fig.add_subplot(gs[0, 2])
    pb_base, ps_base = z_bulk_pca, z_sc_pca
    ax.scatter(pb_base[:, 0], pb_base[:, 1], s=10, alpha=0.4, c=C_G,
               label=f'GTEx Bulk', rasterized=True)
    ax.scatter(ps_base[:, 0], ps_base[:, 1], s=80, alpha=0.9, c=C_H,
               label=f'HCA SC', zorder=5, edgecolors='white', lw=0.5)
    score_base = alignment_score(pb_base, ps_base)
    ax.text(0.05, 0.95, f'kNN mixing score = {score_base:.3f}',
            transform=ax.transAxes, fontsize=10, color=TEXT, va='top',
            bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.85, pad=4))
    ax.set_title('Naïve PCA Baseline (no alignment)\n→ compare with VAE panels',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(alpha=0.3)

    # ── Panel 6: per-dimension latent variance bar chart ─────────────
    ax = fig.add_subplot(gs[1, 2])
    combined_z = np.vstack([z_bulk_vae, z_sc_vae])
    var_per_dim = combined_z.var(axis=0)
    # also compute per-modality variance to see specialisation
    var_bulk = z_bulk_vae.var(axis=0)
    var_sc   = z_sc_vae.var(axis=0)
    sort_idx = np.argsort(var_per_dim)[::-1][:30]
    x = np.arange(30); w = 0.3
    ax.bar(x - w, var_bulk[sort_idx], w, color=C_G, alpha=0.7, label='Bulk variance')
    ax.bar(x,     var_sc[sort_idx],   w, color=C_H, alpha=0.7, label='SC variance')
    ax.bar(x + w, var_per_dim[sort_idx], w, color='#58a6ff', alpha=0.7, label='Combined')
    ax.set_xlabel('Latent dimension (top 30 by total variance)')
    ax.set_ylabel('Variance')
    ax.set_title('Per-Dimension Latent Variance\n(should overlap → shared structure)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(axis='y', alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# 7. CCA BASELINE (works with small n)
# When we have very few SC samples (e.g. 8 pseudobulk donors vs 803 bulk)
# the adversarial VAE cannot align properly because the discriminator
# trivially separates 8 vs 803.
# CCA finds linear projections of two datasets that maximise correlation
# between them — it works with any n ≥ 2 and is the correct method here.
# When the full single-cell data (100k+ individual cells) is available,
# switch back to the VAE for non-linear alignment.
# ══════════════════════════════════════════════════════════════════════

def aggregate_bulk_to_pseudobulk(bulk_raw_expr, bulk_names, n_groups=100, seed=42):
    """
    Pool GTEx individual donors into n_groups pseudobulk groups by summing
    raw counts, then CPM+log normalise. This puts bulk on the same 'scale'
    as HCA pseudobulk so direct CCA comparison is valid.
    Returns (n_groups × n_genes) expression matrix.
    """
    rng = np.random.default_rng(seed)
    n_samples = bulk_raw_expr.shape[0]
    idx = rng.permutation(n_samples)
    chunks = np.array_split(idx, n_groups)

    # bulk_raw_expr is (samples × genes); we need (genes × samples) for filter
    expr_T = bulk_raw_expr.T   # (genes, samples)

    agg = np.zeros((expr_T.shape[0], n_groups), dtype=np.float64)
    for g, chunk in enumerate(chunks):
        agg[:, g] = expr_T[:, chunk].sum(axis=1)

    # CPM + log2
    lib = agg.sum(axis=0, keepdims=True)
    cpm = agg / lib * 1e6
    agg_log = np.log2(cpm + 1)
    return agg_log.T.astype(np.float32)  # (n_groups × n_genes)


def cca_alignment(bulk_pb, sc_expr, n_components=None):
    """
    Canonical Correlation Analysis on pseudobulk representations.
    Both inputs must have the same number of rows (samples).
    bulk_pb: (n, genes) — aggregated bulk pseudo-groups
    sc_expr: (n, genes) — HCA pseudobulk donors

    n_components defaults to min(n_samples, 8).
    NOTE: sklearn CCA requires n_samples_X == n_samples_Y, so we fit on
    the paired aggregated datasets, then transform all bulk individually.
    """
    from sklearn.cross_decomposition import CCA
    n = min(len(bulk_pb), len(sc_expr))
    n_components = n_components or min(n - 1, 8)

    print(f"\nCCA on pseudobulk-matched data (n_components={n_components}) ...")
    # Standardise per gene within each dataset before CCA
    s1 = StandardScaler().fit_transform(bulk_pb)
    s2 = StandardScaler().fit_transform(sc_expr)

    # CCA fit on first min(n,n) rows
    cca = CCA(n_components=n_components, max_iter=5000)
    cca.fit(s1[:n], s2[:n])

    z_bulk_cca = cca.transform(s1)[:, :2]   # project all bulk groups
    z_sc_cca   = cca.transform(s2)[:, :2]   # project all SC donors

    corrs = [np.corrcoef(cca.transform(s1[:n])[:, i],
                         cca.transform(s2[:n])[:, i])[0, 1]
             for i in range(n_components)]
    print(f"  Canonical correlations: {[f'{c:.3f}' for c in corrs]}")
    return z_bulk_cca, z_sc_cca, corrs, cca


def fig_cca(z_bulk_cca, z_sc_cca, z_bulk_pca, z_sc_pca, corrs, out_path):
    """Compare CCA alignment vs naïve PCA."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
    fig.suptitle('Cross-Modality Alignment: GTEx Bulk vs HCA Pseudobulk\n'
                 '(Note: only 8 SC donors — will improve dramatically with individual cells)',
                 fontsize=15, fontweight='bold', color=TEXT, y=1.01)

    for ax in axes:
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.tick_params(colors=MUTED)

    n_bulk = len(z_bulk_cca); n_sc = len(z_sc_cca)

    # Panel 1: PCA baseline
    ax = axes[0]
    ax.scatter(z_bulk_pca[:, 0], z_bulk_pca[:, 1], s=8, alpha=0.35, c=C_G,
               label=f'GTEx Bulk (n={n_bulk})', rasterized=True)
    ax.scatter(z_sc_pca[:, 0],   z_sc_pca[:, 1],   s=100, alpha=0.95, c=C_H,
               label=f'HCA Pseudo (n={n_sc})', zorder=5, edgecolors='white', lw=0.8)
    score_pca = alignment_score(z_bulk_pca, z_sc_pca)
    ax.text(0.05, 0.95, f'mixing score = {score_pca:.4f}',
            transform=ax.transAxes, fontsize=11, color=TEXT, va='top',
            bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.85, pad=4))
    ax.set_title('Naïve PCA\n(no alignment)', fontsize=13, fontweight='bold', pad=10,
                 color=TEXT)
    ax.set_xlabel('PC1', color=MUTED); ax.set_ylabel('PC2', color=MUTED)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(alpha=0.3)

    # Panel 2: CCA projection (top 2 components)
    ax = axes[1]
    ax.scatter(z_bulk_cca[:, 0], z_bulk_cca[:, 1], s=8, alpha=0.35, c=C_G,
               label=f'GTEx Bulk (n={n_bulk})', rasterized=True)
    ax.scatter(z_sc_cca[:, 0],   z_sc_cca[:, 1],   s=100, alpha=0.95, c=C_H,
               label=f'HCA Pseudo (n={n_sc})', zorder=5, edgecolors='white', lw=0.8)
    score_cca = alignment_score(z_bulk_cca, z_sc_cca)
    ax.text(0.05, 0.95, f'mixing score = {score_cca:.4f}',
            transform=ax.transAxes, fontsize=11, color=TEXT, va='top',
            bbox=dict(facecolor=CARD, edgecolor=GRID, alpha=0.85, pad=4))
    ax.set_title('CCA Alignment\n(maximises bulk–SC correlation)',
                 fontsize=13, fontweight='bold', pad=10, color=TEXT)
    ax.set_xlabel('CC1', color=MUTED); ax.set_ylabel('CC2', color=MUTED)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(alpha=0.3)

    # Panel 3: canonical correlations bar chart
    ax = axes[2]
    colors_bar = plt.cm.Blues(np.linspace(0.4, 1.0, len(corrs)))
    bars = ax.bar(range(len(corrs)), corrs, color=colors_bar, edgecolor='none', alpha=0.9)
    ax.set_xticks(range(len(corrs)))
    ax.set_xticklabels([f'CC{i+1}' for i in range(len(corrs))], color=MUTED)
    ax.set_ylabel('Canonical Correlation', color=MUTED)
    ax.set_ylim(0, 1)
    ax.axhline(1.0, color=MUTED, ls=':', lw=1)
    ax.set_title('Canonical Correlations per Component\n(1.0 = perfectly correlated)',
                 fontsize=13, fontweight='bold', pad=10, color=TEXT)
    for i, (bar, c) in enumerate(zip(bars, corrs)):
        ax.text(i, c + 0.02, f'{c:.3f}', ha='center', fontsize=9, color=TEXT, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--latent_dim', type=int,   default=64)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--beta_max',   type=float, default=1.0,
                        help='Max KL weight (anneal from 0 over warmup_frac of epochs)')
    parser.add_argument('--lam_max',    type=float, default=0.1,
                        help='Max adversarial weight')
    parser.add_argument('--warmup_frac',type=float, default=0.3,
                        help='Fraction of epochs for annealing warmup')
    parser.add_argument('--no_adv',     action='store_true',
                        help='Disable adversarial loss (ablation)')
    parser.add_argument('--sc_source', type=str, default=None,
                        help='Path to .h5ad file with individual cells. '
                             'If omitted, uses HCA pseudobulk .npz')
    parser.add_argument('--max_cells', type=int, default=50000,
                        help='Max SC cells to load (subsampled if larger, default 50k)')
    parser.add_argument('--mode', choices=['vae', 'cca', 'both'], default='both',
                        help='vae: deep model only  cca: linear only  both: run both')
    args = parser.parse_args()

    if args.no_adv:
        args.lam_max = 0.0
        print("Adversarial loss DISABLED (ablation mode)")

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────
    bulk_expr, bulk_names = load_gtex(GTEX_PATH)
    if args.sc_source:
        sc_expr, sc_names = load_h5ad_cells(args.sc_source, max_cells=args.max_cells)
    else:
        sc_expr, sc_names = load_hca(PSEUDO)

    bulk_shared, sc_shared, shared_genes = get_shared_genes(
        bulk_expr, bulk_names, sc_expr, sc_names)

    n_genes = bulk_shared.shape[1]
    print(f"\nShared gene space: {n_genes:,} genes")
    print(f"Bulk: {bulk_shared.shape}   SC: {sc_shared.shape}")

    # Scale to zero mean / unit std per gene (computed on bulk, applied to both)
    # This puts both modalities on comparable numerical ranges.
    scaler = StandardScaler()
    bulk_scaled = scaler.fit_transform(bulk_shared).astype(np.float32)
    sc_scaled   = scaler.transform(sc_shared).astype(np.float32)

    bulk_t = torch.from_numpy(bulk_scaled)
    sc_t   = torch.from_numpy(sc_scaled)

    # ── Naïve PCA baseline ──────────────────────────────────────────
    print("\nComputing naïve PCA baseline ...")
    z_bulk_pca_full, z_sc_pca_full = pca_baseline(bulk_scaled, sc_scaled, n_components=50)

    pca_score = alignment_score(z_bulk_pca_full, z_sc_pca_full)
    print(f"  Naïve PCA kNN mixing score: {pca_score:.4f}")

    # ── CCA alignment ────────────────────────────────────────────────
    # CCA requires n_samples_X == n_samples_Y and works on pseudobulk scale.
    # We aggregate the 803 GTEx donors into 100 pseudobulk groups so both
    # datasets are at the same "pseudobulk" resolution before comparing.
    if args.mode in ('cca', 'both'):
        print("\nPreparing pseudobulk representations for CCA ...")
        # Load raw GTEx for aggregation (before scaling)
        df_raw      = pd.read_csv(GTEX_PATH, sep='\t', skiprows=2, compression='gzip')
        expr_raw_np = df_raw.iloc[:, 2:].values.astype(np.float64).T  # (samples, genes)
        gene_names_raw = df_raw['Description'].values.astype(str)
        g_up = {n.upper(): i for i, n in enumerate(gene_names_raw)}
        sel  = np.array([g_up[g] for g in shared_genes if g in g_up])
        bulk_raw_shared = expr_raw_np[:, sel]

        # Aggregate GTEx 803→100 groups
        bulk_pb = aggregate_bulk_to_pseudobulk(bulk_raw_shared, shared_genes, n_groups=100)

        # SC pseudobulk: if we loaded h5ad cells, aggregate them per donor
        # (CCA needs donor-level aggregates, not individual cells)
        if args.sc_source:
            import anndata, scipy.sparse as sp_lib
            adata_cca = anndata.read_h5ad(args.sc_source)
            sc_sym_col = ('feature_name' if 'feature_name' in adata_cca.var.columns
                          else 'gene_symbols')
            gnames_cca = np.array(adata_cca.var[sc_sym_col].astype(str))
            sc_up = {n.upper(): i for i, n in enumerate(gnames_cca)}
            sc_sel_cca = np.array([sc_up[g] for g in shared_genes if g in sc_up])
            X_cca = adata_cca.X
            if sp_lib.issparse(X_cca): X_cca = X_cca.toarray()
            X_cca = X_cca[:, sc_sel_cca].astype(np.float64)
            donors_cca = adata_cca.obs['donor_id'].values
            unique_d   = np.unique(donors_cca)
            sc_pb = np.zeros((len(unique_d), len(shared_genes)))
            for i, d in enumerate(unique_d):
                mask = donors_cca == d
                sc_pb[i] = X_cca[mask].sum(axis=0)
            lib = sc_pb.sum(axis=1, keepdims=True)
            sc_pb = np.log2(sc_pb / lib * 1e6 + 1).astype(np.float32)
            print(f"  SC pseudobulk: {sc_pb.shape[0]} donors")
        else:
            sc_up = {n.upper(): i for i, n in enumerate(sc_names)}
            sc_sel = np.array([sc_up[g] for g in shared_genes if g in sc_up])
            sc_pb  = sc_expr[:, sc_sel]

        z_bulk_cca, z_sc_cca, corrs, _ = cca_alignment(bulk_pb, sc_pb)
        fig_cca(z_bulk_cca, z_sc_cca, z_bulk_pca_full, z_sc_pca_full, corrs,
                os.path.join(OUT_DIR, 'cross_modality_cca.png'))

    if args.mode == 'cca':
        print("\nDone (CCA only mode).")
        sys.exit(0)

    # ── Build and train model ────────────────────────────────────────
    model = CrossModalityVAE(
        n_genes    = n_genes,
        latent_dim = args.latent_dim,
        enc_hidden = (1024, 512),
        dec_hidden = (512, 1024),
        disc_hidden= (128, 64),
        dropout    = 0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    history = train(
        model, bulk_t, sc_t,
        epochs      = args.epochs,
        lr          = args.lr,
        batch_size  = args.batch_size,
        beta_max    = args.beta_max,
        lam_max     = args.lam_max if not args.no_adv else 0.0,
        warmup_frac = args.warmup_frac,
        device      = device,
    )

    # ── Extract latent codes ─────────────────────────────────────────
    print("\nExtracting latent representations ...")
    z_bulk_vae, z_sc_vae = get_latent(model, bulk_t, sc_t, device)

    vae_score = alignment_score(z_bulk_vae, z_sc_vae)
    print(f"  VAE latent kNN mixing score: {vae_score:.4f}")
    print(f"  Baseline kNN mixing score:   {pca_score:.4f}")
    print(f"  Improvement: {(vae_score - pca_score):.4f}")

    # ── Save model ───────────────────────────────────────────────────
    model_path = os.path.join(OUT_DIR, 'cross_modality_vae.pt')
    torch.save({
        'model_state': model.state_dict(),
        'shared_genes': shared_genes,
        'scaler_mean': scaler.mean_,
        'scaler_std': scaler.scale_,
        'args': vars(args),
        'history': history,
        'vae_score': vae_score,
        'pca_score': pca_score,
    }, model_path)
    print(f"Saved model: {model_path}")

    # ── Visualise ────────────────────────────────────────────────────
    fig_results(
        z_bulk_vae, z_sc_vae,
        z_bulk_pca_full, z_sc_pca_full,
        bulk_scaled, sc_scaled,
        history,
        os.path.join(OUT_DIR, 'cross_modality_latent_space.png'),
    )

    print("\nDone.")
