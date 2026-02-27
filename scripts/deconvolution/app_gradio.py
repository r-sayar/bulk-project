"""
Bulk RNA-seq Deconvolution Explorer — Gradio Demo
==================================================

A self-contained demo that:
  1. Simulates single-cell reference data (or loads real data if available)
  2. Builds a signature matrix
  3. Creates synthetic bulk mixtures with user-controlled proportions
  4. Runs NNLS, NMF, and Neural W-CLS deconvolution
  5. Shows interactive comparison plots

Usage:
  pip install gradio torch numpy scipy scikit-learn matplotlib
  python app_gradio.py

For a quick demo without real scRNA-seq data, the app generates
synthetic reference profiles so it runs anywhere.
"""

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import nnls
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
import io, warnings
warnings.filterwarnings("ignore")

# ─── Styling ─────────────────────────────────────────────────────────
COLORS = {
    "true":    "#1b9e77",
    "nnls":    "#d95f02",
    "nmf":     "#7570b3",
    "neural":  "#e7298a",
    "bg":      "#fafafa",
    "grid":    "#e0e0e0",
}
CELL_TYPE_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.facecolor": COLORS["bg"],
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": COLORS["grid"],
})


# ═══════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC REFERENCE BUILDER (demo-ready, no real data needed)
# ═══════════════════════════════════════════════════════════════════════
def build_synthetic_reference(n_genes=2000, K=5, n_cells_per_type=200, seed=42):
    """
    Create a synthetic scRNA-seq-like reference with K cell types.
    Each cell type has ~50 marker genes with elevated expression.
    """
    rng = np.random.default_rng(seed)

    # Base expression (low, noisy)
    S = rng.exponential(0.5, size=(n_genes, K)).astype(np.float32)

    # Add marker genes per cell type
    markers_per_type = n_genes // K
    for k in range(K):
        start = k * markers_per_type
        end = start + markers_per_type // 4  # top 25% of allocated genes are markers
        S[start:end, k] += rng.exponential(3.0, size=(end - start,))

    # Generate single cells by adding Poisson noise around signature
    n_cells = n_cells_per_type * K
    labels = np.repeat(np.arange(K), n_cells_per_type)
    X = np.zeros((n_genes, n_cells), dtype=np.float32)
    for i in range(n_cells):
        k = labels[i]
        rate = S[:, k] * rng.uniform(0.8, 1.2)  # cell-level variation
        X[:, i] = rng.poisson(np.maximum(rate, 0.01))

    # Normalize to CPM-like
    lib_sizes = X.sum(axis=0, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1
    X_norm = X / lib_sizes * 1e4

    # Recompute signature from normalized data
    S_clean = np.zeros((n_genes, K), dtype=np.float32)
    for k in range(K):
        mask = labels == k
        S_clean[:, k] = X_norm[:, mask].mean(axis=1)

    return X_norm, S_clean, labels


# ═══════════════════════════════════════════════════════════════════════
# 2. BULK SIMULATION
# ═══════════════════════════════════════════════════════════════════════
def simulate_bulks(X_norm, labels, K, n_samples=800, cells_per_bulk=200,
                   dirichlet_alpha=2.0, seed=42):
    rng = np.random.default_rng(seed)
    pool_by_k = [np.where(labels == k)[0] for k in range(K)]

    n_genes = X_norm.shape[0]
    B = np.zeros((n_genes, n_samples), dtype=np.float32)
    P = np.zeros((n_samples, K), dtype=np.float32)

    for i in range(n_samples):
        props = rng.dirichlet(np.ones(K) * dirichlet_alpha)
        counts = rng.multinomial(cells_per_bulk, props)
        P[i, :] = counts / counts.sum()

        cols = []
        for k in range(K):
            if counts[k] > 0:
                pick = rng.choice(pool_by_k[k], size=counts[k], replace=True)
                cols.extend(pick.tolist())

        B[:, i] = X_norm[:, cols].mean(axis=1)

    return B, P


# ═══════════════════════════════════════════════════════════════════════
# 3. DECONVOLUTION METHODS
# ═══════════════════════════════════════════════════════════════════════
def deconvolve_nnls(S, b):
    p, _ = nnls(S, b)
    return p / p.sum() if p.sum() > 0 else p

def deconvolve_nmf(B, K, seed=42):
    model = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=2000)
    W = model.fit_transform(B.T)
    W = np.maximum(W, 0)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return W / row_sums


# ─── Neural W-CLS v3 ────────────────────────────────────────────────
class WeightNet(nn.Module):
    def __init__(self, n_genes, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, x_log):
        logw = self.net(x_log)
        logw = logw - logw.mean(dim=1, keepdim=True)
        logw = torch.clamp(logw, -2.0, 2.0)
        return torch.exp(logw), logw

class WRidgeSolver(nn.Module):
    def __init__(self, S_tensor, lam_init=1e-2):
        super().__init__()
        self.register_buffer("S", S_tensor)
        self.log_lam = nn.Parameter(torch.tensor(np.log(lam_init), dtype=torch.float32))
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, b, w):
        S, K = self.S, self.S.shape[1]
        lam = F.softplus(self.log_lam) + 1e-6
        temp = F.softplus(self.log_temp) + 0.1

        Sw = S.unsqueeze(0) * w.unsqueeze(-1)
        A = torch.matmul(Sw.transpose(1, 2), Sw)
        A = A + lam * torch.eye(K, device=b.device).unsqueeze(0)
        rhs = torch.matmul(Sw.transpose(1, 2), (w * b).unsqueeze(-1))
        p_raw = torch.linalg.solve(A, rhs).squeeze(-1)
        return F.softmax(p_raw / temp, dim=1), p_raw

class NeuralWCLS(nn.Module):
    def __init__(self, S_tensor, hidden=64, dropout=0.2, lam_init=1e-2):
        super().__init__()
        self.weightnet = WeightNet(S_tensor.shape[0], hidden, dropout)
        self.solver = WRidgeSolver(S_tensor, lam_init)

    def forward(self, b):
        x_log = torch.log1p(torch.clamp(b, min=0))
        w, logw = self.weightnet(x_log)
        p, p_raw = self.solver(b, w)
        return p, w, logw, p_raw


def train_neural_wcls(S, B, P_true, epochs=300, patience=30, progress_cb=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = B.T.astype(np.float32)
    Y = P_true.astype(np.float32)

    idx = np.arange(X.shape[0])
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42)
    idx_tr, idx_val = train_test_split(idx_tr, test_size=0.15, random_state=42)

    X_tr_t = torch.from_numpy(X[idx_tr]).to(device)
    Y_tr_t = torch.from_numpy(Y[idx_tr]).to(device)
    X_val_t = torch.from_numpy(X[idx_val]).to(device)
    Y_val_t = torch.from_numpy(Y[idx_val]).to(device)
    X_te_t = torch.from_numpy(X[idx_te]).to(device)

    loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=64, shuffle=True)
    S_t = torch.from_numpy(S.astype(np.float32)).to(device)

    model = NeuralWCLS(S_t).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse = nn.MSELoss()

    best_val, best_state, wait = float("inf"), None, 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            p, w, logw, p_raw = model(xb)

            loss = (mse(p, yb)
                    + 0.2 * torch.sum(yb * (torch.log(yb + 1e-8) - torch.log(p + 1e-8)), 1).mean()
                    + 0.05 * torch.sum(p * torch.log(p + 1e-8), 1).mean()  # entropy
                    + 1e-4 * (logw ** 2).mean())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = mse(model(X_val_t)[0], Y_val_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if progress_cb and ep % 10 == 0:
            progress_cb(ep / epochs, f"Epoch {ep}/{epochs} | val MSE: {val_loss:.5f}")

        if wait >= patience:
            break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        P_te = model(X_te_t)[0].cpu().numpy()

    return model, P_te, Y[idx_te], X[idx_te], device


# ═══════════════════════════════════════════════════════════════════════
# 4. PLOTTING
# ═══════════════════════════════════════════════════════════════════════
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def pearson(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def plot_deconv_result(true_props, nnls_props, neural_props, K):
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.35)

    ct_labels = [f"C{k}" for k in range(K)]
    x = np.arange(K)
    w = 0.28

    # Panel 1: Bar chart comparison
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(x - w, true_props, w, color=COLORS["true"], label="True", edgecolor="white", linewidth=0.8)
    ax1.bar(x, nnls_props, w, color=COLORS["nnls"], label="NNLS", edgecolor="white", linewidth=0.8)
    ax1.bar(x + w, neural_props, w, color=COLORS["neural"], label="Neural W-CLS", edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ct_labels)
    ax1.set_ylim(0, min(1.05, max(true_props.max(), nnls_props.max(), neural_props.max()) * 1.15))
    ax1.set_ylabel("Proportion")
    ax1.set_title("Single Sample Deconvolution", fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")

    # Panel 2: Residuals
    ax2 = fig.add_subplot(gs[1])
    resid_nnls = nnls_props - true_props
    resid_neural = neural_props - true_props
    ax2.barh(x - 0.15, resid_nnls, 0.3, color=COLORS["nnls"], label="NNLS error", alpha=0.8)
    ax2.barh(x + 0.15, resid_neural, 0.3, color=COLORS["neural"], label="Neural error", alpha=0.8)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(ct_labels)
    ax2.set_xlabel("Prediction − True")
    ax2.set_title("Residuals", fontweight="bold")
    ax2.legend(fontsize=9)

    # Panel 3: Pie charts
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    ax3.set_title("Composition Overview", fontweight="bold", pad=15)
    sub_axes = [
        fig.add_axes([0.68, 0.25, 0.14, 0.55]),  # True
        fig.add_axes([0.84, 0.25, 0.14, 0.55]),  # Neural
    ]
    for ax_pie, data, title in zip(sub_axes, [true_props, neural_props], ["True", "Neural"]):
        wedges, _ = ax_pie.pie(data, colors=CELL_TYPE_COLORS[:K], startangle=90)
        ax_pie.set_title(title, fontsize=10, fontweight="bold")
        for w_obj in wedges:
            w_obj.set_edgecolor("white")
            w_obj.set_linewidth(1.5)

    plt.suptitle("Bulk RNA-seq Deconvolution Results", fontsize=14, fontweight="bold", y=1.02)
    return fig


def plot_test_scatter(P_neural_te, Y_te, K):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(Y_te.ravel(), P_neural_te.ravel(), s=15, alpha=0.5, color=COLORS["neural"])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    r = pearson(P_neural_te, Y_te)
    ax.set_xlabel("True proportion")
    ax.set_ylabel("Predicted proportion")
    ax.set_title(f"All test samples  (r = {r:.3f}, RMSE = {rmse(P_neural_te, Y_te):.4f})",
                 fontweight="bold")

    ax = axes[1]
    for k in range(K):
        ax.scatter(Y_te[:, k], P_neural_te[:, k], s=25, alpha=0.6,
                   color=CELL_TYPE_COLORS[k], label=f"C{k}", edgecolors="white", linewidth=0.3)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("True proportion")
    ax.set_ylabel("Predicted proportion")
    ax.set_title("Per cell-type scatter", fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════
# 5. GRADIO APP
# ═══════════════════════════════════════════════════════════════════════

# Global state (populated on first run)
STATE = {}

def initialize_model(n_genes, K, n_bulks, dirichlet_alpha, progress=gr.Progress()):
    """Build reference, simulate bulks, train the neural model."""
    progress(0.05, "Building synthetic reference...")
    X_norm, S, labels = build_synthetic_reference(n_genes=n_genes, K=K)

    progress(0.15, f"Simulating {n_bulks} bulk mixtures...")
    B, P_true = simulate_bulks(X_norm, labels, K, n_samples=n_bulks,
                               dirichlet_alpha=dirichlet_alpha)

    progress(0.25, "Training Neural W-CLS v3...")
    model, P_te, Y_te, X_te, device = train_neural_wcls(
        S, B, P_true, epochs=300, patience=30,
        progress_cb=lambda frac, msg: progress(0.25 + frac * 0.65, msg)
    )

    # Store everything
    STATE.update({
        "model": model, "S": S, "B": B, "P_true": P_true,
        "X_norm": X_norm, "labels": labels, "K": K,
        "P_te": P_te, "Y_te": Y_te, "X_te": X_te, "device": device,
    })

    # Generate test scatter plot
    progress(0.95, "Generating evaluation plots...")
    fig = plot_test_scatter(P_te, Y_te, K)
    r = pearson(P_te, Y_te)
    rm = rmse(P_te, Y_te)

    stats = (f"**Model trained!**\n\n"
             f"- Reference: {n_genes} genes × {K} cell types\n"
             f"- Training bulks: {n_bulks} (α={dirichlet_alpha})\n"
             f"- Test RMSE: **{rm:.4f}** | Pearson r: **{r:.3f}**\n"
             f"- Prediction diversity: {', '.join(f'C{k}: {P_te[:,k].std():.3f}' for k in range(K))}")

    return fig, stats


def deconvolve_single(c0, c1, c2, c3, c4):
    """Create a bulk with given proportions and deconvolve it."""
    if "model" not in STATE:
        return None, "⚠️ Please train the model first (click 'Train Model')."

    K = STATE["K"]
    S = STATE["S"]
    X_norm = STATE["X_norm"]
    labels = STATE["labels"]
    model = STATE["model"]
    device = STATE["device"]

    # Build proportions from sliders
    raw = np.array([c0, c1, c2, c3, c4][:K], dtype=np.float32)
    if raw.sum() == 0:
        raw = np.ones(K) / K
    true_props = raw / raw.sum()

    # Simulate a single bulk
    rng = np.random.default_rng()
    cells_per_bulk = 300
    counts = rng.multinomial(cells_per_bulk, true_props)
    pool_by_k = [np.where(labels == k)[0] for k in range(K)]

    cols = []
    for k in range(K):
        if counts[k] > 0:
            cols.extend(rng.choice(pool_by_k[k], size=counts[k], replace=True).tolist())
    bulk = X_norm[:, cols].mean(axis=1)

    # NNLS
    nnls_props = deconvolve_nnls(S, bulk)

    # Neural W-CLS
    model.eval()
    with torch.no_grad():
        b_t = torch.from_numpy(bulk.reshape(1, -1).astype(np.float32)).to(device)
        neural_props = model(b_t)[0].cpu().numpy().ravel()

    # Plot
    fig = plot_deconv_result(true_props, nnls_props, neural_props, K)

    # Metrics text
    info = (f"**True:**  {', '.join(f'C{k}={true_props[k]:.3f}' for k in range(K))}\n\n"
            f"**NNLS:**  {', '.join(f'C{k}={nnls_props[k]:.3f}' for k in range(K))} "
            f"| RMSE={rmse(nnls_props, true_props):.4f}\n\n"
            f"**Neural:** {', '.join(f'C{k}={neural_props[k]:.3f}' for k in range(K))} "
            f"| RMSE={rmse(neural_props, true_props):.4f}")

    return fig, info


def random_proportions():
    """Generate random Dirichlet proportions for the sliders."""
    props = np.random.dirichlet(np.ones(5) * 2.0)
    return [float(round(p, 3)) for p in props]


# ─── Build UI ────────────────────────────────────────────────────────
with gr.Blocks(
    title="Bulk RNA-seq Deconvolution Explorer",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="pink",
        font=gr.themes.GoogleFont("Source Sans Pro"),
    ),
    css="""
    .main-title { text-align: center; margin-bottom: 0.5em; }
    .method-badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
                    font-size: 0.85em; font-weight: 600; }
    """
) as demo:

    gr.Markdown("""
    # 🧬 Bulk RNA-seq Deconvolution Explorer
    **Compare NNLS vs Neural Weighted Constrained Least Squares (W-CLS v3)**

    This demo trains a neural deconvolution model on synthetic bulk RNA-seq data,
    then lets you dial in cell-type proportions and see how each method performs.
    """)

    with gr.Tab("🔧 Train Model"):
        gr.Markdown("### Configure and train the deconvolution model")

        with gr.Row():
            n_genes = gr.Slider(500, 5000, value=2000, step=500, label="Number of genes")
            n_types = gr.Slider(3, 8, value=5, step=1, label="Number of cell types (K)")

        with gr.Row():
            n_bulks = gr.Slider(200, 2000, value=800, step=100, label="Number of synthetic bulks")
            dir_alpha = gr.Slider(0.5, 5.0, value=2.0, step=0.5, label="Dirichlet α (higher = more uniform)")

        train_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")

        train_stats = gr.Markdown("*Click 'Train Model' to begin...*")
        train_plot = gr.Plot(label="Test Set Evaluation")

        train_btn.click(
            initialize_model,
            inputs=[n_genes, n_types, n_bulks, dir_alpha],
            outputs=[train_plot, train_stats],
        )

    with gr.Tab("🔬 Deconvolve"):
        gr.Markdown("### Set true cell-type proportions and compare methods")
        gr.Markdown("*Proportions are auto-normalized to sum to 1.*")

        with gr.Row():
            s0 = gr.Slider(0, 1, value=0.20, step=0.01, label="C0")
            s1 = gr.Slider(0, 1, value=0.20, step=0.01, label="C1")
            s2 = gr.Slider(0, 1, value=0.20, step=0.01, label="C2")
            s3 = gr.Slider(0, 1, value=0.20, step=0.01, label="C3")
            s4 = gr.Slider(0, 1, value=0.20, step=0.01, label="C4")

        with gr.Row():
            run_btn = gr.Button("🧪 Deconvolve", variant="primary")
            rand_btn = gr.Button("🎲 Random Proportions", variant="secondary")

        result_info = gr.Markdown()
        result_plot = gr.Plot(label="Deconvolution Comparison")

        run_btn.click(
            deconvolve_single,
            inputs=[s0, s1, s2, s3, s4],
            outputs=[result_plot, result_info],
        )

        rand_btn.click(
            random_proportions,
            outputs=[s0, s1, s2, s3, s4],
        )

    with gr.Tab("📖 About"):
        gr.Markdown("""
        ## How it works

        **The problem**: Bulk RNA-seq measures average gene expression across millions
        of cells. We want to infer what fraction of cells belongs to each type.

        **The math**: Given a signature matrix **S** (genes × cell types) and a bulk
        expression vector **b**, find proportions **p** such that **Sp ≈ b** with p ≥ 0, Σp = 1.

        ### Methods compared

        | Method | Type | Key idea |
        |--------|------|----------|
        | **NNLS** | Classical | Non-negative least squares: min ‖Sp − b‖² s.t. p ≥ 0 |
        | **Neural W-CLS** | Deep learning | Learn per-gene importance weights via a neural network, then solve a weighted ridge regression in closed form |

        ### What was wrong with v2 (the collapsing model)

        The original Neural W-CLS collapsed to predicting only C3 because:
        1. **Too few training samples** (80 bulks, 60 for training) for a large network
        2. **softplus + renormalize** simplex projection amplifies any dominance
        3. **No entropy regularization** — nothing prevented collapse

        ### v3 fixes
        - **10× more synthetic bulks** (800 default)
        - **Smaller weight network** (single hidden layer, 64 units)
        - **Softmax with learned temperature** instead of softplus
        - **Entropy regularization** to prevent collapse
        - **Cosine LR schedule + early stopping**
        - **Higher Dirichlet α** for more balanced training mixtures
        """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
