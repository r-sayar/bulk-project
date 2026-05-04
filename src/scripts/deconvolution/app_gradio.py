"""
Bulk RNA-seq Deconvolution Explorer — Gradio Demo
==================================================

A lightweight Gradio app for exploring NNLS-based deconvolution with:
  1. A ready-to-use synthetic reference
  2. Manual demo sliders for synthetic mixtures
  3. Local file upload from the desktop for bulk profiles

Usage:
  pip install gradio numpy scipy matplotlib pandas
  python app_gradio.py
"""

import gzip
import os
import warnings

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.optimize import nnls

warnings.filterwarnings("ignore")


COLORS = {
    "pred": "#d95f02",
    "bg": "#fafafa",
    "grid": "#e0e0e0",
}

CELL_TYPE_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]

DEFAULT_GENES = 2000
DEFAULT_CELL_TYPES = 5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.facecolor": COLORS["bg"],
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": COLORS["grid"],
})


def build_synthetic_reference(n_genes=DEFAULT_GENES, K=DEFAULT_CELL_TYPES,
                              n_cells_per_type=200, seed=42):
    """Create a synthetic scRNA-seq-like reference with K cell types."""
    rng = np.random.default_rng(seed)

    signature = rng.exponential(0.5, size=(n_genes, K)).astype(np.float32)
    markers_per_type = n_genes // K
    for k in range(K):
        start = k * markers_per_type
        end = start + markers_per_type // 4
        signature[start:end, k] += rng.exponential(3.0, size=(end - start,))

    n_cells = n_cells_per_type * K
    labels = np.repeat(np.arange(K), n_cells_per_type)
    cells = np.zeros((n_genes, n_cells), dtype=np.float32)
    for i in range(n_cells):
        k = labels[i]
        rate = signature[:, k] * rng.uniform(0.8, 1.2)
        cells[:, i] = rng.poisson(np.maximum(rate, 0.01))

    lib_sizes = cells.sum(axis=0, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1
    X_norm = cells / lib_sizes * 1e4

    S_clean = np.zeros((n_genes, K), dtype=np.float32)
    for k in range(K):
        S_clean[:, k] = X_norm[:, labels == k].mean(axis=1)

    gene_names = np.array([f"G{i}" for i in range(n_genes)])
    return X_norm, S_clean, labels, gene_names


def ensure_reference_ready():
    """Initialize the synthetic reference once and keep it in memory."""
    if "S" not in STATE:
        X_norm, S, labels, gene_names = build_synthetic_reference()
        STATE.update({
            "X_norm": X_norm,
            "S": S,
            "labels": labels,
            "gene_names": gene_names,
            "K": DEFAULT_CELL_TYPES,
        })


def deconvolve_nnls(signature, bulk_vector):
    """Solve min ||Sp - b||^2 such that p >= 0, then normalize."""
    props, _ = nnls(signature, bulk_vector)
    if props.sum() <= 0:
        return props
    return props / props.sum()


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def infer_separator(filepath):
    open_fn = open if not filepath.endswith(".gz") else gzip.open
    with open_fn(filepath, "rt") as handle:
        first_line = handle.readline()
    return "\t" if "\t" in first_line else ","


def load_bulk_profile(filepath, expected_genes):
    """
    Load a single bulk profile from a local CSV/TSV/TXT file.

    Supported shapes:
      - expected_genes rows × 1 numeric column
      - 1 row × expected_genes numeric columns
      - matrix with multiple samples; first numeric sample is used
    """
    if not filepath:
        raise ValueError("No file was uploaded.")

    sep = infer_separator(filepath)
    compression = "gzip" if filepath.endswith(".gz") else None
    df = pd.read_csv(filepath, sep=sep, compression=compression)

    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("The uploaded file does not contain any numeric columns.")

    if numeric.shape[0] == expected_genes:
        vector = numeric.iloc[:, 0].to_numpy(dtype=np.float32)
        sample_name = numeric.columns[0]
    elif numeric.shape[1] == expected_genes:
        vector = numeric.iloc[0].to_numpy(dtype=np.float32)
        sample_name = str(df.index[0]) if len(df.index) else "uploaded_sample"
    else:
        raise ValueError(
            f"Uploaded file shape does not match the demo reference. "
            f"Expected {expected_genes} genes in either rows or columns, "
            f"got {numeric.shape[0]} rows × {numeric.shape[1]} numeric columns."
        )

    if np.allclose(vector.sum(), 0):
        raise ValueError("The uploaded sample appears to contain only zeros.")

    return np.maximum(vector, 0), str(sample_name)


def plot_uploaded_result(pred_props, K, sample_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f"C{k}" for k in range(K)]
    x = np.arange(K)

    axes[0].bar(x, pred_props, color=CELL_TYPE_COLORS[:K], edgecolor="white", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, max(pred_props.max() * 1.2, 0.1))
    axes[0].set_ylabel("Proportion")
    axes[0].set_title(f"Predicted Composition for {sample_name}", fontweight="bold")

    axes[1].pie(
        pred_props,
        labels=labels,
        autopct="%1.1f%%",
        colors=CELL_TYPE_COLORS[:K],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.4},
    )
    axes[1].set_title("Composition Overview", fontweight="bold")

    plt.tight_layout()
    return fig


def deconvolve_input(uploaded_file):
    ensure_reference_ready()

    K = STATE["K"]
    S = STATE["S"]
    if not uploaded_file:
        return None, "Upload a bulk expression file to run prediction."

    bulk, sample_name = load_bulk_profile(uploaded_file, S.shape[0])
    pred_props = deconvolve_nnls(S, bulk)
    fig = plot_uploaded_result(pred_props, K, os.path.basename(sample_name))
    info = (
        f"**Uploaded sample:** `{os.path.basename(uploaded_file)}`\n\n"
        f"- Predictor: **NNLS**\n"
        f"- Numeric profile used: `{sample_name}`\n"
        f"- Genes expected by current reference: **{S.shape[0]}**\n"
        f"- Predicted proportions: {', '.join(f'C{k}={pred_props[k]:.3f}' for k in range(K))}\n\n"
        f"*Note: this still needs a real saved W-CLS checkpoint if you want W-CLS inference instead.*"
    )
    return fig, info


def app_status():
    ensure_reference_ready()
    return (
        f"**Reference ready**\n\n"
        f"- Synthetic reference loaded automatically\n"
        f"- Genes: **{STATE['S'].shape[0]}**\n"
        f"- Cell types: **{STATE['K']}**\n"
        f"- Local upload is enabled below\n"
        f"- Manual proportion sliders removed"
    )


STATE = {}

with gr.Blocks(
    title="Bulk RNA-seq Deconvolution Explorer",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="pink",
        font=gr.themes.GoogleFont("Source Sans Pro"),
    ),
) as demo:
    gr.Markdown("""
    # Bulk RNA-seq Deconvolution Explorer
    Upload a bulk expression file from your desktop and predict cell-type proportions.
    """)

    status = gr.Markdown()

    with gr.Tab("Deconvolve"):
        gr.Markdown("### Upload a bulk expression file")
        file_input = gr.File(
            label="Select a bulk expression file from your desktop",
            type="filepath",
            file_types=[".csv", ".tsv", ".txt", ".gz"],
        )
        gr.Markdown(
            "*Accepted formats: CSV/TSV/TXT. The app predicts from the uploaded file only.*"
        )

        with gr.Row():
            run_btn = gr.Button("Deconvolve", variant="primary")

        result_info = gr.Markdown()
        result_plot = gr.Plot(label="Deconvolution Result")

        run_btn.click(
            deconvolve_input,
            inputs=[file_input],
            outputs=[result_plot, result_info],
        )

    with gr.Tab("About"):
        gr.Markdown("""
        ## How it works

        This UI removes the model-training workflow and predicts directly from an
        uploaded bulk profile.

        The current version builds a synthetic reference once, then solves:
        **Sp ≈ b** subject to **p ≥ 0**, where:
        - **S** is the signature matrix
        - **b** is the bulk expression vector
        - **p** is the estimated cell-type composition

        Uploaded files must currently match the reference shape expected by the app.
        If you want `W-CLS`, the app needs a saved model checkpoint to load.
        """)

    demo.load(app_status, outputs=status)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
