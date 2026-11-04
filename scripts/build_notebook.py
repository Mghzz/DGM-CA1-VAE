"""Build and execute the standalone Jupyter Notebook for Question 2, Part 2."""
import os
import sys
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.7"
        }
    }
    
    cells = []
    
    # Header
    cells.append(nbf.v4.new_markdown_cell("""# Deep Generative Models (DGM) — Homework 1
## Question 2, Part 2: Variational Autoencoder (VAE) Implementation & Experimental Study
**Course**: Deep Generative Models  
**Institution**: University of Tehran, Department of Electrical and Computer Engineering  
**Instructor**: Dr. Mostafa Tavasolipour  
**Hardware & Environment**: NVIDIA GeForce RTX 5060 Laptop GPU, PyTorch with CUDA, Python 3.14  

---

### Overview
This notebook presents the end-to-end implementation, verification, and evaluation of **four VAE architectures** trained on a 20-dimensional synthetic dataset generated from 8 clusters in a 2D latent space:
1. **Model A**: Standard Isotropic Gaussian Prior $\\mathcal{N}(0, I_2)$ with closed-form analytical KL divergence.
2. **Model B**: Fixed 8-component Gaussian Mixture Model (GMM) prior with component means on a circle of radius $r=2.0$ and $\\sigma=0.40$.
3. **Model C**: Trainable 8-component GMM prior with learnable component means $\\mu_k$ and diagonal variances $\\sigma_k^2$.
4. **Model D**: Trainable 4-component GMM prior with learnable component means $\\mu_k$ and diagonal variances $\\sigma_k^2$.
"""))

    # Imports and device
    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from configs.config import ProjectConfig
from src.data.dataset_loader import get_dataloaders, load_dataset
from src.models import VAE, build_vae
from src.training.checkpointing import load_checkpoint
from src.evaluation.metrics import (
    compute_reconstruction_mse,
    compute_unbiased_rbf_mmd2,
    compute_bandwidth_median_heuristic,
    evaluate_cluster_coverage
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Active Device: {device}")
if device.type == "cuda":
    print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")
"""))

    # Stage 1: Data exploration
    cells.append(nbf.v4.new_markdown_cell("""## Stage 1: Dataset Generation & 2D Exploration
The dataset comprises 9,600 observations in $\\mathbb{R}^{20}$ generated from 8 circular clusters in an underlying 2D source space $s$, mapped through nonlinear trigonometric and polynomial features, perturbed with Gaussian noise ($0.07$), and split into:
- **Train Split**: 70% (6,720 samples)
- **Validation Split**: 15% (1,440 samples)
- **Test Split**: 15% (1,440 samples)
Features are standardized using the training set mean and standard deviation.

> **Generation & Baseline Verification Commands**:
> ```bash
> python dataset.py
> python -m src.data.verify_dataset
> ```
"""))

    cells.append(nbf.v4.new_code_cell("""splits = load_dataset()
print("Dataset Splitting Summary:")
print(f"  - x_train shape: {splits.x_train.shape}, y_train shape: {splits.y_train.shape}")
print(f"  - x_val shape:   {splits.x_val.shape}, y_val shape:   {splits.y_val.shape}")
print(f"  - x_test shape:  {splits.x_test.shape}, y_test shape:  {splits.y_test.shape}")

print("\\nVisualizing Ground Truth 2D Source Coordinates and Training Set PCA Projection:")
fig_dir = Path("results/figures")
display(Image(filename=str(fig_dir / "source_2d_scatter.png"), width=700))
display(Image(filename=str(fig_dir / "train_pca_2d_scatter.png"), width=700))
"""))

    # Stage 2: Architecture
    cells.append(nbf.v4.new_markdown_cell("""## Stage 2: VAE Architecture & Loss Formulations
- **Encoder**: $20 \\to 128 \\to 128 \\to 2$, with ReLU activations, outputting mean $\\mu_\\phi(x)$ and log-variance $\\log\\sigma_\\phi^2(x)$.
- **Decoder**: $2 \\to 128 \\to 128 \\to 20$, with ReLU activations, outputting the reconstructed mean $\\mu_\\theta(z)$.
- **Observation Model**: Isotropic Gaussian likelihood with fixed $\\sigma_x = 0.30$:
$$\\mathcal{L}_{\\text{recon}}(x) = \\frac{1}{2\\sigma_x^2} \\|x - \\mu_\\theta(z)\\|^2 + \\frac{D}{2}\\log(2\\pi\\sigma_x^2)$$
For $D=20, \\sigma_x=0.30$, the scaling factor is $\\frac{1}{2 \\times 0.09} = \\frac{1}{0.18} \\approx 5.5556$, and the constant is $\\approx -5.7006$.
- **KL Divergence**:
  - **Model A**: Analytical closed-form Gaussian KL:
  $$D_{\\text{KL}}(q_\\phi(z|x) \\parallel \\mathcal{N}(0, I)) = -\\frac{1}{2} \\sum_{j=1}^2 \\left(1 + \\log\\sigma_j^2 - \\mu_j^2 - \\sigma_j^2\\right)$$
  - **Models B, C, D**: Single-sample Monte Carlo KL with numerically stable `logsumexp`:
  $$D_{\\text{KL}} \\approx \\log q_\\phi(z|x) - \\log p(z), \\quad z \\sim q_\\phi(z|x)$$
"""))

    # Stage 3: Loading Checkpoints
    cells.append(nbf.v4.new_markdown_cell("""## Stage 3 & 4: Model Checkpoints & Training Dynamics
All models were trained on `cuda:0` using Adam optimizer (learning rate $10^{-3}$, batch size 64) with early stopping (patience 15 epochs on validation total loss) over a maximum of 100 epochs.
Let us restore the best checkpoints and inspect their parameters.

> **Training Execution Command**:
> ```bash
> python -m src.training.run_experiments --device cuda:0
> ```
"""))

    cells.append(nbf.v4.new_code_cell("""models = {}
config = ProjectConfig()
for m_type in ["A", "B", "C", "D"]:
    ckpt_path = Path("checkpoints") / f"model_{m_type}_best.pt"
    model = build_vae(m_type, config=config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    models[m_type] = model
    epoch = ckpt.get("epoch", ckpt.get("metadata", {}).get("best_epoch", "N/A"))
    best_loss = ckpt.get("val_loss", ckpt.get("metadata", {}).get("best_val_loss", 0.0))
    print(f"Loaded Model {m_type}: Best Epoch = {epoch}, Val Loss = {best_loss:.4f}")

# Inspect learned prior parameters for Models C and D
print("\\n--- Model C Learned Prior Parameters (8 Components) ---")
print("Learned Component Centers (2D):\\n", models["C"].prior.centers.detach().cpu().numpy())
print("Learned Component Stds (2D):\\n", torch.exp(models["C"].prior.log_stds).detach().cpu().numpy())

print("\\n--- Model D Learned Prior Parameters (4 Components) ---")
print("Learned Component Centers (2D):\\n", models["D"].prior.centers.detach().cpu().numpy())
print("Learned Component Stds (2D):\\n", torch.exp(models["D"].prior.log_stds).detach().cpu().numpy())
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Training & Validation Learning Curves
The plot below illustrates the training and validation trajectories across all 4 models:
1. **Total ELBO Loss**
2. **Reconstruction Loss (NLL)**
3. **KL Divergence**
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=str(fig_dir / "learning_curves_all_models.png"), width=900))
"""))

    # Stage 5: Evaluation
    cells.append(nbf.v4.new_markdown_cell("""## Stage 5: Evaluation & Comparative Analysis

> **Official Evaluation Execution Command**:
> ```bash
> python -m src.evaluation.run_evaluation --device cuda:0 --seed 42
> ```

### 1. Test Reconstruction Error
Reconstruction error is evaluated strictly on the unseen test set ($N=1,440$) using the deterministic posterior mean $\\mu_\\phi(x)$ passed to the decoder $\\mu_\\theta(\\mu_\\phi(x))$:
- **MSE Per Element**: $\\frac{1}{N \\cdot D} \\sum_{i=1}^N \\|x^{(i)} - \\hat{x}^{(i)}\\|^2$
- **Mean Squared $L_2$ Per Sample**: $\\frac{1}{N} \\sum_{i=1}^N \\|x^{(i)} - \\hat{x}^{(i)}\\|^2 = D \\times \\text{MSE}_{\\text{per\\_element}}$
"""))

    cells.append(nbf.v4.new_code_cell("""metrics_dir = Path("results/metrics")
eval_df = pd.read_csv(metrics_dir / "evaluation_summary.csv")

recon_table = eval_df[["model_type", "checkpoint_epoch", "mse_per_element", "mean_squared_l2_per_sample"]].copy()
recon_table.columns = ["Model", "Best Epoch", "MSE Per Element", "Mean Squared L2 Per Sample"]
display(recon_table.style.format({
    "MSE Per Element": "{:.5f}",
    "Mean Squared L2 Per Sample": "{:.4f}"
}))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 2. Prior Sampling & PCA Projection
For each model, exactly 2,000 synthetic observations are sampled from the prior $z \\sim p(z)$ and deterministically decoded $\\hat{x} = \\mu_\\theta(z)$.
Both real test samples ($N=1,440$) and generated samples ($N=2,000$) are projected onto the **canonical Phase 1 PCA space** fitted strictly on the training set:
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=str(fig_dir / "pca_test_vs_generated_all_grid.png"), width=950))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3. Latent Space Geometry & Prior Densities
Below, we visualize the encoded test set distributions ($z \\sim q_\\phi(z|x)$) alongside each model's prior components / density contours:
"""))

    cells.append(nbf.v4.new_code_cell("""display(Image(filename=str(fig_dir / "latent_space_all_grid.png"), width=800))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 4. Quantitative Distribution Matching & Cluster Coverage
- **Maximum Mean Discrepancy ($\\text{MMD}^2$)**: Computed with the unbiased minimum-variance estimator using an RBF Gaussian kernel:
  - **Data Space**: Median pairwise Euclidean distance on training subset ($N=600$): $\\sigma_{\\text{data}} = 6.4250$.
  - **Latent Space**: Median pairwise distance on 600 standard normal draws: $\\sigma_{\\text{latent}} = 1.6704$.
  - Evaluated on equal sample sizes ($m = n = 1,440$).
- **20D Cluster Validity & Coverage**:
  - Reference cluster centroids $C_k$ and 95th-percentile acceptance radii $R_k$ computed strictly on the validation set in 20D.
  - A generated sample is valid if its distance to the nearest centroid is $\\le R_k$.
  - A cluster is covered if it contains $\\ge 20$ valid samples.
"""))

    cells.append(nbf.v4.new_code_cell("""quant_table = eval_df[[
    "model_type", "data_mmd2", "latent_mmd2", "cluster_validity_rate_pct", "covered_clusters_count"
]].copy()
quant_table.columns = ["Model", "Data MMD^2 (sigma=6.425)", "Latent MMD^2 (sigma=1.670)", "20D Validity Rate (%)", "Covered Clusters (>=20)"]
display(quant_table.style.format({
    "Data MMD^2 (sigma=6.425)": "{:.6f}",
    "Latent MMD^2 (sigma=1.670)": "{:.6f}",
    "20D Validity Rate (%)": "{:.2f}%",
    "Covered Clusters (>=20)": "{}/8"
}))

display(Image(filename=str(fig_dir / "cluster_coverage_comparison.png"), width=750))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5. Mixture Component Assignments & Responsibilities
For the mixture models (B, C, D), we evaluate:
1. **Hard Assignment Proportions**: Proportion of test latent codes assigned to component $k$ via maximum prior density:
   $$k^*(z) = \\arg\\max_k \\mathcal{N}(z; \\mu_k, \\Sigma_k)$$
2. **Mean Posterior Responsibilities**: Expected responsibility across test codes:
   $$\\gamma_k(z) = \\frac{\\pi_k \\mathcal{N}(z; \\mu_k, \\Sigma_k)}{\\sum_{j} \\pi_j \\mathcal{N}(z; \\mu_j, \\Sigma_j)}$$
"""))

    cells.append(nbf.v4.new_code_cell("""with open(metrics_dir / "component_assignments.json", "r") as f:
    comp_data = json.load(f)

for m in ["B", "C", "D"]:
    hard_p = comp_data[m]["hard_proportions"]
    mean_r = comp_data[m]["mean_responsibilities"]
    df_comp = pd.DataFrame({
        "Component": [f"k={k}" for k in hard_p.keys()],
        "Hard Assignment (%)": [f"{v * 100:.2f}%" for v in hard_p.values()],
        "Mean Responsibility (%)": [f"{mean_r[k] * 100:.2f}%" for k in hard_p.keys()]
    })
    print(f"\\n--- Model {m} Component Utilization ---")
    display(df_comp)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Master Synthesis & Research Question Answers

### 1. Reconstruction vs. Generation Quality Trade-Off
- **Model D** achieves the best reconstruction MSE ($0.07764$), followed by **Model A** ($0.07802$). Unimodal or low-component priors give the encoder maximum flexibility to position latent codes for optimal reconstruction.
- However, **Model B** and **Model C** achieve vastly superior generation quality (Data MMD$^2 \\approx 0.0009 - 0.0019$ vs $0.0066$ for Model A; Validity Rate $> 95\\%$ vs $80.1\\%$ for Model A).
- Multi-modal priors eliminate the "prior hole" problem in unimodal VAEs, matching the true clustered structure of the data distribution.

### 2. Fixed GMM vs. Learnable GMM vs. VampPrior
- **Model B (Fixed GMM)** incorporates exact domain oracle knowledge ($r=2.0, \\sigma=0.40$), yielding the best MMD scores.
- **Model C (Learnable GMM)** autonomously discovers 8 dispersed cluster centers in 2D latent space and achieves the highest 20D validity rate ($95.65\\%$).
- **VampPrior** (Tomczak & Welling, 2018) couples the prior directly to the encoder via learned pseudo-inputs $u_k \\in \\mathbb{R}^{20}$ in data space: $p(z) = \\frac{1}{K} \\sum q_\\phi(z|u_k)$, eliminating the need to parameterize Gaussian components directly in latent space.

### 3. Verification & Unit Testing Suite
All 57 mathematical contracts, loss functions, architecture modules, and evaluation metrics can be verified end-to-end via:
> **Unit Test Execution Command**:
> ```bash
> python -m unittest discover -s tests -v
> ```


---
*Notebook generation completed successfully.*
"""))

    nb.cells = cells
    return nb

if __name__ == "__main__":
    notebook_path = Path("vae_experiments.ipynb")
    
    print("Building notebook structure...")
    nb = create_notebook()
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to: {notebook_path.resolve()}")
    
    print("Executing notebook to populate all cell outputs...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook executed and updated successfully at: {notebook_path.resolve()}")
