# When Lattice Algebra Leaks: Horizontal Fusion Attacks on ML-DSA

This repository contains the Proof-of-Concept (PoC) artifacts for the paper **"When Lattice Algebra Leaks: Horizontal Fusion Attacks on ML-DSA"**.

We demonstrate a side-channel attack framework targeting the algebraic redundancy in lattice-based cryptography (specifically ML-DSA/Dilithium). By combining **Joint Leakage Fusion** (aggregating leakage from $w=Ay$ and $u=cs_1$) with a deterministic **Algebraic Sieve** (exploiting INTT diffusion), we achieve full key recovery on ARM Cortex-M4 implementations with high efficiency.

## 📂 Project Structure

```text
.
├── data/                   # Place raw binary trace files here (Not included in repo)
├── results/                # Output directory for PDF plots and key candidates
├── scripts/                # Core analysis scripts
│   ├── mldsa_stepwise_pcc_lines.py  # Step-wise PCC trend analysis (Top 10 Plots)
│   ├── attack_full_enum.py          # Full-space enumeration attack
│   └── run_sieves.py                # Dual-Domain Sieve implementation
├── requirements.txt        # Python dependencies
└── README.md               # This file

```

## 🚀 Environment Setup

### 1. Prerequisites

The attack scripts are optimized for high-performance computing (64-core CPU recommended for full enumeration) and support GPU acceleration via `cupy`.

**Required Hardware:**

* **CPU:** Multi-core support (OpenMP/BLAS optimized).
* **RAM:** >64GB recommended for full dataset loading.
* **GPU (Optional but Recommended):** NVIDIA GPU for fast FFT filtering and correlation.

### 2. Installation

If running in **GitHub Codespaces** or a local Python environment:

```bash
# Update pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

```

*(Note: If you do not have an NVIDIA GPU available in your environment, switch `cupy` to `numpy` in the imports, though performance will decrease.)*

## 🧪 Usage Workflow

### Step 1: Data Preparation

Ensure your binary trace files are placed in the `data/` directory (or update the `DATA_DIR` path in the scripts).
Expected file format: Flat `float32` binaries reshaped to `(N_traces, 256, Trace_Len)`.

### Step 2: Step-wise PCC Trend Analysis

Run this script to visualize how the correct key candidate emerges as the number of traces increases (e.g., from 4 to 40). This generates the PDF plots showing the "Top 10" candidates.

```bash
python scripts/mldsa_stepwise_pcc_lines.py

```

**Output:** PDFs in `results/mldsa_65/` (e.g., `mldsa_65_u_cs1_pcc_lines_coeff_6_num_40.pdf`).

### Step 3: Full Key Recovery & Sieving

Run the full enumeration attack to recover all 256 coefficients and apply the Dual-Domain Sieve to verify correctness.

```bash
python scripts/attack_full_enum.py

```

## 📊 Methodology Summary

### 1. Joint Leakage Fusion (The Attack)

We utilize the **Holographic** view of the secret key :

* **Leakage A:** Sparse polynomial multiplication  (Input-dependent).
* **Leakage B:** Matrix-vector multiplication  (Structure-dependent).
* **Fusion:** We compute the Pearson Correlation Coefficient (PCC) across the full key space () to rank candidates.

### 2. Algebraic Sieve (The Verification)

To eliminate false positives without oracle access:

* **INTT Diffusion:** We verify that  falls within the small norm range . A single error in the NTT domain results in a uniform random distribution in the normal domain, making false positives cryptographically negligible.
* **Protocol Check:** We verify  satisfies the public key equation.

## 📈 Results

| Implementation | Optimization | Sampling Rate | Traces Required |
| ---- | --- | --- | --- |
| **Unprotected** | -O3 (Time) | 25 MSa/s | **< 10** |
| **Unprotected** | -O0 | 25 MSa/s | **< 15** |
| **Masked (CHES 2024)** | -O3 | 25 MSa/s | **< 1,000** |

## ⚠️ Disclaimer

This code is provided for research and educational purposes only. The traces and keys provided in this repository are from a laboratory test environment and do not represent real-world sensitive data.

```
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
tqdm>=4.65.0
pandas>=2.0.0
cupy-cuda11x; platform_system!="Darwin"
```