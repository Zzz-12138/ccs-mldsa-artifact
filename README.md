# When Lattice Algebra Leaks: CCS Artifact

This repository contains a compact artifact-evaluation package for the ML-DSA horizontal fusion experiments. The default recovery path uses two measured operations, `A*y` and `c*s1`, and reports standalone operation recovery, LD-guided fusion, and fusion with the algebraic sieve.

## Tested Platform

The CUDA code was developed and tested on NVIDIA GeForce RTX 3090 GPUs (Ampere, compute capability `sm_86`). The Makefile intentionally targets `sm_86` because this is the verified setup. GPUs with the same compute capability are expected to run the code, subject to memory limits, but the submitted runs were verified on RTX 3090.

## Data

The trace archives are distributed as artifact/release assets rather than normal Git files:

- `data_unprotected_all.tar.gz`: 20-trace subsets for unprotected ML-DSA-44/65/87.
- `data_masked_44.tar.gz`: 200-trace subset for first-order masked ML-DSA-44.
- `data_masked_65.tar.gz`: 200-trace subset for first-order masked ML-DSA-65.
- `data_masked_87.tar.gz`: 200-trace subset for first-order masked ML-DSA-87.

The compact dataset intentionally excludes raw INTT traces. INTT is not part of the default recovery path; it is discussed as an ablation/leakage-surface observation and can be documented with precomputed logs.

After downloading the archives into `archives/`, verify and extract from the repository root:

```bash
sha256sum -c SHA256SUMS.txt
tar -xzf archives/data_unprotected_all.tar.gz
tar -xzf archives/data_masked_44.tar.gz
tar -xzf archives/data_masked_65.tar.gz
tar -xzf archives/data_masked_87.tar.gz
```

## Build

```bash
make
```

This builds:

- `bin/fusion_unprotected_artifact`
- `bin/fusion_masked_artifact`

## Default Runs

Unprotected, all three variants:

```bash
bash scripts/run_unprotected_all.sh
```

Masked/protected, all three variants:

```bash
bash scripts/run_masked_all.sh
```

If some GPUs are busy, restrict execution to free devices:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_masked_all.sh
```

The masked script defaults to `CUDA_VISIBLE_DEVICES=0,1` when the variable is not already set, and passes `use_intt=0`, matching the submitted compact dataset and default recovery path.

## Output Columns

The reviewer-facing CSVs contain:

- `Succ_cs1`: recovered coefficients using `c*s1` alone.
- `Succ_ay`: recovered coefficients using `A*y` alone.
- `Succ_fusion`: recovered coefficients using LD-guided fusion.
- `Succ_fusion_sieve`: recovered coefficients after the algebraic sieve.
- timing columns for scoring, fusion, sieve, and total runtime.

The subset is intended for artifact evaluation and sanity-check reproduction. Full-run CSV/log summaries should be used to document the complete paper-scale experiments.

