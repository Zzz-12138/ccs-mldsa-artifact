#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

: "${CUDA_VISIBLE_DEVICES:=0,1}"

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" ./bin/fusion_masked_artifact \
  data/masked_ches24/mldsa_44 1 44 o3 100 32 20 200 20 2000 0 \
  results/masked_mldsa44.csv

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" ./bin/fusion_masked_artifact \
  data/masked_ches24/mldsa_65 1 65 o3 100 32 20 200 20 2000 0 \
  results/masked_mldsa65.csv

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" ./bin/fusion_masked_artifact \
  data/masked_ches24/mldsa_87 1 87 o3 100 32 20 200 20 2000 0 \
  results/masked_mldsa87.csv
