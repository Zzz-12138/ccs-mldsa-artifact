#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
./bin/fusion_unprotected_artifact data/unprotected/mldsa_44 1 44 o3 100 32 5 20 5 2000 results/unprotected_mldsa44.csv
./bin/fusion_unprotected_artifact data/unprotected/mldsa_65 1 65 o3 100 32 5 20 5 2000 results/unprotected_mldsa65.csv
./bin/fusion_unprotected_artifact data/unprotected/mldsa_87 1 87 o3 100 32 5 20 5 2000 results/unprotected_mldsa87.csv
