#!/usr/bin/env bash
set -euo pipefail

mkdir -p archives

for archive in \
  data_unprotected_all.tar.gz \
  data_masked_44.tar.gz \
  data_masked_65.tar.gz \
  data_masked_87.tar.gz
do
  echo "[reassemble] ${archive}"
  cat "archive_parts/${archive}.part-"* > "archives/${archive}"
done

sha256sum -c SHA256SUMS.txt
