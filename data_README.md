
# Artifact Data

The raw trace data is distributed as release assets, not as normal Git files.

Archives:

- data_unprotected_all.tar.gz: 20-trace representative subset for unprotected ML-DSA-44/65/87.
- data_masked_44.tar.gz: 200-trace representative subset for first-order masked ML-DSA-44.
- data_masked_65.tar.gz: 200-trace representative subset for first-order masked ML-DSA-65.
- data_masked_87.tar.gz: 200-trace representative subset for first-order masked ML-DSA-87.

All datasets contain only the two operations used in the default recovery path:
A*y and c*s1. INTT raw traces are intentionally excluded from the default
artifact dataset; INTT is documented through ablation logs/results.

After downloading the archives, extract them from the repository root:

tar -xzf data_unprotected_all.tar.gz
tar -xzf data_masked_44.tar.gz
tar -xzf data_masked_65.tar.gz
tar -xzf data_masked_87.tar.gz

Verify archive integrity with:

sha256sum -c SHA256SUMS.txt
