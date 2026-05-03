# Artifact Data

The compact trace archives are stored as split files in `archive_parts/`.

From the repository root:

```bash
bash scripts/reassemble_archives.sh
tar -xzf archives/data_unprotected_all.tar.gz
tar -xzf archives/data_masked_44.tar.gz
tar -xzf archives/data_masked_65.tar.gz
tar -xzf archives/data_masked_87.tar.gz
```

The archives contain only the two operations used in the default recovery path: `A*y` and `c*s1`. Raw INTT traces are intentionally excluded from the compact artifact dataset.
