# Streamlined CRISPR Screen Analysis

This package provides a lightweight toolkit for performing key steps of CRISPR screen analysis without loading the entire dataset into memory. It operates directly on standard AnnData `.h5ad` files using backed access so that large cell-by-gene matrices remain on disk while QC, pseudo-bulk aggregation, and differential expression calculations stream across the data.

## Features
- Quality control filters for low quality cells, perturbations, and genes with automatic verification of gene symbol columns. Filtered cell × gene matrices are persisted as `{dataset}_filtered.h5ad`.
- Pseudo-bulk aggregation for effect size estimation using both averaged log counts and pseudo-bulk counts. Each estimator produces an AnnData file of effect sizes for downstream inspection.
- Differential expression testing with Wald and Wilcoxon tests that can skip lowly expressed genes for stability. Result matrices are saved as AnnData files containing effect sizes, statistics, and p-values.

## Development
Run the tests with:

```bash
pytest
```
