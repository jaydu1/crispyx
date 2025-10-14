# Streamlined CRISPR Screen Analysis

This package provides a lightweight toolkit for performing key steps of CRISPR screen analysis without loading the entire dataset into memory. It operates directly on standard AnnData `.h5ad` files using backed access so that large cell-by-gene matrices remain on disk while QC, pseudo-bulk aggregation, and differential expression calculations stream across the data.

## Features
- Quality control filters for low quality cells, perturbations, and genes with automatic verification of gene symbol columns. Filtered cell × gene matrices are persisted as `{dataset}_filtered.h5ad`.
- Pseudo-bulk aggregation for effect size estimation using both averaged log counts and pseudo-bulk counts. Each estimator produces an AnnData file of effect sizes for downstream inspection.
- Differential expression testing with Wald and Wilcoxon tests that can skip lowly expressed genes for stability. Result matrices are saved as AnnData files containing effect sizes, statistics, and p-values.
- Negative binomial GLM differential expression that regresses out measured covariates while reusing a streamed design matrix solver optimised for sparse counts. Fits can be initialised via Poisson IRLS, include early stopping for lowly expressed genes, and write results (including convergence diagnostics) to disk.

## Development
Run the tests with:

```bash
pytest
```

## Benchmarking

The `benchmarking` directory contains a reusable script for profiling the
streaming analysis methods. Generate the synthetic demo dataset with
`python benchmarking/generate_demo_dataset.py` (or provide your own `.h5ad`)
and then execute `python benchmarking/run_benchmarks.py` to generate CSV and
Markdown summaries alongside the intermediate `.h5ad` outputs in the selected
results directory.

## Documentation

Sphinx configuration files live under `docs/` so the package can be published on
Read the Docs. Build the documentation locally with:

```bash
sphinx-build docs docs/_build
```
