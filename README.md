# Streamlined CRISPR Screen Analysis

This package provides a lightweight toolkit for performing key steps of CRISPR screen analysis without loading the entire dataset into memory. It operates directly on standard AnnData `.h5ad` files using backed access so that large cell-by-gene matrices remain on disk while QC, pseudo-bulk aggregation, and differential expression calculations stream across the data.

## Features
- Quality control filters for low quality cells, perturbations, and genes with automatic verification of gene symbol columns. When a gene column is not provided the toolkit falls back to `adata.var_names` and logs the decision, and control labels such as `ctrl`/`nontarget` are detected automatically. Filtered cell × gene matrices are persisted as `{dataset}_filtered.h5ad`.
- Pseudo-bulk aggregation for effect size estimation using both averaged log counts and pseudo-bulk counts. Each estimator produces an AnnData file of effect sizes for downstream inspection.
- Differential expression testing with Wald and Wilcoxon tests that can skip lowly expressed genes for stability. Result matrices are saved as AnnData files containing effect sizes, statistics, and p-values, and they inherit the same automatic control selection used in the earlier steps.
- Lightweight `preview_backed` helper that opens a backed AnnData file, prints a small metadata summary, and returns the read-only object for further inspection without fully materialising the matrix.
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
results directory. The Markdown report now opens with a short narrative that
highlights overall success rates, category-level runtimes, and any dependency
issues before presenting the detailed tables.

Benchmark outputs report more than the maximum absolute differences between
streaming and reference pipelines. Each differential expression comparison
now includes Pearson and Spearman correlations, top-`k` overlaps (by default
`k=50`) for effect sizes, statistics, and *p*-values, as well as AUROC scores
when ground-truth labels are present in the merged results. These metrics
surface whether rankings agree in addition to absolute magnitudes, making it
easier to spot systematic discrepancies.

Every run additionally emits a machine-readable
`benchmark_results_summary.json` containing aggregate counts, runtime averages,
and grouped error metadata so dashboards can ingest the results directly
without parsing tables.

## Documentation

Sphinx configuration files live under `docs/` so the package can be published on
Read the Docs. Build the documentation locally with:

```bash
sphinx-build docs docs/_build
```
