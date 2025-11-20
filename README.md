# CRISPYx

A lightweight toolkit for streaming CRISPR screen analysis that processes large datasets without loading them into memory. Uses backed AnnData `.h5ad` files to perform QC, pseudo-bulk aggregation, and differential expression while data remains on disk.

## Features

- **Streaming QC** – Filters low-quality cells, perturbations, and genes with automatic control label detection and adaptive thresholds
- **Scanpy-style API** – Familiar `cx.pp`, `cx.pb`, and `cx.tl` namespaces for quality control, pseudo-bulk, and differential expression
- **Pseudo-bulk aggregation** – Average log expression and pseudo-bulk counts for effect size estimation
- **Differential expression** – Wald, Wilcoxon, and negative binomial GLM tests with multi-core support
- **Adaptive processing** – Automatic QC parameter calculation and dataset standardization with caching
- **Production-ready** – Comprehensive benchmarking scripts with logging and multi-dataset support

## Quick Start

```python
import crispyx as cx

# Open dataset without loading into memory
adata = cx.read_h5ad_ondisk("data/demo_benchmark.h5ad")

# Quality control with adaptive thresholds
adata = cx.pp.qc_summary(
    adata,
    perturbation_column="perturbation",
    min_genes=5,
    min_cells_per_perturbation=5,
)

# Pseudo-bulk aggregation
adata_pb = cx.pb.average_log_expression(
    adata,
    perturbation_column="perturbation",
)

# Differential expression with multi-core support
adata = cx.tl.rank_genes_groups(
    adata,
    perturbation_column="perturbation",
    method="wilcoxon",
    n_jobs=4,  # Use 4 cores
)

# Access results
print(adata.uns["rank_genes_groups"])
de_results = adata.uns["rank_genes_groups"].load()
```

## Installation

```bash
pip install -e .
```

## Benchmarking

### Single Dataset

```bash
cd benchmarking
./run_benchmark.sh config/Adamson.yaml
```

### Multiple Datasets

```bash
cd benchmarking
./run_benchmark.sh config/*.yaml
```

See `benchmarking/README.md` for detailed configuration options, adaptive features, and output structure.

## Testing

```bash
pytest  # Run all tests
pytest tests/test_workflow.py  # Run specific test file
```

## Documentation

Build docs locally:

```bash
sphinx-build docs docs/_build
```

## Output Files

Results use standardized naming: `{procedure}_{method}.h5ad`

```
results/DatasetName/
├── qc_filtered.h5ad           # QC output
├── de_wald.h5ad               # Wald test results
├── de_wilcoxon.h5ad           # Wilcoxon test results
├── pb_avg_log_effects.h5ad    # Average log expression
├── results.csv                # Benchmark summary
└── summary.json               # Metadata with adaptive QC params
```
