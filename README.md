# crispyx

A lightweight toolkit for streaming CRISPR screen analysis that processes large datasets without loading them into memory. Uses backed AnnData `.h5ad` files to perform QC, pseudo-bulk aggregation, and differential expression while data remains on disk.

## Features

- **Streaming QC** – Filters low-quality cells, perturbations, and genes with automatic control label detection and adaptive thresholds
- **Streaming preprocessing** – Normalize and log-transform large datasets without loading into memory
- **Streaming dimension reduction** – Memory-efficient PCA with automatic method selection (sparse covariance for ≤15K genes, IncrementalPCA for larger) and KNN graph construction
- **Scanpy-style API** – Familiar `cx.pp`, `cx.pb`, and `cx.tl` namespaces for quality control, pseudo-bulk, and differential expression
- **Scanpy-style plotting** – `cx.pl` helpers for rank genes groups, volcano/MA plots, PCA visualization, and QC summaries without loading counts
- **Pseudo-bulk aggregation** – Average log expression and pseudo-bulk counts for effect size estimation
- **Differential expression** – t-test, Wilcoxon, and negative binomial GLM tests with multi-core support and adaptive memory management (`memory_limit_gb`); Wilcoxon uses a batched parallel Numba kernel (prange over perturbation groups) with float32 processing (~50% memory reduction vs float64)
- **LFC shrinkage** – apeGLM adaptive shrinkage for more accurate log-fold change estimates
- **Resume/checkpoint** – Long-running analyses can be resumed after interruption
- **Adaptive processing** – Automatic QC parameter calculation and dataset standardization with caching; gene chunk sizing is memory-aware (cell-count OOM caps are lifted on 128 GB+ systems)
- **Production-ready** – Comprehensive benchmarking scripts with logging, lazy-loading GLM comparisons, and multi-dataset support

See [CHANGELOG.md](CHANGELOG.md) for version history and breaking changes.

## Quick Start

```python
import crispyx as cx

# Open dataset without loading into memory
adata = cx.read_h5ad_ondisk("data/demo_benchmark.h5ad")

# Streaming normalization + log1p (for t-test/Wilcoxon)
adata_norm = cx.pp.normalize_total_log1p(
    adata,  # Pass AnnData object directly
    output_dir="results/",
    data_name="normalized",
)

# Quality control with adaptive thresholds
adata = cx.pp.qc_summary(
    adata,
    perturbation_column="perturbation",
    min_genes=5,
    min_cells_per_perturbation=5,
)

# Streaming PCA (memory-efficient, auto-selects optimal method)
cx.pp.pca(adata, n_comps=50)

# KNN graph for clustering/UMAP
cx.pp.neighbors(adata, n_neighbors=15)

# PCA plotting (Scanpy-style)
cx.pl.pca_variance_ratio(adata)  # Variance explained
cx.pl.pca(adata, color="perturbation")  # Scatter colored by perturbation

# Pseudo-bulk aggregation
adata_pb = cx.pb.average_log_expression(
    adata,
    perturbation_column="perturbation",
)

# Differential expression with multi-core support
adata = cx.tl.rank_genes_groups(
    adata,
    perturbation_column="perturbation",
    method="t-test",  # or "wilcoxon", "nb-glm"
    n_jobs=4,  # Use 4 cores
)

# For NB-GLM with LFC shrinkage (two-step workflow)
result = cx.nb_glm_test(
    adata,
    perturbation_column="perturbation",
)
shrunk = cx.shrink_lfc(result.result_path)

# Resume interrupted analyses
result = cx.nb_glm_test(
    adata,
    perturbation_column="perturbation",
    resume=True,  # Skip completed perturbations
    checkpoint_interval=10,  # Save progress every 10 perturbations
)

# Access results
print(adata.uns["rank_genes_groups"])
de_results = adata.uns["rank_genes_groups"].load()

# Plotting (Scanpy-style)
cx.pl.rank_genes_groups(adata, n_genes=20, sharey=False)
df = cx.pl.rank_genes_groups_df(adata, group="perturbation_A", n_genes=200)
cx.pl.volcano(de_df=df, group="perturbation_A")
cx.pl.ma(
    data=adata,
    de_result=adata,
    group="perturbation_A",
    reference="control",
    perturbation_column="perturbation",
    mean_mode="raw",
)
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

> **Note:** The Docker image pre-compiles Numba JIT kernels at build time, eliminating
> cold-start delays on the first Wilcoxon run. The Singularity SLURM script runs a
> single-threaded warmup step automatically on first use (guarded by a cache sentinel),
> preventing LLVM OOM on multi-core HPC nodes.

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

## Output Structure

crispyx uses a standardized directory structure with `crispyx_` prefixed filenames:

- **Preprocessing**: `crispyx_qc_filtered.h5ad`, `crispyx_qc_filtered_preprocessed.h5ad`, `crispyx_pb_avg_log.h5ad`, `crispyx_pb_pseudobulk.h5ad`
- **Differential expression**: `crispyx_de_t_test.h5ad`, `crispyx_de_wilcoxon.h5ad`, `crispyx_de_nb_glm.h5ad`
- **Reference comparisons**: `scanpy_de_*.csv`, `edger_de_glm.csv`, `pertpy_de_pydeseq2.csv`

See [benchmarking/README.md](benchmarking/README.md#output-structure) for the complete directory structure and file naming conventions.
