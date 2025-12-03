# CRISPYx

A lightweight toolkit for streaming CRISPR screen analysis that processes large datasets without loading them into memory. Uses backed AnnData `.h5ad` files to perform QC, pseudo-bulk aggregation, and differential expression while data remains on disk.

## Features

- **Streaming QC** – Filters low-quality cells, perturbations, and genes with automatic control label detection and adaptive thresholds
- **Scanpy-style API** – Familiar `cx.pp`, `cx.pb`, and `cx.tl` namespaces for quality control, pseudo-bulk, and differential expression
- **Pseudo-bulk aggregation** – Average log expression and pseudo-bulk counts for effect size estimation
- **Differential expression** – t-test, Wilcoxon, and negative binomial GLM tests with multi-core support
- **Adaptive processing** – Automatic QC parameter calculation and dataset standardization with caching
- **Production-ready** – Comprehensive benchmarking scripts with logging, lazy-loading GLM comparisons, and multi-dataset support

## Migration Notes

### v0.3.0+ (November 2025)

**Breaking Changes: File Naming Standardization**
- All CRISPYx output files now use `crispyx_` prefix for clarity
- Benchmark method names updated to match output filenames
- Output directory structure reorganized (see "Output Structure" section below)

**File Naming Changes:**
- `qc_filtered.h5ad` → `crispyx_qc_filtered.h5ad`
- `pb_avg_log_effects.h5ad` → `crispyx_pb_avg_log.h5ad`
- `pb_pseudobulk_effects.h5ad` → `crispyx_pb_pseudobulk.h5ad`
- `de_t_test.h5ad` → `crispyx_de_t_test.h5ad`
- `de_wilcoxon.h5ad` → `crispyx_de_wilcoxon.h5ad`
- `de_nb_glm.h5ad` → `crispyx_de_nb_glm.h5ad`

**Reference Tool Outputs:**
- Scanpy: `scanpy_de_t_test.csv`, `scanpy_de_wilcoxon.csv`
- edgeR: `edger_de_glm.csv`
- Pertpy: `pertpy_de_pydeseq2.csv`

**Benchmark Method Renaming:**
- `quality_control` → `crispyx_qc_filtered`
- `average_log_expression` → `crispyx_pb_avg_log`
- `pseudobulk_expression` → `crispyx_pb_pseudobulk`
- `t_test` → `crispyx_de_t_test`
- `wilcoxon_test` → `crispyx_de_wilcoxon`
- `nb_glm_test` → `crispyx_de_nb_glm`
- `scanpy_quality_control_comparison` → `scanpy_qc_filtered`
- `scanpy_t_test_comparison` → `scanpy_de_t_test`
- `scanpy_wilcoxon_comparison` → `scanpy_de_wilcoxon`
- `edger_direct_comparison` → `edger_de_glm`
- `pertpy_pydeseq2_comparison` → `pertpy_de_pydeseq2`

**Recommended Actions:**
- Clear old benchmark results: `rm -rf benchmarking/results/*`
- Re-run benchmarks to generate files with new naming structure
- Update any scripts that reference old file paths

### v0.2.0 (2024)

**Breaking Changes:**
- `wald_test` renamed to `t_test` for consistency with Scanpy
- Backward compatibility: `method="wald"` automatically maps to `t_test`

**New Features:**
- Added `nb_glm_test` for negative binomial GLM differential expression
- Lazy-loading GLM comparisons: edgeR and PyDESeq2 automatically use `nb_glm` results
- Standardized comparison CSV format
- Comparison results cached in `.benchmark_cache/` directory

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
    method="t-test",  # or "wilcoxon", "nb-glm"
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

## Output Structure

CRISPYx uses a standardized directory structure with module-prefixed filenames for clarity.

### Benchmark Results

```
benchmarking/results/DatasetName/
├── .cache/
│   └── standardized_DatasetName.h5ad    # Cached standardized dataset
├── .benchmark_cache/                    # JSON metadata for caching
│   ├── crispyx_qc_filtered.json
│   ├── crispyx_pb_avg_log.json
│   ├── crispyx_pb_pseudobulk.json
│   ├── crispyx_de_t_test.json
│   ├── crispyx_de_wilcoxon.json
│   ├── crispyx_de_nb_glm.json
│   ├── scanpy_qc_filtered.json
│   ├── scanpy_de_t_test.json
│   ├── scanpy_de_wilcoxon.json
│   ├── edger_de_glm.json
│   └── pertpy_de_pydeseq2.json
├── preprocessing/
│   ├── crispyx_qc_filtered.h5ad         # Quality control filtered data
│   ├── crispyx_pb_avg_log.h5ad          # Average log expression pseudobulk
│   └── crispyx_pb_pseudobulk.h5ad       # Sum-based pseudobulk expression
├── de/
│   ├── crispyx_de_t_test.h5ad           # CRISPYx t-test results
│   ├── crispyx_de_wilcoxon.h5ad         # CRISPYx Wilcoxon test results
│   ├── crispyx_de_nb_glm.h5ad           # CRISPYx negative binomial GLM results
│   ├── scanpy_de_t_test.csv             # Scanpy t-test results
│   ├── scanpy_de_wilcoxon.csv           # Scanpy Wilcoxon test results
│   ├── edger_de_glm.csv                 # edgeR GLM results
│   └── pertpy_de_pydeseq2.csv           # PyDESeq2 results via Pertpy
├── comparisons/                         # Detailed comparison data (optional)
│   ├── edger_glm.csv                    # CRISPYx vs edgeR detailed comparison
│   └── pertpy_pydeseq2.csv              # CRISPYx vs PyDESeq2 detailed comparison
├── results.csv                          # Benchmark summary table
├── results.md                           # Markdown report
└── summary.json                         # Metadata with adaptive QC params
```

### File Naming Convention

**CRISPYx outputs**: `crispyx_{operation}_{method}.h5ad`
- Preprocessing: `crispyx_qc_filtered.h5ad`, `crispyx_pb_avg_log.h5ad`, `crispyx_pb_pseudobulk.h5ad`
- Differential expression: `crispyx_de_t_test.h5ad`, `crispyx_de_wilcoxon.h5ad`, `crispyx_de_nb_glm.h5ad`

**Reference tool outputs**: `{tool}_de_{method}.csv`
- Examples: `scanpy_de_t_test.csv`, `edger_de_glm.csv`, `pertpy_de_pydeseq2.csv`

**Comparison files**: `{tool}_{method}.csv` (in `comparisons/` directory)
- Examples: `comparisons/edger_glm.csv`, `comparisons/pertpy_pydeseq2.csv`

### Standalone Analysis

When using CRISPYx outside of benchmarking, outputs follow the same naming but can be customized:

```python
import crispyx as cx

# Default naming with crispyx_ prefix
adata_qc = cx.pp.qc_summary(
    adata,
    perturbation_column="perturbation",
    output_dir="results",
    data_name="qc_filtered",  # Will create: crispyx_qc_filtered.h5ad
)

# Custom naming (overrides automatic prefixing)
adata_custom = cx.tl.rank_genes_groups(
    adata,
    perturbation_column="perturbation",
    method="t-test",
    output_dir="results",
    data_name="crispyx_my_analysis_t_test",  # Custom name
)
```
