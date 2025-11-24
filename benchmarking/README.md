# Benchmarking

Scripts for profiling CRISPYx streaming methods against reference implementations (Scanpy, Pertpy, edgeR, PyDESeq2).

## Quick Start

### Single Dataset
```bash
./run_benchmark.sh config/Adamson.yaml
```

### Multiple Datasets
```bash
# Run specific datasets
./run_benchmark.sh config/Adamson.yaml config/Frangieh.yaml

# Run all datasets in config directory
./run_benchmark.sh config/*.yaml
```

### Generate Configs for All Datasets
```bash
# Auto-generates individual config files (one per dataset)
./inspect_datasets.sh

# Or with custom memory limit
./inspect_datasets.sh --memory-limit 64

# Or using reference config for memory limit
./inspect_datasets.sh --reference-config benchmark_config.yaml

# Then run all
./run_benchmark.sh config/*.yaml
```

## Configuration

Each dataset has its own YAML config file with all parameters.

### Single-Dataset Config (Recommended)
```yaml
dataset_path: "/data/projects/SeqExpDesign/data/origin/Adamson.h5ad"
output_dir: "benchmarking/results/Adamson"

# Dataset column configuration
perturbation_column: "gene"
control_label: null           # Auto-detects 'CTRL'
gene_name_column: null        # Uses var.index

# Quality control parameters - set to null to use adaptive calculation
qc_params: null               # Will be calculated adaptively based on data distribution

# Resource limits
resource_limits:
  time_limit: 3600            # 1 hour per method
  memory_limit: 128.0         # GB per method

# Parallelization configuration
parallel_config:
  n_cores: 32                 # or null to auto-detect

# Adaptive QC mode
force_restandardize: false    # Set to true to regenerate standardized files
adaptive_qc_mode: conservative  # or 'aggressive'

# Methods to run (null = run all available methods)
methods_to_run: null

# Progress and output options
show_progress: true
quiet: false
```

**Example**: See `config/Adamson.yaml`

## Adaptive Features

### Automatic QC Parameters
When `qc_params: null`, calculates from data:
- **Conservative** (default): 10th percentile thresholds, retains ~90% data
- **Aggressive**: 5th percentile thresholds, retains ~95% data

Example adaptive output calculated from data distribution:
```json
{
  "min_genes": 50,
  "min_cells_per_perturbation": 50,
  "min_cells_per_gene": 5,
  "chunk_size": 8192
}
```

Note: Chunk size is calculated based on memory limits and dataset dimensions.

### Dataset Standardization
Automatically:
- Renames perturbation column to `'perturbation'`
- Standardizes control labels to `'control'`
- Caches to `{output_dir}/.cache/standardized_{dataset}.h5ad`
- Reuses cache unless `force_restandardize: true`

## Scripts

### `run_benchmark.sh`
Run benchmarks on one or more datasets.

```bash
# Single dataset
./run_benchmark.sh config/Adamson.yaml

# Multiple datasets
./run_benchmark.sh config/Adamson.yaml config/Frangieh.yaml

# All datasets
./run_benchmark.sh config/*.yaml

# Datasets matching pattern
./run_benchmark.sh config/Replogle-*.yaml
```

**Features**:
- Accepts one or more config files as arguments
- Runs each dataset independently
- Separate log file for each dataset
- Continues on failure, reports summary at end
- Logs: `logs/{timestamp}_benchmark.log` (main) + `logs/{timestamp}_{dataset}.log` (per-dataset)

### `inspect_datasets.sh`
Wrapper script for dataset inspection with logging.

```bash
# Basic usage
./inspect_datasets.sh

# With custom memory limit
./inspect_datasets.sh --memory-limit 64

# Using reference config for memory limit
./inspect_datasets.sh --reference-config benchmark_config.yaml
```

**Features**:
- Logs all output to `logs/{timestamp}_inspect_datasets.log`
- Real-time console output with `tee`
- Passes all arguments to `inspect_datasets.py`

### `inspect_datasets.py`
Python script that auto-detects column structures and generates individual config files for all datasets.

```bash
python inspect_datasets.py
```

**Options**:
- `--reference-config PATH`: Read memory limit from existing YAML config (default: `benchmark_config.yaml`)
- `--memory-limit GB`: Override memory limit in GB for chunk size calculation

**Output**: Individual config files in `config/` directory (one per dataset)
- `config/Adamson.yaml`
- `config/Frangieh.yaml`
- `config/Replogle-GW-k562.yaml`
- etc.

**Memory-Aware Chunk Sizes**: When a memory limit is specified (either via `--reference-config` or `--memory-limit`), the script calculates optimal chunk sizes constrained by that memory limit. Otherwise, it uses available system memory.

## Output Structure

All CRISPYx outputs now use the `crispyx_` prefix to clearly distinguish them from reference tool outputs.

```
benchmarking/results/
├── DatasetName/
│   ├── .cache/
│   │   └── standardized_DatasetName.h5ad  # Cached standardized dataset
│   ├── .benchmark_cache/                  # Cached benchmark results (JSON metadata)
│   │   ├── crispyx_qc_filtered.json       # QC benchmark cache
│   │   ├── crispyx_pb_avg_log.json        # Avg log expression cache
│   │   ├── crispyx_pb_pseudobulk.json     # Pseudobulk cache
│   │   ├── crispyx_de_t_test.json         # t-test cache
│   │   ├── crispyx_de_wilcoxon.json       # Wilcoxon cache
│   │   ├── crispyx_de_nb_glm.json         # NB-GLM cache
│   │   ├── scanpy_qc_filtered.json        # Scanpy QC comparison cache
│   │   ├── scanpy_de_t_test.json          # Scanpy t-test comparison cache
│   │   ├── scanpy_de_wilcoxon.json        # Scanpy Wilcoxon comparison cache
│   │   ├── edger_de_glm.json              # edgeR comparison cache
│   │   └── pertpy_de_pydeseq2.json        # PyDESeq2 comparison cache
│   ├── preprocessing/
│   │   ├── crispyx_qc_filtered.h5ad       # Quality control filtered data
│   │   ├── crispyx_pb_avg_log.h5ad        # Average log expression pseudobulk
│   │   └── crispyx_pb_pseudobulk.h5ad     # Sum-based pseudobulk expression
│   ├── de/
│   │   ├── crispyx_de_t_test.h5ad         # CRISPYx t-test results
│   │   ├── crispyx_de_wilcoxon.h5ad       # CRISPYx Wilcoxon test results
│   │   ├── crispyx_de_nb_glm.h5ad         # CRISPYx negative binomial GLM results
│   │   ├── scanpy_de_t_test.csv           # Scanpy t-test results
│   │   ├── scanpy_de_wilcoxon.csv         # Scanpy Wilcoxon test results
│   │   ├── edger_de_glm.csv               # edgeR GLM results
│   │   └── pertpy_de_pydeseq2.csv         # PyDESeq2 results via Pertpy
│   ├── comparisons/                       # Detailed comparison data (optional)
│   │   ├── edger_glm.csv                  # CRISPYx vs edgeR detailed comparison
│   │   └── pertpy_pydeseq2.csv            # CRISPYx vs PyDESeq2 detailed comparison
│   ├── results.csv                        # Benchmark summary table
│   ├── results.md                         # Markdown report
│   └── summary.json                       # Metadata with adaptive QC params
└── logs/
    └── {timestamp}_*.log
```

### File Naming Convention

**CRISPYx outputs**: `crispyx_{operation}_{method}.h5ad`
- Examples: `crispyx_qc_filtered.h5ad`, `crispyx_de_t_test.h5ad`, `crispyx_pb_avg_log.h5ad`

**Reference tool outputs**: `{tool}_de_{method}.csv`
- Examples: `scanpy_de_t_test.csv`, `edger_de_glm.csv`, `pertpy_de_pydeseq2.csv`

**Comparison files**: `{tool}_{method}.csv` (in `comparisons/` directory)
- Examples: `comparisons/edger_glm.csv`, `comparisons/pertpy_pydeseq2.csv`

This naming convention makes it easy to identify the source and type of each file at a glance.

### Summary Metadata
`summary.json` includes adaptive decisions:
```json
{
  "adaptive_qc": true,
  "adaptive_qc_mode": "conservative",
  "qc_params_used": {
    "min_genes": 50,
    "min_cells_per_perturbation": 50,
    "min_cells_per_gene": 5,
    "chunk_size": 8192
  },
  "standardized_dataset_path": "results/Adamson/.cache/standardized_Adamson.h5ad",
  "original_dataset_path": "/data/projects/SeqExpDesign/data/origin/Adamson.h5ad"
}
```

## Benchmark Methods

**CRISPYx Streaming Pipeline** (6 methods):
1. `crispyx_qc_filtered` - Streaming quality control filters
2. `crispyx_pb_avg_log` - Average log-normalized expression per perturbation
3. `crispyx_pb_pseudobulk` - Pseudo-bulk log fold-change per perturbation
4. `crispyx_de_t_test` - t-test differential expression (parallelized)
5. `crispyx_de_wilcoxon` - Wilcoxon rank-sum differential expression (parallelized)
6. `crispyx_de_nb_glm` - Negative binomial GLM differential expression (parallelized)

**Reference Comparisons** (5 methods):
7. `scanpy_qc_filtered` - QC comparison against Scanpy
8. `scanpy_de_t_test` - t-test comparison against Scanpy
9. `scanpy_de_wilcoxon` - Wilcoxon comparison against Scanpy
10. `edger_de_glm` - GLM comparison against edgeR (via rpy2, parallelized, uses `crispyx_de_nb_glm` if available)
11. `pertpy_de_pydeseq2` - GLM comparison against PyDESeq2 via Pertpy (parallelized, uses `crispyx_de_nb_glm` if available)

**Comparison Metrics**: 
- **Accuracy**: Pearson/Spearman correlation, top-k overlap, max absolute difference, AUROC
- **Performance**: Runtime (seconds), peak memory (MB), average memory (MB)
  - **Runtime**: Measured excluding data loading overhead (timer starts after cache files are loaded)
  - **Peak Memory**: Maximum memory usage during execution
  - **Average Memory**: Mean memory usage sampled at 0.1-second intervals via background thread

**Standardized Comparison Pipeline**: All comparison methods (scanpy, edgeR, pertpy) load pre-computed CRISPYx streaming results from cache files to ensure consistent timing and avoid pipeline variation. This means:
- **t-test/Wilcoxon comparisons** require `crispyx_de_t_test` or `crispyx_de_wilcoxon` to exist first
- **GLM comparisons** (edgeR, PyDESeq2) require `crispyx_de_nb_glm` to exist first
- If required cache files are missing, the benchmark will fail explicitly with a clear error message

**Cache Dependencies**:
- `scanpy_de_t_test`: Loads from `crispyx_de_t_test.h5ad`
- `scanpy_de_wilcoxon`: Loads from `crispyx_de_wilcoxon.h5ad`
- `edger_de_glm`: Loads from `crispyx_de_nb_glm.h5ad` (GLM-to-GLM comparison)
- `pertpy_de_pydeseq2`: Loads from `crispyx_de_nb_glm.h5ad` (GLM-to-GLM comparison)

**Note**: The `pertpy_statsmodels` comparison is excluded because it does not support parallelization and is extremely slow on large datasets. PyDESeq2 provides equivalent GLM-based validation.

## Performance Tips

1. **Use adaptive QC** (`qc_params: null`) for optimal parameters
2. **Enable parallelization** (`n_cores: null` auto-detects)
3. **Reuse standardized datasets** (`force_restandardize: false`)
4. **One config per dataset** - Simple, flexible, easy to version control
5. **Run datasets incrementally**:
   ```bash
   # Test on one first
   ./run_benchmark.sh config/Adamson.yaml
   
   # Then run more
   ./run_benchmark.sh config/Adamson.yaml config/Frangieh.yaml config/Tian-crispra.yaml
   
   # Or run all
   ./run_benchmark.sh config/*.yaml
   ```
6. **Run in background** for long jobs:
   ```bash
   nohup ./run_benchmark.sh config/*.yaml &
   # Check latest log file
   tail -f benchmarking/logs/$(ls -1t benchmarking/logs/*.log | head -1)
   ```

## Workflow Example

### Step 1: Generate Config Files (One Time)
```bash
cd benchmarking
./inspect_datasets.sh
```
**Result**: Creates `config/Adamson.yaml`, `config/Frangieh.yaml`, etc.  
**Log**: `logs/inspect_datasets_{timestamp}.log`

### Step 2: Edit Config If Needed
```bash
vim config/Adamson.yaml
# Adjust memory limits, cores, QC mode, etc.
```

### Step 3: Run Benchmarks
```bash
# Single dataset
./run_benchmark.sh config/Adamson.yaml

# Multiple datasets
./run_benchmark.sh config/Adamson.yaml config/Frangieh.yaml

# All datasets
./run_benchmark.sh config/*.yaml
```

### Step 4: Check Results
```bash
# List completed datasets
ls -1 benchmarking/results/

# View summary
cat benchmarking/results/Adamson/summary.json
cat benchmarking/results/Adamson/results.csv

# Check logs (sorted chronologically)
ls -1t benchmarking/logs/ | head
```

## Troubleshooting

**Permission issues**: `chmod +x *.sh`  
**Check progress**: `tail -f logs/{timestamp}_benchmark.log`  
**Force re-standardization**: Set `force_restandardize: true`  
**Override adaptive QC**: Specify explicit `qc_params` in config  
**View all logs**: `ls -1 logs/` (timestamps at beginning for chronological sorting)

