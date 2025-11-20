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
control_label: null           # Auto-detect
gene_name_column: "gene_symbols"

# Quality control parameters
qc_params: null               # null = adaptive calculation
adaptive_qc_mode: conservative

# Resource limits
resource_limits:
  time_limit: 36000           # 10 hours
  memory_limit: 128           # GB

# Parallelization
parallel_config:
  n_cores: 16                 # or null to auto-detect

# Other options
force_restandardize: false
show_progress: true
quiet: false
methods_to_run: null          # null = run all methods
```

**Example**: See `config/adamson_only.yaml`

## Adaptive Features

### Automatic QC Parameters
When `qc_params: null`, calculates from data:
- **Conservative** (default): 10th percentile thresholds, retains ~90% data
- **Aggressive**: 5th percentile thresholds, retains ~95% data

Example adaptive output for 65k cell dataset:
```json
{
  "min_genes": 50,
  "min_cells_per_perturbation": 50,
  "min_cells_per_gene": 5,
  "chunk_size": 8192
}
```

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
- Logs: `logs/benchmark_{timestamp}.log` (main) + `logs/{dataset}_{timestamp}.log` (per-dataset)

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
- Logs all output to `logs/inspect_datasets_{timestamp}.log`
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

```
benchmarking/results/
├── DatasetName/
│   ├── .cache/
│   │   └── standardized_DatasetName.h5ad  # Cached standardized dataset
│   ├── crispyx/
│   │   ├── qc_filtered.h5ad
│   │   ├── de_wald.h5ad
│   │   ├── de_wilcoxon.h5ad
│   │   └── pb_avg_log_effects.h5ad
│   ├── scanpy/                # Scanpy comparison outputs
│   │   └── de_wilcoxon.csv
│   ├── pertpy/                # Pertpy comparison outputs
│   │   ├── de_edger_wald.h5ad
│   │   └── de_pydeseq2_wald.h5ad
│   ├── results.csv            # Benchmark summary table
│   ├── results.md             # Markdown report
│   └── summary.json           # Metadata with adaptive QC params
└── logs/
    └── benchmark_*.log
```

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

## Comparison Methods

**Quality Control**: Scanpy in-memory pipeline vs CRISPYx streaming  
**Differential Expression**:
- Scanpy: t-test, Wilcoxon
- Pertpy: edgeR, PyDESeq2, statsmodels
- CRISPYx: Wald, Wilcoxon, NB-GLM (all with multi-core support)

**Metrics**: Pearson/Spearman correlation, top-k overlap, max absolute difference, AUROC

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
   tail -f benchmarking/logs/benchmark_*.log
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

# Check logs
ls -lt benchmarking/logs/ | head
```

## Troubleshooting

**Permission issues**: `chmod +x *.sh`  
**Check progress**: `tail -f logs/benchmark_*.log`  
**Force re-standardization**: Set `force_restandardize: true`  
**Override adaptive QC**: Specify explicit `qc_params` in config

