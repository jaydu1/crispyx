# Benchmarking

Benchmark crispyx streaming methods against Scanpy, Pertpy, edgeR, and PyDESeq2.

## Quick Start

```bash
# Native mode
./run_benchmark.sh config/Adamson.yaml

# Docker mode
DOCKER_BUILDKIT=1 docker build -t crispyx-benchmark -f benchmarking/Dockerfile .
./run_benchmark.sh --use-docker config/Adamson.yaml
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--use-docker` | Run in Docker container |
| `--build-docker` | Build image before running |
| `--force` | Force re-run. With `--methods`, clears cache for specified methods only (preserves others for reports). Without `--methods`, clears entire cache. |
| `--clean` | Delete output directory first |
| `--methods X Y` | Run only specified methods |

## Configuration

Each dataset has its own YAML config. See `config/Adamson.yaml` for a complete example.

```yaml
dataset_path: "/path/to/dataset.h5ad"
output_dir: "benchmarking/results/DatasetName"
perturbation_column: "gene"
control_label: null              # Auto-detects 'CTRL'
qc_params: null                  # Adaptive QC (recommended)
resource_limits:
  memory_limit: 128.0            # GB
```

## Benchmark Methods

**crispyx**: `crispyx_qc_filtered`, `crispyx_pb_avg_log`, `crispyx_pb_pseudobulk`, `crispyx_de_t_test`, `crispyx_de_wilcoxon`, `crispyx_de_nb_glm`

**Reference**: `scanpy_qc_filtered`, `scanpy_de_t_test`, `scanpy_de_wilcoxon`, `edger_de_glm`, `pertpy_de_pydeseq2`

**Metrics**: Runtime, peak memory, Pearson/Spearman correlation, top-k overlap

## Output Structure

```
results/DatasetName/
├── .benchmark_cache/*.json         # Benchmark metadata
├── preprocessing/*.h5ad            # QC and pseudobulk results
├── de/*.h5ad, *.csv                # DE results
├── comparisons/*.csv               # Detailed comparisons
└── results.csv                     # Summary table
```

## Module Structure

The benchmarking tools are organized into focused modules:

- `constants.py`: Shared constants (method names, display order, cache version)
- `cache.py`: Cache I/O (save/load results, path resolution)
- `formatting.py`: Display utilities (method formatting, markdown tables)
- `comparison.py`: Statistical comparisons between DE methods
- `visualization.py`: Overlap heatmap generation
- `run_benchmarks.py`: Main benchmark runner
- `generate_results.py`: Report generation from cached results
- `rerun_scanpy.py`: Rerun Scanpy methods without resource limits

## Rerun Scanpy (for Large Datasets)

For large datasets where Scanpy times out or runs out of memory during benchmarks,
use the rerun_scanpy script AFTER benchmarks complete:

```bash
# Native mode - rerun Scanpy methods for one dataset
./run_rerun_scanpy.sh config/Replogle-GW-k562.yaml

# Rerun specific methods only
./run_rerun_scanpy.sh --methods scanpy_de_wilcoxon config/Feng-ts.yaml

# Force re-run even if outputs exist
./run_rerun_scanpy.sh --force config/Replogle-GW-k562.yaml

# Skip report regeneration
./run_rerun_scanpy.sh --no-report config/Replogle-GW-k562.yaml
```

**SLURM submission (HPC):**

```bash
cd benchmarking/singularity

# Submit single dataset
./submit_rerun_scanpy.sh Replogle-GW-k562.yaml

# Submit with options
./submit_rerun_scanpy.sh Feng-ts.yaml --force

# Submit all large datasets (default)
./submit_rerun_scanpy.sh
```

**Key features:**
- No time/memory limits (methods run to completion)
- Outputs saved to same locations as benchmarks
- Does NOT modify .benchmark_cache
- Automatically regenerates reports with accuracy comparisons

## Docker

```bash
# Build
DOCKER_BUILDKIT=1 docker build -t crispyx-benchmark -f benchmarking/Dockerfile .

# Run with memory limit
docker run --rm --memory=64g -v $(pwd):/workspace crispyx-benchmark \
  python -m benchmarking.tools.run_benchmarks --config benchmarking/config/Adamson.yaml
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Permission denied | `chmod +x *.sh` |
| Docker image not found | Build with command above |
| Out of memory | Increase `--memory` flag or `memory_limit` in config |

---
See [CHANGELOG.md](../CHANGELOG.md) for version history.