#!/bin/bash
#SBATCH --job-name=crispyx-bench
#SBATCH --partition=intel
#
# SLURM job script for running CRISPYx benchmarks with Singularity
#
# IMPORTANT: Use submit_benchmark.sh to auto-configure resources from config:
#   ./submit_benchmark.sh Adamson_subset.yaml
#   ./submit_benchmark.sh Replogle-GW-k562.yaml
#
# Log files are named: {dataset}_{YYYYMMDD_HHMMSS}.out/.err
#
# Or override manually:
#   sbatch --time=24:00:00 --mem=256G --cpus-per-task=32 slurm_benchmark.sh config.yaml

set -e

# ============================================================================
# IMPORTANT: Set PROJECT_ROOT to the absolute path of your project on HPC
# This is required because SLURM copies scripts to a spool directory,
# breaking relative path resolution.
# ============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-/lustre1/g/ids_du/Streamlining-CRISPR-Screen-Analysis}"

# Derived paths from PROJECT_ROOT
SINGULARITY_DIR="$PROJECT_ROOT/benchmarking/singularity"

# Configuration
CONFIG_FILE="${1:-Adamson_subset.yaml}"
MEMORY_LIMIT_GB="${2:-128}"  # Memory limit in GB for Singularity (from submit_benchmark.sh)
SIF_FILE="${CRISPYX_SIF:-$SINGULARITY_DIR/crispyx-benchmark.sif}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/benchmarking/results}"
LOGS_DIR="${LOGS_DIR:-$PROJECT_ROOT/benchmarking/logs}"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/benchmarking/config}"

# Use SLURM-allocated resources
N_CORES="${SLURM_CPUS_PER_TASK:-8}"

# Create output directories (using absolute paths)
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Print job info
echo "========================================"
echo "CRISPYx Benchmark - SLURM Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $N_CORES"
echo "SLURM Memory: $((SLURM_MEM_PER_NODE / 1024))G"
echo "Container Memory Limit: ${MEMORY_LIMIT_GB}G"
echo "Config: $CONFIG_FILE"
echo "Start time: $(date)"
echo "========================================"

# Validate files exist
if [ ! -f "$SIF_FILE" ]; then
    echo "ERROR: Singularity image not found: $SIF_FILE"
    exit 1
fi

# Paths are already absolute, no need to resolve
# Just verify they exist
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# Debug: List contents of data directory
echo "Data directory contents:"
ls -la "$DATA_DIR"
echo ""

# Check if the specific data file exists (extract from config if possible)
echo "Checking for h5ad files:"
ls -la "$DATA_DIR"/*.h5ad 2>/dev/null || echo "No .h5ad files found in $DATA_DIR"
echo ""

# Set thread limits
export OMP_NUM_THREADS=$N_CORES
export MKL_NUM_THREADS=$N_CORES
export OPENBLAS_NUM_THREADS=$N_CORES
export NUMBA_NUM_THREADS=$N_CORES

# Run benchmark
# Debug: Test if bind mount works by listing the data dir inside container
echo "Testing bind mount inside container:"
singularity exec \
    --bind "$DATA_DIR:/workspace/data:ro" \
    "$SIF_FILE" \
    ls -la /workspace/data/
echo ""

# Note: We bind the entire PROJECT_ROOT to /workspace to preserve relative paths
# This is simpler and more reliable than binding individual subdirectories
# Memory limits are enforced by SLURM (--mem flag in sbatch), not Singularity
# Singularity's --memory flag requires cgroups v2 in unified mode, which many
# HPC clusters don't support. SLURM's memory enforcement is sufficient.
singularity exec \
    --bind "$PROJECT_ROOT:/workspace:rw" \
    --pwd /workspace \
    --env "OMP_NUM_THREADS=$N_CORES" \
    --env "MKL_NUM_THREADS=$N_CORES" \
    --env "OPENBLAS_NUM_THREADS=$N_CORES" \
    --env "NUMBA_NUM_THREADS=$N_CORES" \
    --env "PYTHONUNBUFFERED=1" \
    "$SIF_FILE" \
    python -m benchmarking.tools.run_benchmarks --config "benchmarking/config/$CONFIG_FILE"

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
