#!/bin/bash
#SBATCH --job-name=rerun-scanpy
#SBATCH --partition=intel
#
# SLURM job script for rerunning Scanpy methods with Singularity
#
# IMPORTANT: Use submit_rerun_scanpy.sh to auto-configure resources from config:
#   ./submit_rerun_scanpy.sh Adamson_subset.yaml
#   ./submit_rerun_scanpy.sh Replogle-GW-k562.yaml
#
# Log files are named: rerun_scanpy_{dataset}_{YYYYMMDD_HHMMSS}.out/.err

set -e

# ============================================================================
# IMPORTANT: Set PROJECT_ROOT to the absolute path of your project on HPC
# ============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-/lustre1/g/ids_du/Streamlining-CRISPR-Screen-Analysis}"

# Derived paths from PROJECT_ROOT
SINGULARITY_DIR="$PROJECT_ROOT/benchmarking/singularity"

# Configuration
CONFIG_FILE="${1:-Adamson_subset.yaml}"
MEMORY_LIMIT_GB="${2:-128}"
EXTRA_ARGS="${3:-}"
SIF_FILE="${CRISPYX_SIF:-$SINGULARITY_DIR/crispyx-benchmark.sif}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/benchmarking/results}"
LOGS_DIR="${LOGS_DIR:-$PROJECT_ROOT/benchmarking/logs}"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_ROOT/benchmarking/config}"

# Use SLURM-allocated resources
N_CORES="${SLURM_CPUS_PER_TASK:-8}"

# Create output directories
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Print job info
echo "========================================"
echo "Rerun Scanpy - SLURM Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $N_CORES"
echo "SLURM Memory: $((SLURM_MEM_PER_NODE / 1024))G"
echo "Config: $CONFIG_FILE"
[ -n "$EXTRA_ARGS" ] && echo "Extra args: $EXTRA_ARGS"
echo "Start time: $(date)"
echo "========================================"

# Validate files exist
if [ ! -f "$SIF_FILE" ]; then
    echo "ERROR: Singularity image not found: $SIF_FILE"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# Set thread limits
export OMP_NUM_THREADS=$N_CORES
export MKL_NUM_THREADS=$N_CORES
export OPENBLAS_NUM_THREADS=$N_CORES
export NUMBA_NUM_THREADS=$N_CORES

# Build command with optional extra args
CMD="python -m benchmarking.tools.rerun_scanpy --config benchmarking/config/$CONFIG_FILE"
if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

echo "Running: $CMD"
singularity exec \
    --bind "$PROJECT_ROOT:/workspace:rw" \
    --pwd /workspace \
    --env "OMP_NUM_THREADS=$N_CORES" \
    --env "MKL_NUM_THREADS=$N_CORES" \
    --env "OPENBLAS_NUM_THREADS=$N_CORES" \
    --env "NUMBA_NUM_THREADS=$N_CORES" \
    --env "PYTHONUNBUFFERED=1" \
    "$SIF_FILE" \
    bash -c "$CMD"

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
