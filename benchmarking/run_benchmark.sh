#!/bin/bash
# CRISPyx Benchmark Runner
# Runs benchmarks on one or more datasets with optional Docker support
#
# Usage:
#   ./run_benchmark.sh config.yaml                    # Single dataset (native)
#   ./run_benchmark.sh config1.yaml config2.yaml      # Multiple datasets
#   ./run_benchmark.sh config/*.yaml                  # All datasets
#   ./run_benchmark.sh --use-docker config.yaml       # Run in Docker
#   ./run_benchmark.sh --use-docker --docker-image myimage:tag config/*.yaml

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_ENV="${PYTHON_ENV:-$(command -v python)}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs"
MAIN_LOG="${LOG_DIR}/${TIMESTAMP}_benchmark.log"

# Docker settings
USE_DOCKER=false
DOCKER_IMAGE=""
BUILD_DOCKER=false

# Create log directory
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# Function to log errors
log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$MAIN_LOG" >&2
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] config.yaml [config2.yaml ...]"
    echo ""
    echo "Options:"
    echo "  --use-docker           Run benchmarks in Docker container"
    echo "  --docker-image IMAGE   Docker image to use (default: crispyx-benchmark:latest)"
    echo "  --build-docker         Build Docker image before running"
    echo "  --force                Force re-run all methods (ignore cache)"
    echo "  --clean                Delete output directory before running"
    echo "  --methods METHODS      Comma-separated list of methods to run (e.g., crispyx,pydeseq2)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 config/Adamson.yaml                         # Single dataset (native)"
    echo "  $0 --use-docker config/Adamson.yaml            # Single dataset (Docker)"
    echo "  $0 --use-docker --build-docker config/*.yaml   # Build and run all"
    echo "  $0 config/Adamson.yaml config/Frangieh.yaml    # Multiple datasets"
    echo "  $0 config/*.yaml                               # All datasets"
    echo "  $0 --methods crispyx,pydeseq2 config/*.yaml    # Run specific methods only"
}

# Parse arguments
CONFIGS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --use-docker)
            USE_DOCKER=true
            EXTRA_ARGS+=("--use-docker")
            shift
            ;;
        --docker-image)
            DOCKER_IMAGE="$2"
            EXTRA_ARGS+=("--docker-image" "$2")
            shift 2
            ;;
        --build-docker)
            BUILD_DOCKER=true
            EXTRA_ARGS+=("--build-docker")
            shift
            ;;
        --force)
            EXTRA_ARGS+=("--force")
            shift
            ;;
        --clean)
            EXTRA_ARGS+=("--clean")
            shift
            ;;
        --methods)
            # Convert comma-separated to space-separated for Python argparse
            METHODS_STR="${2//,/ }"
            # shellcheck disable=SC2206
            EXTRA_ARGS+=("--methods" $METHODS_STR)
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            # Config file argument
            CONFIGS+=("$1")
            shift
            ;;
    esac
done

# Check if config files provided
if [ ${#CONFIGS[@]} -eq 0 ]; then
    log_error "No config files provided"
    show_usage
    exit 1
fi

# Start logging
log "=========================================="
log "Benchmark Started"
log "=========================================="
log "Script directory: $SCRIPT_DIR"
log "Project root: $PROJECT_ROOT"
log "Python environment: $PYTHON_ENV"
if [ "$USE_DOCKER" = true ]; then
    log "Docker mode: enabled"
    [ -n "$DOCKER_IMAGE" ] && log "Docker image: $DOCKER_IMAGE"
fi
log "Log file: $MAIN_LOG"
log "Number of datasets: ${#CONFIGS[@]}"
log ""

# Convert relative paths to absolute paths before changing directory
ABS_CONFIGS=()
for arg in "${CONFIGS[@]}"; do
    # Convert to absolute path if relative
    if [[ "$arg" = /* ]]; then
        ABS_CONFIGS+=("$arg")
    else
        ABS_CONFIGS+=("$(cd "$(dirname "$arg")" && pwd)/$(basename "$arg")")
    fi
done

# Change to project root
cd "$PROJECT_ROOT"

# Track results
TOTAL_CONFIGS=${#ABS_CONFIGS[@]}
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_DATASETS=()

# Process each config file
CURRENT=0
for CONFIG_FILE in "${ABS_CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    log ""
    log "=========================================="
    log "Dataset $CURRENT/$TOTAL_CONFIGS: $(basename "$CONFIG_FILE")"
    log "=========================================="
    
    # Check if config exists
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_DATASETS+=("$(basename "$CONFIG_FILE") - File not found")
        continue
    fi
    
    log "Config: $CONFIG_FILE"
    
    # Create dataset-specific log
    DATASET_LOG="${LOG_DIR}/${TIMESTAMP}_$(basename "$CONFIG_FILE" .yaml).log"
    log "Dataset log: $DATASET_LOG"
    
    # Run benchmark with extra arguments (Docker flags, force-rerun, etc.)
    if "$PYTHON_ENV" -m benchmarking.tools.run_benchmarks --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}" > "$DATASET_LOG" 2>&1; then
        log "✓ Benchmark completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        EXIT_CODE=$?
        log_error "Benchmark failed with exit code: $EXIT_CODE"
        log_error "Check detailed log: $DATASET_LOG"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_DATASETS+=("$(basename "$CONFIG_FILE") - Exit code $EXIT_CODE")
    fi
done

# Final summary
log ""
log "=========================================="
log "Benchmark Completed"
log "=========================================="
log "Total datasets: $TOTAL_CONFIGS"
log "Successful: $SUCCESS_COUNT"
log "Failed: $FAILED_COUNT"

if [ $FAILED_COUNT -gt 0 ]; then
    log ""
    log "Failed datasets:"
    for failed in "${FAILED_DATASETS[@]}"; do
        log "  ✗ $failed"
    done
fi

log ""
log "Results location: benchmarking/results/"
log "Main log: $MAIN_LOG"
log "Dataset logs: ${LOG_DIR}/${TIMESTAMP}_*.log"
log ""
log "=========================================="

# Exit with error if any failed
if [ $FAILED_COUNT -gt 0 ]; then
    exit 1
fi
