#!/bin/bash
# CRISPyx Benchmark Runner
# Runs benchmarks on one or more datasets
#
# Usage:
#   ./run_benchmark.sh config.yaml                    # Single dataset
#   ./run_benchmark.sh config1.yaml config2.yaml      # Multiple datasets
#   ./run_benchmark.sh config/*.yaml                  # All datasets

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_ENV="${PYTHON_ENV:-$(command -v python)}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs"
MAIN_LOG="${LOG_DIR}/${TIMESTAMP}_benchmark.log"

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

# Check if config files provided
if [ $# -eq 0 ]; then
    log_error "No config files provided"
    echo "Usage: $0 config.yaml [config2.yaml ...]"
    echo ""
    echo "Examples:"
    echo "  $0 config/Adamson.yaml                    # Single dataset"
    echo "  $0 config/Adamson.yaml config/Frangieh.yaml  # Multiple datasets"
    echo "  $0 config/*.yaml                          # All datasets"
    exit 1
fi

# Start logging
log "=========================================="
log "Benchmark Started"
log "=========================================="
log "Script directory: $SCRIPT_DIR"
log "Project root: $PROJECT_ROOT"
log "Python environment: $PYTHON_ENV"
log "Log file: $MAIN_LOG"
log "Number of datasets: $#"
log ""

# Convert relative paths to absolute paths before changing directory
CONFIGS=()
for arg in "$@"; do
    # Convert to absolute path if relative
    if [[ "$arg" = /* ]]; then
        CONFIGS+=("$arg")
    else
        CONFIGS+=("$(cd "$(dirname "$arg")" && pwd)/$(basename "$arg")")
    fi
done

# Change to project root
cd "$PROJECT_ROOT"

# Track results
TOTAL_CONFIGS=${#CONFIGS[@]}
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_DATASETS=()

# Process each config file
CURRENT=0
for CONFIG_FILE in "${CONFIGS[@]}"; do
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
    
    # Run benchmark
    if "$PYTHON_ENV" -m benchmarking.tools.run_benchmarks --config "$CONFIG_FILE" > "$DATASET_LOG" 2>&1; then
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
