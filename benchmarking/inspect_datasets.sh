#!/bin/bash
# Dataset Inspection Script for CRISPyx
# Auto-detects dataset structures and generates individual config files
#
# Usage:
#   ./inspect_datasets.sh [--memory-limit GB] [--reference-config PATH]

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_ENV="/data/miniforge3/envs/pert/bin/python"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/inspect_datasets_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to log errors
log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

# Start logging
log "=========================================="
log "Dataset Inspection Started"
log "=========================================="
log "Script directory: $SCRIPT_DIR"
log "Project root: $PROJECT_ROOT"
log "Python environment: $PYTHON_ENV"
log "Log file: $LOG_FILE"
log ""

# Change to project root
cd "$PROJECT_ROOT"
log "Changed to project directory: $(pwd)"

# Build command with optional arguments
INSPECT_CMD="$PYTHON_ENV benchmarking/inspect_datasets.py"

# Pass through all arguments
if [ $# -gt 0 ]; then
    log "Arguments: $@"
    INSPECT_CMD="$INSPECT_CMD $@"
fi

# Run inspection with output to both console and log file
log ""
log "=========================================="
log "Running Dataset Inspection"
log "=========================================="
log ""

if $INSPECT_CMD 2>&1 | tee -a "$LOG_FILE"; then
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -eq 0 ]; then
        log ""
        log "=========================================="
        log "✓ Dataset inspection completed successfully!"
        log "=========================================="
        log ""
        log "Generated config files in: benchmarking/config/"
        log "Log file: $LOG_FILE"
        log ""
        log "Next steps:"
        log "  1. Review configs: ls -1 benchmarking/config/*.yaml"
        log "  2. Run single dataset: ./run_benchmark.sh config/Adamson.yaml"
        log "  3. Run multiple datasets: ./run_benchmark.sh config/*.yaml"
        log ""
        log "=========================================="
    else
        log_error "Dataset inspection failed with exit code: $EXIT_CODE"
        exit $EXIT_CODE
    fi
else
    EXIT_CODE=$?
    log_error "Dataset inspection failed with exit code: $EXIT_CODE"
    log_error "Check log for details: $LOG_FILE"
    exit $EXIT_CODE
fi
