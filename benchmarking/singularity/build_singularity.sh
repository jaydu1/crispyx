#!/bin/bash
# Build Singularity image from local Docker image
#
# Prerequisites:
#   1. Docker image must be built: docker build -t crispyx-benchmark -f benchmarking/Dockerfile .
#   2. Singularity must be installed (see install instructions below)
#
# Usage:
#   ./benchmarking/singularity/build_singularity.sh
#
# To install Singularity on Ubuntu/Debian:
#   # Install dependencies
#   sudo apt-get update && sudo apt-get install -y \
#       build-essential libseccomp-dev pkg-config squashfs-tools cryptsetup \
#       golang-go uidmap
#   
#   # Or use the pre-built package (easier):
#   sudo apt-get install -y singularity-container
#
# For other systems, see: https://docs.sylabs.io/guides/latest/admin-guide/installation.html

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
SIF_FILE="$OUTPUT_DIR/crispyx-benchmark.sif"

echo "=== Building Singularity image from local Docker image ==="
echo "Project root: $PROJECT_ROOT"
echo "Output file: $SIF_FILE"

# Check if Docker image exists
if ! docker image inspect crispyx-benchmark:latest &> /dev/null; then
    echo "ERROR: Docker image 'crispyx-benchmark:latest' not found."
    echo "Build it first with:"
    echo "  cd $PROJECT_ROOT"
    echo "  DOCKER_BUILDKIT=1 docker build -t crispyx-benchmark -f benchmarking/Dockerfile ."
    exit 1
fi

# Check if Singularity is installed
if ! command -v singularity &> /dev/null; then
    echo "ERROR: Singularity is not installed."
    echo "Install with: sudo apt-get install -y singularity-container"
    echo "Or see: https://docs.sylabs.io/guides/latest/admin-guide/installation.html"
    exit 1
fi

# Remove existing SIF file if present
if [ -f "$SIF_FILE" ]; then
    echo "Removing existing SIF file..."
    rm -f "$SIF_FILE"
fi

# Convert Docker image to Singularity SIF
echo "Converting Docker image to Singularity..."
singularity build "$SIF_FILE" docker-daemon://crispyx-benchmark:latest

echo ""
echo "=== SUCCESS ==="
echo "Singularity image created: $SIF_FILE"
echo ""
echo "To transfer to HPC:"
echo "  scp $SIF_FILE user@hpc:/path/to/destination/"
echo ""
echo "To run on HPC:"
echo "  singularity exec --bind /data:/workspace/data crispyx-benchmark.sif python -m benchmarking.tools.run_benchmarks --config config.yaml"
