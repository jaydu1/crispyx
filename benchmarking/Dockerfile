# Dockerfile for CRISPYx Benchmarking
# Provides reproducible environment with Python, R, and all dependencies
# 
# Build: docker build -t crispyx-benchmark benchmarking/
# Run:   docker run --rm -v $(pwd):/workspace crispyx-benchmark --help

FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="CRISPYx Benchmark"
LABEL org.opencontainers.image.description="Reproducible benchmarking environment for CRISPR screen analysis"
LABEL org.opencontainers.image.source="https://github.com/your-org/Streamlining-CRISPR-Screen-Analysis"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set locale to avoid R encoding issues
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for R and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # R and build tools
    r-base \
    r-base-dev \
    # Build dependencies for Python packages
    build-essential \
    gcc \
    g++ \
    gfortran \
    # HDF5 for h5py/anndata
    libhdf5-dev \
    pkg-config \
    # Other dependencies
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    # Git for pip installs from repos
    git \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Bioconductor and R packages
# Pin versions for reproducibility
# Note: R 4.2 from Debian bookworm requires Bioconductor 3.16
RUN R -e "install.packages('BiocManager', repos='https://cloud.r-project.org/')" \
    && R -e "BiocManager::install(version='3.16', ask=FALSE)" \
    && R -e "BiocManager::install(c('edgeR', 'DESeq2', 'apeglm'), ask=FALSE, update=FALSE)" \
    && R -e "install.packages('RhpcBLASctl', repos='https://cloud.r-project.org/')" \
    # Cleanup R package cache
    && rm -rf /tmp/downloaded_packages /tmp/Rtmp*

# Create workspace directory
WORKDIR /workspace

# Install Python dependencies
# Copy only requirements first for better layer caching
COPY pyproject.toml /workspace/

# Install core dependencies first (for caching)
RUN pip install --no-cache-dir \
    "numpy>=1.23" \
    "scipy>=1.10,<=1.12.0" \
    "pandas>=1.5" \
    "anndata>=0.9" \
    h5py \
    "numba>=0.59" \
    "scanpy>=1.9.2" \
    "statsmodels>=0.14"

# Install DE/benchmark dependencies
RUN pip install --no-cache-dir \
    "pydeseq2>=0.4" \
    "pertpy>=0.4" \
    rpy2 \
    "psutil>=5.9" \
    "pyyaml>=6.0" \
    "tqdm>=4.65" \
    filelock

# Copy source code and install crispyx
COPY src/ /workspace/src/
COPY pyproject.toml /workspace/
RUN pip install --no-cache-dir -e /workspace

# Copy benchmarking code
COPY benchmarking/ /workspace/benchmarking/

# Copy worker entrypoint to /app directory
RUN mkdir -p /app && cp /workspace/benchmarking/tools/docker_worker.py /app/docker_worker.py

# Set working directory for benchmark execution
WORKDIR /workspace

# Default entrypoint runs the docker worker
ENTRYPOINT ["python", "/app/docker_worker.py"]

# Default command shows help
CMD ["--help"]
