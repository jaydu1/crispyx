# Singularity Setup for HPC

This directory contains scripts for running CRISPYx benchmarks on HPC systems using Singularity.

## Quick Start

### 1. Build the Docker Image (Local Machine)

```bash
cd /path/to/Streamlining-CRISPR-Screen-Analysis
DOCKER_BUILDKIT=1 docker build -t crispyx-benchmark -f benchmarking/Dockerfile .
```

### 2. Convert to Singularity (Local Machine)

**Option A: If Singularity is installed locally:**

```bash
chmod +x benchmarking/singularity/build_singularity.sh
./benchmarking/singularity/build_singularity.sh
```

This creates `benchmarking/singularity/crispyx-benchmark.sif`

**Option B: If Singularity is NOT installed locally:**

Save Docker image as tarball and convert on HPC:

```bash
# On local machine - save Docker image
docker save crispyx-benchmark:latest -o crispyx-benchmark.tar

# Transfer to HPC
scp crispyx-benchmark.tar user@hpc:/path/to/destination/

# On HPC - convert to SIF
singularity build crispyx-benchmark.sif docker-archive://crispyx-benchmark.tar

# Clean up tarball (optional)
rm crispyx-benchmark.tar
```

### 3. Transfer to HPC

```bash
# Transfer SIF file
scp benchmarking/singularity/crispyx-benchmark.sif user@hpc:/path/to/project/benchmarking/singularity/

# Transfer scripts
scp benchmarking/singularity/*.sh user@hpc:/path/to/project/benchmarking/singularity/

# Transfer config files
scp -r benchmarking/config user@hpc:/path/to/project/benchmarking/

# Transfer data
scp -r data user@hpc:/path/to/project/
```

### 4. Run on HPC

**Interactive mode:**

```bash
cd /path/to/project/benchmarking/singularity
chmod +x run_singularity.sh
./run_singularity.sh config/Adamson.yaml
```

**SLURM batch job:**

```bash
cd /path/to/project/benchmarking/singularity
sbatch slurm_benchmark.sh Adamson.yaml

# With custom resources
sbatch --mem=128G --cpus-per-task=16 slurm_benchmark.sh Replogle_K562.yaml
```

## File Structure

```
benchmarking/singularity/
├── README.md              # This file
├── build_singularity.sh   # Script to build SIF from Docker image
├── run_singularity.sh     # Script to run benchmarks with Singularity
├── slurm_benchmark.sh     # SLURM job submission script
└── crispyx-benchmark.sif  # Singularity image (after building)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CRISPYX_SIF` | `./crispyx-benchmark.sif` | Path to Singularity image |
| `DATA_DIR` | `../data` | Path to data directory |
| `RESULTS_DIR` | `./results` | Path to results directory |
| `N_CORES` | `8` | Number of CPU cores |

## Bind Mounts

The scripts mount the following directories:

| Host Path | Container Path | Mode |
|-----------|----------------|------|
| `$DATA_DIR` | `/workspace/data` | Read-only |
| `$RESULTS_DIR` | `/workspace/benchmarking/results` | Read-write |
| `$LOGS_DIR` | `/workspace/benchmarking/logs` | Read-write |
| `$CONFIG_DIR` | `/workspace/benchmarking/config` | Read-only |

## Troubleshooting

### "Singularity image not found"

Make sure the `.sif` file is in the correct location:
```bash
export CRISPYX_SIF=/path/to/crispyx-benchmark.sif
```

### Permission issues with bind mounts

Some HPC systems require explicit bind paths in `singularity.conf`. Contact your HPC admin if you see "permission denied" errors.

### R packages not found

The Singularity image includes all R packages (edgeR, DESeq2, apeglm). If you see errors, the image may need rebuilding.

### Out of memory

Increase memory allocation:
```bash
sbatch --mem=128G slurm_benchmark.sh config/large_dataset.yaml
```

## Comparison: Docker vs Singularity

| Feature | Docker (local) | Singularity (HPC) |
|---------|----------------|-------------------|
| Image format | Layered filesystem | Single `.sif` file |
| Privileges | Requires root/docker group | No root needed |
| GPU support | `--gpus` flag | `--nv` flag |
| Networking | Isolated by default | Uses host network |
| User namespace | Runs as root inside | Runs as your user |
