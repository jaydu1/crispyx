# Development Scripts

This directory contains development, profiling, and debugging scripts used for benchmarking optimization research. These scripts are **not required** for production benchmarking.

## Scripts Overview

### Profiling Scripts

- **`profile_contiguous_io.py`**: I/O access pattern profiling
  - Analyzes contiguous vs. non-contiguous memory access patterns
  - Helps optimize data loading strategies

- **`profile_irls_computation.py`**: IRLS bottleneck analysis
  - Profiles Iteratively Reweighted Least Squares computation
  - Identifies performance bottlenecks in GLM fitting

- **`profile_nb_glm.py`**: NB-GLM step-level timing
  - Detailed timing analysis of Negative Binomial GLM steps
  - Helps optimize the core statistical model

### Debugging Scripts

- **`debug_lfcshrink.py`**: lfcShrink discrepancy debugging
  - Investigates discrepancies in log-fold change shrinkage
  - Compares results between different implementations

### Benchmarking Scripts

- **`benchmark_frozen_control.py`**: Frozen control optimization benchmark
  - Tests optimization strategies for frozen control genes
  - Validates performance improvements

- **`benchmark_numba_kernels.py`**: Numba kernel performance testing
  - Compares performance of different Numba-accelerated kernels
  - Helps select optimal implementations

## Usage

These scripts can be run directly from the workspace root:

```bash
cd /home/jinhongd/Streamlining-CRISPR-Screen-Analysis

# Example: Profile NB-GLM on a dataset
python -m benchmarking.dev.profile_nb_glm data/Adamson_subset.h5ad

# Example: Debug lfcShrink discrepancies
python -m benchmarking.dev.debug_lfcshrink
```

## Notes

- These scripts import from `benchmarking.tools` for core functionality
- They are independent of the main benchmarking pipeline
- Results are typically saved to dated directories (e.g., `20260109/`)
