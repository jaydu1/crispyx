# Changelog

All notable changes to crispyx are documented here.

## [0.6.0] - 2026-01

### Breaking Changes
- **Removed `shrink_logfc` parameter**: The deprecated `shrink_logfc` parameter has been 
  removed from `nb_glm_test()`. Use the two-step workflow instead:
  ```python
  result = cx.nb_glm_test(path, perturbation_column="perturbation")
  shrunk = cx.shrink_lfc(result.result_path)
  ```

### Fixed
- **t-test p-value calculation**: Changed from normal distribution approximation to proper 
  t-distribution with Welch-Satterthwaite degrees of freedom, matching Scanpy's implementation.
  P-value overlap with Scanpy improved from 0.975-0.996 to 1.000 across all top-k thresholds.
- **t-test/Wilcoxon HDF5 write error for large datasets**: Fixed "object header message is 
  too large" error when saving results for datasets with many perturbation groups (e.g., 2000+).
  Changed from recarray-based `rank_genes_groups` storage to layer-based AnnData storage,
  consistent with `nb_glm_test` output format.
- **NB-GLM memory management**: Improved memory estimation for parallel workers to account
  for control cache serialization overhead (1.5× factor). Added `gc.collect()` after freeing
  global dispersion matrix to ensure memory is released before spawning joblib workers.
- **Benchmark subprocess fork/OpenMP conflict**: Fixed "fork() called from a process already 
  using GNU OpenMP" error for wilcoxon_test in benchmarks. Changed spawn context to cover 
  all crispyx DE methods that use Numba kernels.
- **Numerical warnings suppressed**: Fixed divide-by-zero warning in t-test Welch df 
  calculation and "Mean of empty slice" warning in shrink_lfc.

### Added
- **`scanpy_format` parameter for DE tests**: Added optional `scanpy_format: bool = False` 
  parameter to `t_test()`, `wilcoxon_test()`, and `nb_glm_test()`. When True, writes 
  Scanpy-compatible `uns["rank_genes_groups"]` structure for interoperability with 
  `sc.get.rank_genes_groups_df()` and similar Scanpy utilities.
- **`tl.shrink_lfc()` namespace method**: Added `cx.tl.shrink_lfc()` for API consistency 
  with other tools. Equivalent to calling `cx.shrink_lfc()` directly.
- **Resume/checkpoint documentation**: Added comprehensive documentation for `resume` and 
  `checkpoint_interval` parameters in usage guide and README.

### Improved
- **QC performance for dense datasets**: Added numba-accelerated dense→CSR conversion 
  achieving 60× speedup for the write phase. Replogle-E-k562 (310K cells) QC improved 
  from 122s to 54s (2.3× faster).
- **Adaptive QC algorithm**: Automatically detects dense vs sparse storage format and 
  routes to optimized code paths. Sparse datasets use in-memory CSR caching; dense 
  datasets avoid expensive format conversion.
- **`wilcoxon_test` docstring**: Complete parameter documentation for all 14 parameters
  including `min_cells_expressed`, `chunk_size`, `tie_correct`, `n_jobs`, `resume`, etc.
- **Tutorial notebook**: Added NB-GLM and LFC shrinkage sections demonstrating the 
  two-step workflow.
- **Usage guide**: Expanded NB-GLM options section with shrink_lfc examples and 
  prior_scale_mode documentation.

### Refactored
- **Benchmarking tools module**: Reorganized `benchmarking/tools/` into focused modules:
  - `constants.py`: Centralized constants (method names, display order, cache version)
  - `cache.py`: Cache I/O functions (save/load results, path resolution)
  - `formatting.py`: Display formatting (method names, emoji indicators, markdown tables)
  - Updated `comparison.py`, `visualization.py`, `generate_results.py`, `run_benchmarks.py` 
    to import from shared modules
  - Removed ~600 lines of duplicate code across the module
- **QC module cleanup**: Removed ~159 lines of dead/duplicate code from `qc.py`:
  - Removed unused `_filter_genes_and_count_nnz` function (superseded by optimized paths)
  - Consolidated duplicate `_calculate_qc_chunk_size` to use shared `calculate_optimal_chunk_size` from `data.py`
- **NB-GLM cleanup**: Removed redundant `peak_rss` logging at end of `nb_glm_test()` 
  (superseded by the `profiling` parameter which provides proper memory tracking).

## [0.5.0] - 2026-01

### Fixed
- **apeGLM LFC shrinkage accuracy**: Corrected NB negative log-likelihood formulation in `_fit_gene_apeglm_lbfgsb()` - now uses clamped linear predictor (`eta_clamped`) instead of raw linear predictor (`xbeta`). Correlation with PyDESeq2 improved from ρ ≈ 0.16 to ρ > 0.94.

### Removed
- **Joint NB-GLM fitting mode**: Removed ~1,600 lines of experimental joint model code
  - `fit_method` parameter removed from `nb_glm_test()` (only "independent" fitting remains)
  - Removed classes: `JointControlCache`, `SufficientStatsCache`, `JointModelResult`
  - Removed functions: `estimate_joint_model_lbfgsb()`, `compute_sufficient_stats_streaming()`
  - Removed kernels: `_accumulate_perturbation_blocks_numba()`, `_batch_schur_solve_numba()`
  - Removed benchmark method: `crispyx_de_nb_glm_joint`

### Migration
- Remove `fit_method` argument from `nb_glm_test()` calls (no longer supported)
- Clear cached joint results: `rm benchmarking/results/*/crispyx_de_nb_glm_joint*.h5ad`

## [0.4.0] - 2025-12

### Changed
- `size_factor_scope` now defaults to `"global"` (was `"per_comparison"`) for better memory efficiency
- `use_control_cache=True` is now the default for NB-GLM tests
- NB-GLM uses L-BFGS-B optimization only (removed IRLS optimizer)
- Numba kernels extracted to `crispyx/_kernels.py` for better modularity

### Removed
- `optimization_method` parameter from `NBGLMFitter` - L-BFGS-B is the only optimizer
- `use_sparse`, `use_numba`, `joint_optimizer` parameters from `nb_glm_test()`
- `estimate_joint_model_streaming()` function - use `estimate_joint_model_lbfgsb()` instead
- Sparsity utility functions (inlined): `compute_sparsity()`, `should_use_sparse_ops()`, `sparse_row_sums()`, `sparse_col_sums()`, `sparse_mean()`

## [0.3.0] - 2025-11

### Changed
- All crispyx output files now use `crispyx_` prefix (e.g., `crispyx_qc_filtered.h5ad`)
- Benchmark method names updated to match output filenames
- Output directory structure reorganized with `preprocessing/`, `de/`, and `comparisons/` subdirectories

### Migration
- Clear old benchmark results: `rm -rf benchmarking/results/*`
- Re-run benchmarks to generate files with new naming structure

## [0.2.0] - 2024

### Changed
- `wald_test` renamed to `t_test` for consistency with Scanpy
- Backward compatibility: `method="wald"` automatically maps to `t_test`

### Added
- `nb_glm_test` for negative binomial GLM differential expression
- Lazy-loading GLM comparisons: edgeR and PyDESeq2 automatically use `nb_glm` results
- Standardized comparison CSV format with `.benchmark_cache/` directory
