# Changelog

All notable changes to crispyx are documented here.

## [0.7.3] - 2026-01-31

### Added
- **Scanpy-style plotting namespace (`cx.pl`)** with on-disk safe helpers for DE and QC plots.
- **On-demand materialization of `uns['rank_genes_groups']`** for plotting without loading counts.
- **DE plotting utilities**: rank-genes plot wrapper, volcano, MA (raw or normalized log1p means), and top-genes bar plots.
- **QC plotting utilities**: perturbation composition and QC summary distributions from `QualityControlResult`.

## [0.7.2] - 2026-01-20

### Added
- **Streaming preprocessing with `cx.pp.normalize_total_log1p()`**: New function for 
  normalizing and log-transforming large h5ad files without loading them into memory.
  - Processes data in chunks (default 4096 cells per chunk)
  - Equivalent to `scanpy.pp.normalize_total()` + `scanpy.pp.log1p()`
  - Supports separate `normalize` and `log1p` flags for flexibility
  - Custom `target_sum` parameter (default 1e4)
  - Scanpy-style API: accepts AnnData objects or paths, returns AnnData wrapper
  - Example: `adata_norm = cx.pp.normalize_total_log1p(adata_ro, output_dir=OUTPUT_DIR)`
  - Also available as direct function: `from crispyx.data import normalize_total_log1p`
  - Required for t-test and Wilcoxon on large datasets that would OOM during preprocessing

- **Unified input resolution with `resolve_data_path()`**: New utility function in
  `crispyx.data` that standardizes input handling across all main crispyx functions.
  - Accepts `str | Path | crispyx.AnnData | anndata.AnnData`
  - Raises clear `TypeError` for in-memory (non-backed) AnnData objects
  - Optional `require_exists` parameter for path validation
  - All QC, DE, and pseudobulk functions now accept AnnData objects directly

### Changed
- **Flexible input types for all main functions**: The following functions now accept
  either a file path or an AnnData object as their first argument:
  - QC: `filter_cells_by_gene_count()`, `filter_perturbations_by_cell_count()`,
    `filter_genes_by_cell_count()`, `quality_control_summary()`
  - DE: `t_test()`, `wilcoxon_test()`, `nb_glm_test()`, `shrink_lfc()`
  - Pseudobulk: `compute_average_log_expression()`, `compute_pseudobulk_expression()`
  - Preprocessing: `normalize_total_log1p()`

### Fixed
- **Singularity cgroups v2 compatibility**: Removed `--memory` flag from Singularity exec 
  in SLURM scripts. Many HPC clusters don't support cgroups v2 in unified mode. Memory 
  limits are now enforced by SLURM's `--mem` allocation instead.

## [0.7.1] - 2026-01-14

### Fixed
- **QC in-memory path memory optimization**: Reduced peak memory for `_qc_in_memory()` 
  by eliminating intermediate copies. Previously made 3 sequential `.copy()` calls 
  (cell filter → perturbation filter → gene filter); now builds combined masks first 
  and performs a single copy at the end. This matches Scanpy's memory efficiency for 
  in-memory QC operations.

### Changed
- **Adaptive chunk size defaults**: Updated `calculate_optimal_chunk_size()` defaults 
  for more conservative memory usage:
  - `max_chunk`: 8192 → 4096 (halved for smaller memory footprint per chunk)
  - `safety_factor`: 4.0 → 8.0 (doubled for more headroom)
- **New `calculate_optimal_gene_chunk_size()`**: Added function for column-based 
  operations (e.g., Wilcoxon test) with gene-aware chunking strategy.
- **Pseudobulk and DE use adaptive chunk sizes**: `average_log_expression()`, `t_test()`, 
  and `wilcoxon_test()` now use `calculate_optimal_chunk_size()` when chunk_size is None, 
  instead of hardcoded defaults.

### Improved
- **Code organization**: Moved inline imports (`gc`, `anndata`, `Counter`) to module-level 
  imports in `qc.py` for cleaner code structure.

## [0.7.0] - 2026-01

### Added
- **Memory-adaptive streaming for large datasets**: `nb_glm_test()` now automatically 
  detects when the full cell×gene matrix would exceed available memory and switches to 
  streaming mode. For Replogle-GW-k562 (2M cells × 8K genes = ~131 GB), this avoids OOM 
  when memory limit is 128 GB.
  - Early memory check before loading `full_X` or `all_cell_matrix`
  - `precompute_global_dispersion_from_path()`: New streaming function that reads chunks 
    directly from h5ad file, never loading the full matrix into memory
  - Falls back to global size factors when per-comparison SF would require loading full matrix
  - Threshold: `estimated_matrix_gb > max_dense_fraction × available_memory`

- **`freeze_control` parameter for massive memory reduction**: New `freeze_control` 
  parameter for `nb_glm_test()` pre-computes frozen sufficient statistics (W_sum, Wz_sum) 
  for control cells during initialization, eliminating the need to pass the control matrix 
  to each worker. This achieves **43× memory reduction** per worker, enabling full 
  parallelization on large datasets:
  - Replogle-GW-k562: Per-worker memory reduced from 34.6 GB to 0.8 GB
  - Workers increased from 2 → 32 on 128 GB memory limit
  - Estimated speedup from ~680 hours to 3-5 hours (100× faster)
  
  **Auto-detection (default)**: When `freeze_control=None` (default), the function
  automatically enables frozen control mode when:
  - Control matrix serialization would limit workers to <4
  - Required settings are met (`dispersion_scope='global'`, `shrink_dispersion=True`)
  
  This means large datasets get optimal parallelization without user intervention.
  Results are mathematically equivalent to standard mode (LFC correlation >0.999).
  ```python
  # Auto-detection (recommended for most cases)
  result = cx.nb_glm_test(
      "large_dataset.h5ad",
      perturbation_column="perturbation",
      n_jobs=32,  # freeze_control auto-enabled if beneficial
  )
  
  # Explicit enable (for testing or forcing)
  result = cx.nb_glm_test(
      "large_dataset.h5ad",
      perturbation_column="perturbation",
      dispersion_scope="global",   # Required for freeze_control
      shrink_dispersion=True,      # Required for freeze_control
      freeze_control=True,         # Explicit enable
      n_jobs=32,
  )
  ```

- **`sort_by_perturbation()` for I/O optimization**: New function to reorder cells by 
  perturbation label, enabling contiguous reads instead of random access. For large datasets 
  like Replogle-GW-k562 (2M cells, 10K perturbations), this provides **46× I/O speedup** 
  per perturbation read and enables efficient parallelization on HDD storage.
  ```python
  sorted_path = cx.sort_by_perturbation(
      "large_dataset.h5ad",
      perturbation_column="perturbation",
  )
  # Creates large_dataset_sorted.h5ad with cells grouped by perturbation
  ```
- **Automatic sorting in `nb_glm_test()`**: Large datasets are now automatically sorted 
  before NB-GLM fitting when beneficial. The function checks if sorting is needed based on:
  - Dataset has ≥360K cells (~1 hour I/O overhead on HDD at 100 IOPS)
  - Dataset has ≥100 perturbations (sufficient parallel workload)
  - Cell contiguity is below 50% (cells scattered across file)
  
  This triggers sorting for 3 benchmark datasets: Replogle-GW-k562 (5.5h → 0.1h), 
  Feng-ts (3.2h → 0.1h), and Feng-gwsnf (1.1h → 0.1h).

### Changed
- **Benchmark default `size_factor_method` changed from `"deseq2"` to `"sparse"`**: The 
  `"deseq2"` method (genes expressed in ALL cells) provides no benefit for sparse single-cell 
  data since it always falls back to `"sparse"` when too few genes qualify. For Replogle-GW-k562, 
  the old code wasted 51 minutes checking before falling back. The new default matches 
  crispyx's library default and is optimal for scRNA-seq.

### Fixed
- **`_deseq2_style_size_factors` 1000× speedup**: Vectorized the all-expressed gene check 
  from O(n_cells × n_genes) nested Python loop to O(nnz + n_genes) sparse column operations. 
  For Replogle-GW-k562 (2M cells × 8K genes), this reduces size factor computation from 
  **51 minutes to ~3 seconds**. Added early termination when <10 genes are expressed in all 
  cells (detected after first chunk), immediately falling back to sparse method.
- **Docker container timezone mismatch**: Fixed timestamps in `.progress.json` checkpoint 
  files showing UTC time instead of local time when running benchmarks in Docker containers.
  Added `/etc/localtime` mount to `docker-compose.yml` and `DockerRunner` to sync container 
  timezone with host.
- **`--force` with `--methods` now preserves cached results**: When using `--force` with 
  specific `--methods`, only the cache for those methods is cleared. Cached results for 
  other methods are preserved and included in the final benchmark report. Previously, 
  `--force` would clear the entire cache, causing reports to only show the re-run methods.
- **NB-GLM memory estimation for joblib pickle overhead**: Fixed critical under-estimation
  of per-worker memory when using joblib's loky backend. The loky backend serializes (pickles)
  all function arguments for each worker process, meaning control_matrix is copied to each
  worker, not shared via copy-on-write. Updated calculation to account for:
  - 2.5× pickle serialization overhead (vs 1.5× previously)
  - 4× work arrays for SE recomputation (vs 2× previously)
  - 2 GB Python/process overhead (vs 200 MB previously)
  For Replogle-GW-k562, this reduces workers from 32 → 5 to stay under 128 GB memory limit.
- **`needs_sorting_for_nbglm()` now checks for existing sorting**: Fixed function that would
  recommend sorting even for already-sorted files. Now checks for `sorting_metadata` in 
  `adata.uns` before recommending sorting, avoiding unnecessary re-sorting operations.

### Improved
- **NB-GLM adaptive memory estimation**: Complete rewrite of memory-aware worker limiting
  to use dataset statistics computed from metadata without loading the full matrix:
  - Control cache counted as shared (copy-on-write) base memory, not per-worker
  - Context-aware `n_work_arrays`: 2 for global dispersion mode, 6 for per-comparison
  - P95 group size for realistic memory estimates (avoids over-conservative from outliers)
  - 200 MB minimum floor per worker for Python/loky process overhead
  - 20% headroom reserve for system overhead and GC
  - Dataset-size-aware caps: n_perts/2 for tiny datasets (<1 GB) to reduce parallelization overhead
  - Works for all dataset sizes from 0.16 GB (Adamson_subset) to 339 GB (Feng-ts)

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
