# Changelog

All notable changes to crispyx are documented here.

## [0.7.4] - 2026-02-21

### Fixed (2026-02-24 — Wilcoxon OOM patch)
- **h5py direct write for Wilcoxon results**: Replaced the end-of-function triple
  allocation (memmap → `np.array()` copy → `AnnData.write()`) with `_write_wilcoxon_result_h5ad()`,
  which writes each memmap directly to HDF5 via h5py one dataset at a time. This avoids
  having the full memmap, its in-memory copy, and the AnnData write buffers all resident
  simultaneously — the primary OOM cause for datasets like Feng-gwsf (2,031 groups × 31,772 genes).

- **glibc malloc_trim after each gene chunk**: Added `_release_chunk_memory()` (gc.collect +
  `malloc_trim(0)`) after each gene chunk's memmap writes. On Linux, glibc holds freed pages
  in thread-local arenas; `malloc_trim(0)` forces return to the OS, preventing RLIMIT_AS
  violations that accumulated across hundreds of gene chunks.

- **Adaptive per-chunk budget cap**: Added a 5%-of-available-memory cap per gene chunk in
  `calculate_optimal_gene_chunk_size()`. For high-cell datasets (e.g., Feng-ts at 1.16M cells)
  this limits chunk size to ~460 genes instead of 512, preventing transient dense-block
  allocations from exceeding the per-chunk memory budget.

- **Lazy result readback for huge output**: `_build_result_from_h5ad()` returns a lazy result
  (empty arrays + AnnData file reference) when the result arrays would exceed 25% of physical
  memory. This prevents OOM during the readback phase for extreme datasets like Huang-HEK293T
  (18,311 groups × 38,606 genes ≈ 40 GB of result arrays).

- **Peak multiplier increased to 5.0×**: `_should_use_streaming()` now uses
  `peak_multiplier=5.0` (was 4.0) to account for glibc arena overhead and freed-but-unreturned
  pages, improving streaming dispatch accuracy for borderline datasets.

- **Removed unused `order_matrix` memmap**: The non-streaming Wilcoxon path no longer
  allocates an `(n_groups × n_genes)` int64 memmap for sort order. Order is now computed
  on-demand during result readback via `np.argsort`, saving ~500 MB for large datasets.

- **Benchmark runner: skip RLIMIT_AS for mmap-backed methods**: Inside Singularity
  containers (where cgroups are unavailable), the benchmark runner previously enforced
  memory limits via `RLIMIT_AS`, which caps total virtual address space (VSZ) — not RSS.
  crispyx methods mmap large h5ad files and numpy memmaps that inflate VSZ to ~42 GB
  while RSS stays at 6–17 GB, causing spurious SIGKILL. Added
  `_uses_mmap_backed_files()` helper to skip `RLIMIT_AS` for crispyx methods; memory
  enforcement relies on parent-process RSS sampling instead.

### Performance
- **Wilcoxon dense-block once per chunk**: Both the standard and streaming paths now
  convert the relevant gene columns to dense **once** per chunk (for all cells combined)
  instead of once per perturbation group. This eliminates ≥ N_groups × scipy sparse→dense
  conversions inside the inner loop — for Feng-gwsnf (4,955 groups) that removes ~1.4 M
  sparse→dense calls per gene chunk, yielding a **~6× I/O speedup** (17 GB read in
  ~2 min vs ~12 min before).

- **Integer row indexing in Wilcoxon**: Replaced boolean-masked dense extraction
  (`csr[mask, :]`) with precomputed integer arrays (`np.where(mask)[0]`) so each group
  row-selection is a single O(k) fancy-index copy instead of a boolean scan + copy.

- **Pre-sorted control Numba kernels**: Added two new `@nb.njit(parallel=True, cache=True)`
  kernels to `_kernels.py`:
  - `_presort_control_nonzeros`: sorts control non-zeros once per gene chunk using
    `nb.prange` over genes (two parallel passes: count → prefix-sum → extract+sort).
  - `_wilcoxon_presorted_ctrl_numba`: merge-based rank-sum test that reuses the
    pre-sorted control; eliminates the O(n_ctrl × log n_ctrl) sort that previously
    ran once per perturbation group × gene.
  
  Kernel microbenchmark (5,824 calls, Adamson scale): **7.38× faster** (27.6 s → 3.7 s).
  Applied to both standard and streaming paths in `de.py`.

- **Batched Wilcoxon Numba kernel (`_wilcoxon_batch_perts_presorted_numba`)**: New
  `@nb.njit(parallel=True, cache=True)` kernel that processes all perturbation groups
  in a single JIT call using `nb.prange` over `n_perts`. Replaced the sequential
  perturbation loop in both standard and streaming paths. Combined with
  `_wilcoxon_single_pert_presorted` for per-perturbation rank-sum logic.

- **Dense array dtype aligned with Scanpy (float32)**: Removed `.astype(np.float64)`
  casts from both standard and streaming Wilcoxon paths in `de.py`. Dense gene blocks
  are kept at the native h5ad dtype (float32), halving per-chunk working-set memory
  (~200 MB → ~100 MB at Huang scale). Output statistic arrays (`u_stats`, `pvals`)
  remain float64 for numerical precision, matching Scanpy's `_ranks()` behaviour.

### Fixed
- **Wilcoxon OOM on very large datasets (>300K cells)**: Cell-count caps in
  `calculate_optimal_gene_chunk_size()` are now gated on `available_memory_gb < 32.0`.
  On 128 GB benchmark servers the caps are lifted, allowing chunk sizes to reach
  384–512 genes/chunk (e.g., Feng-gwsf 322K cells → 384, Feng-ts 1.16M cells → 512).
  On memory-constrained systems (< 32 GB) the protective caps still apply:
  - >1M cells: max_chunk = 32 genes
  - >500K cells: max_chunk = 64 genes
  - >300K cells: max_chunk = 128 genes
  This retains OOM safety on small machines while removing the performance penalty on
  128 GB benchmark servers. Fixes Wilcoxon OOM on Feng-gwsf, Feng-gwsnf, and Feng-ts.

- **NB-GLM OOM on datasets with large control populations**: Enhanced `freeze_control`
  auto-detection to enable when control matrix exceeds 10 GB, regardless of worker count.
  For the Feng datasets (~110K control cells × 36K genes = 32 GB), each worker previously
  copied the full control matrix, causing OOM. Now `freeze_control` auto-enables,
  reducing per-worker memory from ~32 GB to <1 GB.
  
  Auto-enable conditions (either triggers freeze_control):
  1. **Large control matrix** (NEW): control_n × n_genes × 8 bytes > 10 GB
  2. **Worker limitation**: standard mode would limit to <4 workers
  
  This fixes NB-GLM OOM on Feng-gwsf, Feng-gwsnf, and Feng-ts datasets under 128 GB.

- **Scanpy DE "only one sample" errors (Feng-gwsf, Feng-gwsnf, Nadig-JURKAT)**:
  `scanpy_de_wilcoxon` was moved to use `preprocessed_path`, but `preprocessed_path`
  itself was built from the raw (unfiltered) standardized h5ad. Small singleton
  perturbation groups were still present, causing `sc.tl.rank_genes_groups()` to crash.
  Filtering inside individual DE functions was not viable as it distorts timing. Fixed by
  introducing a new `crispyx_preprocess` benchmark step that normalizes the QC-filtered
  output (`crispyx_qc_filtered.h5ad`) → `preprocessed_<dataset>.h5ad`. All four t-test /
  Wilcoxon DE methods now declare `depends_on="crispyx_preprocess"` so the run order is
  enforced: **QC → preprocess → DE**.

- **Wilcoxon OOM in Singularity (HPC) — incomplete float32 dtype alignment**: The O2
  fix ("Removed `.astype(np.float64)` casts") was only partial. `sp.csr_matrix(block,
  dtype=np.float64)` remained in both the standard path and streaming path, forcing
  float64 upstream of the `toarray()` call. On HPC via Singularity the workspace
  `__pycache__` (bind-mounted) only contained the float32 Numba kernel variant
  (`.py311.1.nbc`). The float64 code path caused Numba to trigger LLVM recompilation
  with 32 threads → 40–80 GB peak → SIGKILL. Fixed by removing `dtype=np.float64` from
  both `sp.csr_matrix(block, ...)` calls; blocks now stay at native h5ad dtype (float32).

- **Wilcoxon OOM with many perturbation groups (many-group mask pre-allocation)**:
  `wilcoxon_test` pre-validated all candidates by building a `{label: labels == label}`
  dict — one boolean array of `n_cells` per group. For Huang-HCT116 (18,293 groups ×
  3.4 M cells) this silently allocated **62 GB** before any computation began, causing
  container OOM within the 128 GB Docker limit. Fixed by replacing the dict with a
  single `np.unique(labels)` set-membership check (`O(n_cells log n_cells)` time,
  `O(n_unique)` memory ≈ 18 KB instead of 62 GB).

- **Wilcoxon OOM due to under-estimated peak memory**: `_should_use_streaming()` used
  `peak_multiplier=2.0` which only accounted for memmaps + numpy copy (~23 GB for
  Feng-gwsnf). The actual peak was ~47 GB due to backed h5ad file pages in RSS plus
  AnnData/h5py write overhead. Increased to `peak_multiplier=4.0` so the dispatch now
  correctly triggers streaming for Feng-gwsnf (4955 groups × 36518 genes) at 128 GB.

- **Benchmark runner: removed RLIMIT_CPU from all spawned workers (Feng-gwsf fix)**:
  `_worker()` previously set `RLIMIT_CPU = time_limit` for all methods. `RLIMIT_CPU`
  counts CPU time summed across **all pthreads** in the process; with 32 Numba threads
  the effective wall limit became `21600 / 32 ≈ 675 s` instead of the intended 6 hours.
  After ~4 gene chunks (≈ 23 min wall time) the accumulated CPU time exceeded the limit,
  the kernel delivered `SIGXCPU → SIGKILL` (exit code -9), and the job was misdiagnosed
  as OOM because peak RSS was only 19 GB. `RLIMIT_CPU` is now removed for **all** methods;
  wall-clock enforcement already works correctly via `process.join(timeout=time_limit+5)`
  in the parent process. `time_limit` in `Feng-gwsf.yaml` raised from 21600 → 32400 s
  (6 h → 9 h) to match the dataset's legitimate ~7.9 h runtime
  (83 chunks × ~344 s/chunk). Regression tests added in `tests/test_benchmarking.py`.

### Added
- **`run_preprocess()` benchmark function** (`run_benchmarks.py`): wraps
  `normalize_total_log1p()` as a timed, resumable benchmark step so its wall-clock and
  memory cost are captured separately from DE.

- **`crispyx_preprocess` BenchmarkMethod**: new step between `crispyx_qc_filtered` and the
  DE benchmarks. Normalizes the QC-filtered h5ad with streaming `normalize_total_log1p()`.

- **Singularity HPC Numba warmup guard** (`slurm_benchmark.sh`): A cache-guarded
  single-threaded Numba warmup step runs before the benchmark if the kernel index file
  (`_wilcoxon_batch_perts_presorted_numba-1784.py311.nbi`) is not yet present in the
  workspace `__pycache__`. Uses `NUMBA_NUM_THREADS=1` to keep LLVM peak memory low.
  Subsequent runs skip the warmup automatically.

- **`memory_limit_gb` parameter for `wilcoxon_test()`, `t_test()`, and `shrink_lfc()`**: Unified memory
  budget parameter matching `nb_glm_test()`. For `wilcoxon_test()`, controls whether the
  streaming path is used for large datasets. For `t_test()`, controls automatic cell
  chunk size calculation. For `shrink_lfc()`, limits parallel workers in the "full" method.
  When `None` (default), detects available system memory via
  `psutil`. For HPC environments, set explicitly (e.g., `memory_limit_gb=128`).
  ```python
  # Wilcoxon with memory limit
  result = cx.wilcoxon_test(
      "large_dataset.h5ad",
      perturbation_column="perturbation",
      memory_limit_gb=128,
  )

  # t-test with memory limit
  result = cx.t_test(
      "large_dataset.h5ad",
      perturbation_column="perturbation",
      memory_limit_gb=128,
  )
  ```

- **Adaptive memory dispatch (`_should_use_streaming`)**: Extracted reusable memory dispatch
  function to `crispyx._memory`. Estimates peak memory for output arrays and automatically
  switches to group-batch streaming when the memmap approach would exceed 30% of the memory
  budget. Small datasets that previously worked remain on the fast standard path.

- **Streaming Wilcoxon for large group counts**: New `_wilcoxon_test_streaming()` processes
  perturbation groups in batches, writing results incrementally to h5ad via h5py. This
  keeps peak memory bounded by `batch_size × n_genes` rather than `n_groups × n_genes`,
  enabling datasets with thousands of perturbation groups under 128 GB.

- **Dense array optimisation in Wilcoxon standard path**: Control and perturbation data
  are now extracted directly from sparse matrices per group instead of materialising a
  single dense array for all cells. Reduces per-chunk memory from O(n_cells × chunk_size)
  to O(max_group_size × chunk_size).

- **`calculate_nb_glm_chunk_size()`**: New function that automatically calculates
  memory-aware chunk sizes for NB-GLM operations based on dataset dimensions,
  perturbation group count, and available memory.
  - Small/medium datasets keep default chunk_size=256 (no speed impact)
  - Large memory-constrained datasets automatically reduce chunk size to avoid OOM
  - Considers `memory_limit_gb` parameter when calculating chunk sizes
  - Formula: `chunk_size = min(256, usable_memory / (n_obs × 48 × safety_factor))`
  ```python
  from crispyx.data import calculate_nb_glm_chunk_size
  
  # Returns 256 for small datasets (sufficient memory)
  chunk = calculate_nb_glm_chunk_size(100000, 20000, n_groups=100, available_memory_gb=128)
  
  # Returns ~143 for large datasets (memory-constrained)
  chunk = calculate_nb_glm_chunk_size(1200000, 36000, n_groups=500, available_memory_gb=128)
  ```

- **Auto-adaptive `chunk_size` in `nb_glm_test()`**: The `chunk_size` parameter now defaults 
  to `None`, enabling automatic calculation based on dataset dimensions and `memory_limit_gb`.
  - Previously defaulted to `chunk_size=256`, which caused OOM on large datasets
  - Now calls `calculate_nb_glm_chunk_size()` internally when `chunk_size=None`
  - Datasets that ran successfully before continue to use chunk_size=256 (no speed regression)
  - Large datasets like Frangieh (218K cells) and Feng-ts (1.2M cells) now automatically
    use smaller chunks to fit within memory limits
  ```python
  # Automatic chunk size (recommended)
  result = cx.nb_glm_test(
      "large_dataset.h5ad",
      perturbation_column="perturbation",
      memory_limit_gb=128,  # Chunk size calculated from this
  )
  
  # Manual override still supported
  result = cx.nb_glm_test(
      "large_dataset.h5ad",
      perturbation_column="perturbation",
      chunk_size=64,  # Explicit small chunk
  )
  ```

- **Streaming PCA (`cx.pp.pca()`)**: Memory-efficient PCA for on-disk datasets using hybrid
  method selection:
  - **Sparse covariance method** (`sparse_cov`): ~5× faster than IncrementalPCA for datasets
    with ≤15K genes. Exploits sparsity in X^T @ X computation.
  - **IncrementalPCA method** (`incremental`): Lower memory for datasets with >15K genes.
    Uses sklearn's IncrementalPCA with streaming partial_fit().
  - **Automatic selection** (`method='auto'`): Chooses optimal method based on gene count
    and available memory. Threshold at ~15K genes.
  - Memory-aware chunk size calculation via `calculate_pca_chunk_size()`.
  - Supports highly variable gene filtering (`use_highly_variable=True`).
  - Results stored in `obsm['X_pca']`, `varm['PCs']`, `uns['pca']`.
  ```python
  cx.pp.pca(adata, n_comps=50)  # Auto-selects optimal method
  cx.pp.pca(adata, n_comps=50, method='sparse_cov')  # Force fast method
  ```

- **KNN graph construction (`cx.pp.neighbors()`)**: Compute k-nearest neighbors graph
  from PCA embeddings for downstream clustering and visualization.
  - Supports `pynndescent` (fast approximate, default) and `sklearn` (exact) methods.
  - UMAP-style fuzzy connectivities with exponential decay.
  - Configurable `n_neighbors`, `n_pcs`, `metric` parameters.
  - Results stored in `obsp['distances']`, `obsp['connectivities']`, `uns['neighbors']`.
  ```python
  cx.pp.pca(adata, n_comps=50)
  cx.pp.neighbors(adata, n_neighbors=15)
  ```

- **UMAP embedding (`cx.tl.umap()`)**: Compute UMAP embeddings from pre-computed neighbor
  graph. Memory-efficient: only loads neighbor graph (~1.5GB for 2M cells), not expression
  matrix.
  - Wrapper around `scanpy.tl.umap()` with close-write-reopen pattern for backed data.
  - Configurable `min_dist`, `spread`, `n_components` parameters.
  - Results stored in `obsm['X_umap']`, `uns['umap']`.
  ```python
  cx.pp.pca(adata, n_comps=50)
  cx.pp.neighbors(adata, n_neighbors=15)
  cx.tl.umap(adata, min_dist=0.5)  # Writes X_umap to h5ad
  cx.pl.umap(adata, color='perturbation')  # Scanpy-style plot
  ```

- **New dimension reduction module** (`src/crispyx/dimred.py`): Contains all streaming
  PCA, neighbor computation, and UMAP logic with comprehensive test suite (45 tests).

- **PCA plotting helpers** (`cx.pl.pca`, `cx.pl.pca_variance_ratio`, `cx.pl.pca_loadings`):
  Scanpy-style plotting wrappers for backed AnnData that load only embeddings into memory.
  ```python
  cx.pp.pca(adata, n_comps=50)
  cx.pl.pca(adata, color='perturbation')  # Scatter plot
  cx.pl.pca_variance_ratio(adata)         # Variance explained
  cx.pl.pca_loadings(adata, components=[1, 2, 3])  # Gene loadings
  ```

- **UMAP plotting (`cx.pl.umap()`)**: Scanpy-style UMAP visualization for backed AnnData.
  Loads only X_umap embedding and obs metadata, not expression matrix.
  ```python
  cx.pl.umap(adata, color='perturbation', size=10)
  ```

- **Close-write-reopen pattern for backed data**: `cx.pp.pca()`, `cx.pp.neighbors()`,
  and `cx.tl.umap()` now write results directly to h5ad files when using crispyx.AnnData
  wrapper. This keeps `.X` on disk while persisting embeddings, loadings, and neighbor graphs.
  - No need for `copy=True` in typical workflows
  - New h5ad write helpers: `write_obsm_to_h5ad`, `write_varm_to_h5ad`,
    `write_uns_dict_to_h5ad`, `write_obsp_to_h5ad`
  - PCA plotting functions also accept in-memory AnnData for flexibility
  ```python
  # Clean API - results written to h5ad
  adata = cx.read_h5ad_ondisk("data.h5ad")
  cx.pp.pca(adata, n_comps=50)      # Writes to file
  cx.pp.neighbors(adata)            # Writes to file
  cx.pl.pca(adata, color='group')   # Reads from file
  ```

- **`cx.pl.pca_variance_ratio()` TypeError with default `n_pcs`**: Fixed bug where
  passing `n_pcs=None` explicitly to scanpy's `pca_variance_ratio` overrode the default
  value of 30, causing `TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'`.
  Now only passes `n_pcs` when explicitly specified by the user.

- **Benchmark effect size extraction bug**: Fixed `generate_results.py` to correctly 
  extract effect sizes from crispyx DE outputs. The code checked for `logfoldchange`
  (singular) layer but crispyx stores results in `logfoldchanges` (plural) layer.
  This caused accuracy heatmaps to show artificially low overlap between crispyx 
  methods (t-test, Wilcoxon) and NB-GLM. Effect sizes are now correctly read from
  the `logfoldchanges` layer when available.

- **Benchmark "Ran out of input" subprocess crash handling**: Fixed `run_benchmarks.py`
  error handling when benchmark subprocess crashes during result transmission. Previously,
  if a worker process crashed (e.g., OOM killed) while writing results to the queue,
  the parent would fail with cryptic `EOFError: Ran out of input` from pickle. Now:
  - Checks process exit code before reading from queue
  - Catches `EOFError` and `queue.Empty` with descriptive error messages
  - Reports "Worker process crashed (exitcode=X)" or "Result transmission failed"
  - Increased queue flush delay from 0.5s → 1.0s to prevent race conditions
  - Affected methods: crispyx_de_wilcoxon, crispyx_de_nb_glm on large datasets


### Added
- **72 new unit tests for Wilcoxon dispatch** (`tests/test_wilcoxon_dispatch.py`):
  Covers kernel correctness (`_presort_control_nonzeros`, `_wilcoxon_presorted_ctrl_numba`),
  standard vs streaming numerical parity (tol ≤ 1e-10 on real Adamson_subset data),
  dispatch-mode decisions for all 10 benchmark datasets at 128 GB, chunk-size bounds,
  and memory-estimate accuracy. Full suite: 109 tests (72 new + 37 existing).

- **Rerun Scanpy script (`rerun_scanpy.py`)**: New tool to run Scanpy QC, t-test, 
  and Wilcoxon methods without time/memory limits for datasets where they fail 
  in the main benchmark.
  - Run AFTER benchmarks to extract Scanpy outputs for accuracy comparison
  - Automatically regenerates benchmark reports with updated accuracy tables
  - Does NOT modify .benchmark_cache (preserves benchmark integrity)
  - Usage: `./run_rerun_scanpy.sh config/Replogle-GW-k562.yaml`
  - SLURM submission: `./submit_rerun_scanpy.sh Replogle-GW-k562.yaml`

- **SLURM support for rerun_scanpy**: Added `slurm_rerun_scanpy.sh` and 
  `submit_rerun_scanpy.sh` for running on HPC clusters with Singularity.

### Changed
- **`generate_results.py` now detects reference outputs**: Accuracy comparisons 
  work even when Scanpy failed in benchmark but succeeded via rerun_scanpy.
- **Fixed Scanpy DE file extensions in cache.py**: Corrected expected output 
  paths from `.h5ad` to `.csv` for scanpy_de_t_test and scanpy_de_wilcoxon.
- **`wilcoxon_test()` now passes `n_groups` to chunk calculator**: The function now 
  determines the number of perturbation groups before calculating chunk size, enabling 
  group-aware memory optimization.
- **`rerun_scanpy` now tracks peak memory**: `extract_scanpy_qc()` and 
  `extract_scanpy_de()` record `peak_memory_mb` alongside `elapsed_seconds` in the
  `.reference_extracted` marker file, using `get_peak_memory_mb()` from profiling.
- **Benchmark summary table shows rerun data for failed Scanpy methods**: When a Scanpy
  method originally failed (timeout/error/memory_limit) but was rerun without limits,
  the performance table now shows the original error status annotated with
  "(rerun: no limits)" and includes **Rerun (s)** / **Rerun Mem (MB)** columns with
  the unlimited-run time and memory usage.
- **Cell-count caps in `calculate_optimal_gene_chunk_size()` are now memory-aware**:
  Previously unconditional cell-count caps are now gated on `available_memory_gb < 32.0`.
  On 128 GB benchmark servers the caps are lifted: Feng-gwsf (322K cells) goes from
  128 → 384 genes/chunk; Feng-ts (1.16M cells) goes from 32 → 512 genes/chunk,
  cutting estimated Wilcoxon runtime from ~46 hr to ~2.9 hr. Machines with ≤ 32 GB
  retain the original protective caps (`_CELL_COUNT_CAP_MEMORY_THRESHOLD_GB = 32.0`).
- **Docker image pre-compiles Numba JIT kernels at build time**: New
  `benchmarking/tools/numba_warmup.py` script runs during `docker build`, exercising
  both float32 and float64 type specialisations for all Wilcoxon kernels and baking
  the Numba cache into the image layer. Eliminates the ~25-minute cold-start JIT
  overhead on the first `crispyx_de_wilcoxon` benchmark run.

### Fixed (2026-02-25 — Feng-gwsnf benchmark runner fixes)
- **Singularity PYTHONPATH priority for bind-mounted source** (`slurm_benchmark.sh`):
  Added `--env "PYTHONPATH=/workspace/src:/workspace"` to both `singularity exec`
  commands (Numba warmup and main benchmark). Without this, the SIF-baked
  `site-packages` egg-link took precedence over the bind-mounted host source in
  `mp.spawn` child processes, meaning OOM fixes from 2026-02-24 were never active
  inside the container. This was the root cause of the Feng-gwsnf Wilcoxon SIGKILL
  at 35 GB peak RSS.

- **PyDESeq2 benchmarks use QC-filtered data** (`run_benchmarks.py`):
  `pertpy_de_pydeseq2` and `pertpy_de_lfcshrink` now use `qc_filtered_path` instead
  of the raw standardised `dataset_path`. The raw data contained singleton perturbation
  groups and 36,518 genes (vs 32,373 after QC), inflating the dense count matrix to
  38.3 GiB and causing `MemoryError`. Using the QC-filtered file eliminates singletons
  and reduces the matrix by ~11%.

- **PyDESeq2 int32 count arrays** (`run_benchmarks.py`): Changed `.astype(int)` →
  `.astype(np.int32)` in all three PyDESeq2 functions (`run_pydeseq2_integrated`,
  `run_pydeseq2_base`, `run_pydeseq2_lfcshrink`). `.astype(int)` produced int64
  arrays (8 bytes/element), doubling memory for the dense count matrix. int32 is
  sufficient for scRNA-seq UMI counts (max ~65K) and halves the allocation.

## [0.7.3] - 2026-01-31

### Added
- **Scanpy-style plotting namespace (`cx.pl`)** with on-disk safe helpers for DE and QC plots.
- **On-demand materialization of `uns['rank_genes_groups']`** for plotting without loading counts.
- **DE plotting utilities**: rank-genes plot wrapper, volcano, MA (raw or normalized log1p means), and top-genes bar plots.
- **QC plotting utilities**: perturbation composition and QC summary distributions from `QualityControlResult`.

### Fixed
- **Layer naming consistency for DE results**: Fixed inconsistent HDF5 layer naming where some layers were named `logfoldchange` instead of `logfoldchanges`. All differential expression output now uses standardized layer names for consistency with Scanpy conventions and internal API expectations.

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
