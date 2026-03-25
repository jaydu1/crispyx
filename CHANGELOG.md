# Changelog

All notable changes to crispyx are documented here.

## [0.7.5] - 2026-03-03

### Added
- **Data preparation utilities** (5 new feature groups, `src/crispyx/data.py`,
  `src/crispyx/plotting.py`):

  - **Feature 1 — Backed metadata editing without loading X**:
    `cx.load_obs(path)` and `cx.load_var(path)` read only the `obs`/`var`
    HDF5 group into a Pandas DataFrame (X is never touched).
    `cx.write_obs(path, df)` and `cx.write_var(path, df)` serialise an edited
    DataFrame back to disk in AnnData 0.2.0 encoding (supports categorical,
    string, and numeric dtypes). Shape-mismatch raises `ValueError`.

  - **Feature 2 — Gene name standardisation**:
    `cx.standardise_gene_names(path, ...)` applies a pipeline: (1) strip
    Ensembl version suffixes (`ENSG00000123.4` → `ENSG00000123`), (2) normalise
    `mt-` → `MT-` prefix, (3) optional online Ensembl→symbol lookup via `mygene`
    (optional dependency) with `tqdm` progress bar. `inplace=False` returns a
    `pd.Series` without writing to disk.

  - **Feature 3 — Perturbation label normalisation**:
    `cx.normalise_perturbation_labels(path, column, ...)` strips configurable
    prefixes and suffixes (vectorised `pd.Series.str.replace`), applies an
    optional regex, and maps diverse control aliases (`ctrl`, `scramble`,
    `non-targeting`, …) to a single canonical label (default `"NTC"`).

  - **Feature 4 — Auto-detection of metadata columns**:
    `cx.detect_perturbation_column(adata)` scores obs columns by name,
    dtype, unique-value count, and presence of control synonyms; returns the
    best candidate or `None`. `cx.detect_gene_symbol_column(adata)` scores var
    columns similarly. `cx.infer_columns(adata)` wraps both into one call
    returning `{"perturbation_column": ..., "gene_name_column": ...}`.

  - **Feature 5 — Overlap analysis**:
    `cx.tl.compute_overlap(sets_dict)` returns an `OverlapResult` dataclass
    with `count_matrix`, `jaccard_matrix`, and `set_sizes`.
    `cx.pl.overlap_heatmap(result)` renders the heatmap via `seaborn` (with
    `matplotlib` fallback), supporting `metric="jaccard"` (default) or
    `metric="count"`.

- **`seaborn>=0.12`** added as an explicit required dependency (`pyproject.toml`).

- **`cx.pp.convert_to_csc()` — streaming CSR→CSC conversion** (`src/crispyx/data.py`):
  New function for converting a backed h5ad file from CSR (or dense) to CSC storage
  without loading the full matrix into memory. Uses a two-pass streaming algorithm:
  - Pass 1 counts per-column NNZ → builds CSC `indptr` (O(total_nnz) read, O(n_vars) memory).
  - Pass 2 scatters non-zeros into column order using vectorised per-chunk sort, then writes
    `data`, `indices`, `indptr` to a new HDF5 file in a single sequential write pass.
  - Auto-detects if input is already CSC and returns the source file unchanged (no I/O).
  - Available as `cx.pp.convert_to_csc()` (Scanpy-style namespace) and
    `from crispyx.data import convert_to_csc` (direct import).
  - Peak memory bounded by `total_nnz × 8 bytes` (one float32 + one int32 array) plus
    one row-chunk working buffer.

- **`crispyx_preprocess_csc` benchmark step** (`benchmarking/tools/run_benchmarks.py`):
  New `run_preprocess_csc()` function and corresponding `crispyx_preprocess_csc`
  BenchmarkMethod inserted between `crispyx_preprocess` and the DE steps. Converts the
  normalized preprocessed file to CSC format before Wilcoxon testing, with stale-cache
  detection matching `run_preprocess`. Now listed in `OPTIONAL_METHODS` so it only
  runs when explicitly requested or auto-included by `_resolve_dependencies()`.

- **`run_wilcoxon_with_csc()` — unified CSC+Wilcoxon timing** (`benchmarking/tools/run_benchmarks.py`):
  New function wrapping `run_preprocess_csc` + `wilcoxon_test` into a single timed step.
  The combined wall-time is what `crispyx_de_wilcoxon` now records, along with
  sub-timing breakdown columns `csc_conversion_seconds`, `wilcoxon_seconds`, and
  `was_already_csc` in the benchmark CSV. This gives a single honest number for the
  full Wilcoxon pipeline cost rather than separate timings that could be cherry-picked.

### Changed
- **`crispyx_de_wilcoxon` now reads from the CSC-converted file**: The Wilcoxon benchmark
  step's `data` kwarg was updated from `preprocessed_path` (CSR) to `preprocessed_csc_path`
  (CSC), and `depends_on` changed from `"crispyx_preprocess"` to `"crispyx_preprocess_csc"`.
  This eliminates O(total_nnz × n_chunks) I/O (repeated full-file scans for each gene chunk)
  and replaces it with O(total_nnz) total I/O (each NNZ is read exactly once).

### Performance
- **~18× Wilcoxon speedup on large CSR datasets via CSC preprocessing**:
  For datasets where `wilcoxon_test()` accesses HDF5 data column-by-column (`axis=1`),
  CSR storage forces a full scan of all `data` and `indices` arrays per gene chunk. CSC
  storage makes each chunk access O(nnz_in_chunk), reducing total I/O from
  `n_chunks × file_size` to `1 × file_size`.

  | Dataset | Storage | Current wilcoxon | After CSC | Speedup | Notes |
  |---|---|---|---|---|---|
  | Feng-gwsf | 15.4 GB CSR | 3.35 h (observed) | ~11 min | ~18× | Slow disk, 0.11 GB/s |
  | Feng-gwsnf | 27.2 GB CSR | ~25 min (estimated) | ~1.4 min | ~18× | Fast disk, 1.0 GB/s |
  | Feng-ts | large CSR | unknown | large speedup | ~18× | Same Feng cell pool |
  | Huang-HCT116/HEK293T | large CSR | unknown | large speedup | ~18× | — |

  Memory peak also drops: Feng-gwsnf 38.9 GB → ~12.4 GB (−27 GB) because CSC column
  slices only page in the current chunk's data instead of the entire file.

- **Binary search Wilcoxon rank computation — O(n_ctrl_nz) → O(n_pert_nz × log n_ctrl_nz)**
  (`src/crispyx/_kernels.py`, `src/crispyx/de.py`):
  Replaced the inner merge walk in `_wilcoxon_single_pert_presorted` with two new kernels:
  - `_compute_ctrl_tie_sums`: parallel (`prange` over genes), computes `sum(t³–t)` tie
    corrections for control nonzero tie groups once per gene chunk — eliminating
    per-perturbation recomputation of ctrl tie sums.
  - `_rank_sum_pert_bsearch_numba`: sequential (called inside prange over perts), uses
    `np.searchsorted` to locate each unique pert value in the pre-sorted control array
    in O(log n_ctrl_nz), replacing O(n_ctrl_nz + n_pert_nz) merge walk. Also eliminates
    the 140K-float `ctrl_ranks` allocation that was allocated then immediately discarded.

  Applies to both the standard (single-pass) and streaming (batched-groups) dispatch
  paths — both call `_wilcoxon_batch_perts_presorted_numba`. Activates per gene when
  `zero_frac ≥ 0.5` (typical for 70–95% of scRNA-seq genes after log-normalisation).

  | Dataset | ctrl_n | ctrl_nz/gene (est.) | Speedup (est.) |
  |---|---|---|---|
  | Tian-crispri | ~323 | ~162 | ~12× |
  | Adamson | ~7,840 | ~3,920 | ~40× |
  | Replogle-GW-k562 | ~79,583 | ~39,792 | ~280× |
  | Feng-gwsnf | ~281,285 | ~140,643 | ~750× |
  | Feng-ts | ~534,457 | ~267,229 | ~1,400× |

  Adamson post-optimisation benchmark: **51.3 s total / 28.9 s Wilcoxon** (88 perts ×
  19,568 genes) — confirms no performance regression on small datasets.

### Fixed
- **Replogle-GW-k562 `crispyx_qc_filtered` SIGKILL** (`src/crispyx/qc.py`):
  `quality_control_summary()` estimated in-memory footprint as `file_size_gb × 2` for
  dense HDF5 files. For Replogle-GW-k562 the compressed file is 8.5 GB but the
  uncompressed matrix is 65.6 GB (`n_obs × n_vars × float32`). On a 1 TB RAM node this
  triggered the in-memory path → OOM. Fixed by computing the actual uncompressed size
  (`n_obs × n_vars × dtype_itemsize × 2 / 1e9`) when `storage_format == 'dense'`.

- **Large sparse dataset QC OOM on HPC nodes — Feng-gwsnf SIGKILL** (`src/crispyx/qc.py`):
  The in-memory threshold used `estimated_memory = file_size × 2` for sparse formats and
  compared against the raw `memory_limit_gb`. For Feng-gwsnf (27 GB on disk), this gave
  `estimated = 54 GB < 128 GB (memory_limit)` → in-memory path selected → actual
  decompressed + working copies peak to 120+ GB → SIGKILL. HDF5 gzip achieves ~3-5×
  compression for sparse scRNA-seq, so the 4× multiplier is a conservative but correct
  estimate. Fixed with two changes:
  1. **Multiplier**: sparse `file_size × 2` → `file_size × 4`
  2. **Hard cap**: threshold is now `min(memory_limit_gb × 0.6, 50.0)` — prevents
     1 TB nodes from auto-enabling in-memory for 200 GB files.

  Effective dataset routing after fix:
  | Dataset | File size | Estimated | Threshold | Strategy |
  |---|---|---|---|---|
  | Adamson_subset | 50 MB | 0.2 GB | 50 GB | in-memory ✅ |
  | Adamson | 2 GB | 8 GB | 50 GB | in-memory ✅ |
  | Frangieh | 10 GB | 40 GB | 50 GB | in-memory ✅ |
  | Feng-gwsf | 15 GB | 60 GB | 50 GB | streaming ✅ |
  | Feng-gwsnf | 27 GB | 108 GB | 50 GB | streaming ✅ (was in-memory → SIGKILL) |
  | Replogle dense | 8.5 GB | 65.6 GB (actual) | 50 GB | streaming ✅ |

- **`convert_to_csc()` raises `TypeError` when `chunk_size=None` is passed explicitly**
  (`src/crispyx/data.py`): Changed `chunk_size: int = 4096` to `chunk_size: int | None = None`
  and added a `chunk_size = chunk_size or 4096` guard at the top of the function body.
  Previously, callers that pass `chunk_size=None` (e.g., `run_wilcoxon_with_csc` when
  no `--chunk-size` CLI arg is given) received a `TypeError: 'NoneType' object cannot be
  interpreted as an integer` from `range(0, length, None)` inside `iter_matrix_chunks`.

- **`memory_limit_gb` not passed to QC** (`benchmarking/tools/run_benchmarks.py`):
  `create_benchmark_suite()` never forwarded `memory_limit_gb` to
  `quality_control_summary()`, so even when `--memory-limit 128` was specified the QC
  step ignored it and used `psutil.available` (1 TB on HPC nodes). Added
  `"memory_limit_gb": memory_limit_gb` to `crispyx_qc_filtered` kwargs.

- **`run_wilcoxon_with_csc()` used QC chunk size as Wilcoxon gene chunk size**
  (`benchmarking/tools/run_benchmarks.py`):
  `calculate_adaptive_qc_thresholds()` returns a cell-oriented row chunk size (e.g.,
  4,096 cells). This was being forwarded to `wilcoxon_test()` as the gene chunk size,
  making Wilcoxon process 4,096-gene columns per chunk. For Feng-gwsnf (281K control
  cells), a 4,096-gene chunk produces a 4.6 GB dense block; the correct
  `calculate_optimal_gene_chunk_size()` auto-computes ≤512 genes for 128 GB nodes.
  Fixed by renaming the benchmark kwarg to `csc_chunk_size` (passed only to
  `run_preprocess_csc`) and passing `chunk_size=None` to `wilcoxon_test` so it uses
  its own auto-sizing logic.

### Tests
- **`test_qc_strategy_selection_thresholds`** (`tests/test_qc_parity.py`):
  New unit test that directly validates the `quality_control_summary` strategy-selection
  logic (`file × 4 < min(memory_limit × 0.6, 50 GB)`) across the full dataset size
  spectrum — confirming Adamson/Frangieh stay in-memory while Feng-gwsf/Feng-gwsnf
  route to streaming, and that the 50 GB hard cap is respected on high-memory nodes.

- **10 new unit tests for `convert_to_csc()`** (`tests/test_convert_to_csc.py`):
  Covers basic round-trip correctness, already-CSC fast path (no file written),
  default output path naming, obs/var metadata preservation, per-column slicing
  correctness, empty matrix edge case, `get_matrix_storage_format` reporting,
  `cx.pp` API access, chunk-size independence, and dtype selection (int32/int64).

- **Numba thread-count pinned in `conftest.py`**: Added
  `os.environ.setdefault("NUMBA_NUM_THREADS", "4")` before any imports to prevent
  `Cannot set NUMBA_NUM_THREADS to a different value once the threads have been launched`
  errors when running the full test suite in one pytest session (umap-learn previously
  tried to set 240 threads after our Numba kernels had already launched 32).

- **Suppressed noisy third-party warnings in `pyproject.toml`**:
  Added `filterwarnings` entries for `anndata.OldFormatWarning` (from old-format test
  fixture h5ad files) and `Tight layout not applied` matplotlib `UserWarning` (from
  plotting tests). Previously generated 117+ warning lines per session.

### Fixed (2026-03-03 — Wilcoxon LFC correctness)

- **`wilcoxon_test()` raw-count guard: warning → `ValueError`** (`src/crispyx/de.py`):
  The count-like data check (integer dtype or all-integer-valued floats) previously
  emitted a `logger.warning` and continued, producing silent numerical errors. Now raises
  `ValueError` with a clear message pointing to `cx.pp.normalize_total_log1p`.
  Behaviour now matches scanpy's `check_nonnegative_integers` guard, which also raises
  rather than silently proceeding. The same upgrade applies to `t_test()`.

- **`_wilcoxon_test_streaming()` had no count-like check** (`src/crispyx/de.py`):
  The streaming (large-dataset) dispatch path had no raw-count guard at all. Added
  `_check_not_count_like_streaming()` — checked on the first gene chunk of the first
  group batch — matching the check now present in both the standard and t-test paths.

- **float32 overflow in Wilcoxon LFC formula** (`src/crispyx/de.py`):
  `scipy.sparse.mean(axis=0)` preserves the input dtype (float32 for typical h5ad files).
  The LFC formula `log2((expm1(mean_pert) + 1e-9) / (expm1(mean_ctrl) + 1e-9))` then
  evaluated `expm1` in float32 precision, which overflows to `inf` for mean values above
  ~88.7 (e.g., MT-CYB mean ≈ 80 and MT-CO2 mean ≈ 91 on raw count data). The fix casts
  `control_mean` and `group_mean` to float64 via `np.asarray(..., dtype=np.float64)`
  before calling `expm1`, matching scanpy's behaviour (its `sparse_mean_var_*` kernels
  accumulate into `np.zeros()` float64 arrays). Applied to both the standard and
  streaming Wilcoxon paths. Note: this overflow only manifests when raw counts are passed
  (correctly caught as an error post this fix); on log-normalized data means are ≤15 and
  float32 does not overflow.

- **Tutorial `cx.tl.rank_genes_groups(wilcoxon)` passed raw counts**
  (`notebooks/crispyx_tutorial.ipynb`): The Wilcoxon DE cell was called with `adata_ro`
  (QC-filtered raw counts). Changed to `adata_norm` (the log-normalized AnnData produced
  by `cx.pp.normalize_total_log1p`), consistent with the method's documented requirement.

### Added (2026-03-02 — Wilcoxon chunk-size & I/O optimisation)

- **`calculate_wilcoxon_chunk_size()` — dedicated chunk-size function for Wilcoxon**
  (`src/crispyx/data.py`, `src/crispyx/__init__.py`):
  New function replacing `calculate_optimal_gene_chunk_size()` at the Wilcoxon call
  site. The key difference: **no n_groups cap**. The old function applied
  `n_groups > 2000 → max 384`, `n_groups > 5000 → max 256`, `n_groups > 10000 → max 128`
  as a memory guard designed for NB-GLM (where per-group RAM scales with chunk_size).
  For Wilcoxon, all output arrays are written to on-disk memmaps immediately, so peak
  RAM per chunk is **independent of n_groups** — it is driven purely by the transient
  dense block:

  ```
  transient ≈ chunk_size × n_obs × 12 bytes   (dense float32 + ctrl float64 + pert float32)
  ```

  The new function uses only a single 5%-of-available-memory cell-budget cap:

  ```python
  cell_cap = int(available_memory_gb * 0.05e9 / (n_obs * 12))
  chunk_size = clamp(cell_cap, min_chunk=32, max_chunk=2048)
  ```

  Exported as `cx.calculate_wilcoxon_chunk_size()`. The old
  `calculate_optimal_gene_chunk_size()` is retained unchanged for backward
  compatibility.

### Changed (2026-03-02 — Wilcoxon chunk-size & I/O optimisation)

- **`wilcoxon_test()` now calls `calculate_wilcoxon_chunk_size()`** (`src/crispyx/de.py`
  line ~3453): Replaced `calculate_optimal_gene_chunk_size(n_obs, n_vars, n_groups=…)`
  with `calculate_wilcoxon_chunk_size(n_obs, n_vars, available_memory_gb=memory_limit_gb)`.
  No other paths (NB-GLM, T-test, QC) are affected — each uses its own dedicated function.

- **Vectorized memmap writes in `wilcoxon_test()` and `_wilcoxon_test_streaming()`**
  (`src/crispyx/de.py`): Replaced the per-group row-loop write:

  ```python
  # Before: 4,955 groups × 7 arrays = 34,685 Python calls per gene chunk
  for idx in range(n_groups):
      u_matrix[idx, gene_indices] = chunk_u[idx]
      ...
  ```

  with a single 2-D slice assignment per array:

  ```python
  # After: 7 Python calls per gene chunk
  u_matrix[:, slc] = chunk_u
  pvalue_matrix[:, slc] = chunk_p
  ...
  ```

  Reduces memmap I/O calls from `n_groups × 7` to `7` per chunk and improves OS page-cache
  locality (contiguous column ranges vs scattered row writes). Applied in both the standard
  memmap path and the streaming group-batch path.

### Performance (2026-03-02)

- **Feng-gwsnf: 85 gene chunks → 24 gene chunks (3.4× fewer I/O passes)**:
  With `n_groups = 4,955` perts and 128 GB memory limit, the old n_groups cap forced
  `chunk_size = 384` (85 chunks). The new cell-budget cap gives `chunk_size = 1,345`
  (24 chunks). Per-dataset comparison at 128 GB:

  | Dataset | n\_obs | n\_perts | Old chunk | New chunk | Old chunks | New chunks |
  |---|---|---|---|---|---|---|
  | Feng-gwsnf | 396,458 | 4,955 | 384 (n\_groups cap) | **1,345** | 85 | **24** |
  | Feng-ts | 1,161,864 | 444 | 459 (cell cap) | 459 | 73 | 73 (unchanged) |
  | Replogle-GW-k562 | 1,989,578 | 9,326 | 256 (n\_groups cap) | **268** | 33 | 31 |
  | Adamson | 65,337 | 88 | 512 | **2,048** | 39 | **10** |

  Feng-ts and currently-passing datasets experience no regression (cell cap is already
  active or chunk count only decreases). Proyected Wilcoxon time for Feng-gwsnf on HPC:
  **8.2 h → ~5.5 h** (within 6 h time limit).

### Changed (2026-03-20 — Wilcoxon always-binary-search & vectorization)

- **Wilcoxon kernel: removed `zero_threshold` gate — always uses binary search**
  (`src/crispyx/_kernels.py`):
  `_wilcoxon_single_pert_presorted` previously dispatched per gene based on
  `zero_frac >= 0.5`: genes above the threshold used the fast O(n_pert_nz × log n_ctrl_nz)
  binary search path, while dense genes (zero_frac < 0.5) fell back to the
  O(n_total × log n_total) argsort path — sorting ~281K combined values per gene per
  perturbation for Feng-gwsnf. The threshold was a legacy from the old merge-sort era.
  With binary search, the cost is always dominated by the tiny perturbation side
  (~12–25 non-zeros), making it ~25,000× faster per gene-pert than argsort for dense
  genes on large-control datasets. The same change applied to `_wilcoxon_presorted_ctrl_numba`.

  Results are numerically equivalent — binary search computes identical U-statistics,
  z-scores, and p-values for all zero fractions (validated against Scanpy).

- **Vectorized result mapping (Step 9)** (`src/crispyx/de.py`):
  Replaced 4 × n_groups Python-loop iterations with 4 single 2-D fancy-index assignments:
  `chunk_u[:, valid_gene_indices] = valid_u` (was per-row loop).

- **Vectorized LFC/pts computation (Step 10)** (`src/crispyx/de.py`):
  Replaced n_groups-iteration Python loop with batch numpy operations:
  `np.array(pert_expr_counts)` → 2-D division/log2 — eliminates 4,516 per-group
  `np.divide`/`np.where`/`np.log2` calls per chunk.

### Performance (2026-03-20)

- **Feng-gwsnf Wilcoxon: ~6.4 h → estimated ~1.5–3 h (2–4× faster)**:
  The argsort fallback for dense genes was the dominant bottleneck, consuming
  500–2,000s per chunk for ~800–1,500 dense genes × 4,516 perturbations. Removing
  the threshold gate eliminates this entirely.

  | Fix | Target | Estimated savings (total) |
  |---|---|---|
  | Always binary search | Dense gene argsort elimination | **4,000–16,000s** |
  | Vectorize result mapping | Step 9 loop | 40–80s |
  | Vectorize LFC/pts | Step 10 loop | 80–120s |

  Previously successful datasets (Adamson_subset, Tian, etc.) are unaffected — their
  genes were already mostly sparse (zero_frac ≥ 0.5), so they were already on the
  binary search path.

### Tests (2026-03-20)

- **`TestDenseGeneBinarySearch`** (`tests/test_wilcoxon_dispatch.py`):
  3 new tests verifying correctness of the always-binary-search path:
  - `test_dense_genes_vs_scanpy`: all-dense genes (zero_frac ≈ 0) match Scanpy p-values
  - `test_mixed_genes_vs_scanpy`: mixed dense + sparse genes match Scanpy p-values
  - `test_all_zero_gene`: gene with all zeros → p=1
  All 3 pass; 295 total tests pass (excluding pre-existing `_check_not_count_like` issues).

### Changed (2026-03-12 — Wilcoxon chunk budget & vectorized stacking)

- **`calculate_wilcoxon_chunk_size()` budget increased 5% → 15%, max_chunk 2048 → 4096**
  (`src/crispyx/data.py`):
  The per-chunk memory budget (`_PER_CHUNK_BUDGET_FRACTION`) was too conservative at 5%
  of available RAM, producing 24 chunks for Feng-gwsnf (128 GB node). Profiling showed
  ~112 GB headroom was unused during Wilcoxon. Tripling the budget to 15% (each chunk
  uses ~1/7th of RAM) and raising `max_chunk` from 2048 to 4096 allows larger chunks
  while still leaving ample memory for OS page cache and the Numba thread pool:

  | Dataset | n\_obs | Old chunk (5%) | New chunk (15%) | Old chunks | New chunks |
  |---|---|---|---|---|---|
  | Feng-gwsnf | 396,458 | 1,345 | **4,067** | 24 | **8** |
  | Feng-ts | 1,161,864 | 459 | **1,378** | 73 | **24** |
  | Replogle-GW-k562 | 1,989,578 | 268 | **804** | 31 | **11** |
  | Adamson | 65,337 | 2,048 | **4,096** | 10 | **5** |

- **Vectorized perturbation stacking in `wilcoxon_test()`** (`src/crispyx/de.py`):
  Pre-build `all_pert_flat_idx = np.concatenate([pert_idx[label] for label in candidates])`
  and `pert_row_offsets` once before the chunk loop. Inside the loop, replaced the
  per-group iteration:

  ```python
  # Before: n_groups iterations per chunk (e.g. 4,516 for Feng-gwsnf)
  rows = []
  for label in candidates:
      rows.append(all_valid_dense[pert_idx[label], :])
  all_pert_stacked = np.vstack(rows)
  ```

  with a single fancy-index:

  ```python
  # After: 1 NumPy call per chunk
  all_pert_stacked = all_valid_dense[all_pert_flat_idx, :]
  ```

  Eliminates 4,516 Python-level array allocations and the `np.vstack` copy per chunk.
  The `pert_row_offsets` array is computed once and reused across all 8 chunks.

### Performance (2026-03-12)

- **Feng-gwsnf Wilcoxon: ~8.5 h → ~6.4 h estimated (1.33× faster, 7/8 chunks in 6 h)**:
  Benchmark on 240-core/1 TB machine with `--force --methods crispyx_de_wilcoxon`:

  | Metric | Old (24 chunks, 5%) | New (8 chunks, 15%) |
  |---|---|---|
  | Chunk count | 24 | 8 |
  | Avg chunk wall time | ~1,200s (20 min) | ~2,810s (46.8 min) |
  | Total estimated | ~30,500s (8.5 h) | ~22,900s (6.4 h) |
  | Completed in 6 h | 17/24 (71%) | 7/8 (87.5%) |

  Per-chunk wall time: chunk 0 = 2,950s (includes Numba JIT), chunk 1 = 2,678s,
  chunks 2-6 average = 2,810s. The benchmark timed out after 21,605s with 7/8 chunks
  complete — chunk 7 needed ~22 more minutes. Recommended time limit: 25,200s (7 h).

### Tests (2026-03-12)

- **`TestWilcoxonChunkSize15pct`** (`tests/test_wilcoxon_dispatch.py`):
  6 new unit tests validating the updated 15% budget and 4096 max_chunk:
  `test_feng_gwsnf_larger_chunk` (4067), `test_feng_ts_larger_chunk` (1378),
  `test_small_dataset_hits_max_chunk` (4096), `test_low_memory_still_small` (32),
  `test_replogle_gw_larger_chunk` (804), `test_fewer_chunks_feng_gwsnf` (8 chunks).

### Fixed (2026-03-23 — NB-GLM sort OOM on Feng-ts / Replogle-GW-k562)

- **`_write_sorted_sparse()` OOM when sorting large unsorted datasets**
  (`src/crispyx/data.py`):
  When `nb_glm_test()` detects that cells are not contiguous by perturbation
  (contiguity < 50%), it calls `sort_by_perturbation()` to create a sorted copy.
  For sparse inputs, `_write_sorted_sparse()` loaded the **entire** CSR matrix into
  memory via `backed.X[:]` (~50 GB) then created a reordered copy
  `X_sparse[sort_indices, :]` (~50 GB more), totaling ~100–142 GB — exceeding the
  128 GB SLURM limit and causing SIGKILL within 77–86 seconds.

  Both Feng-ts (0.0% contiguity, 1.16M cells × 36K genes, ~48 GB CSR) and
  Replogle-GW-k562 (0.0% contiguity, 1.99M cells × 8K genes, ~50 GB CSR) triggered
  this path on every benchmark run because no pre-sorted files existed on HPC.

  Fixed by rewriting `_write_sorted_sparse()` to use chunked streaming I/O via h5py:
  - Reads source rows in output order, chunk_size at a time, with sorted disk access
    (rows within each chunk are read in ascending order for sequential I/O).
  - Builds CSR components (data, indices, indptr) incrementally using resizable HDF5
    datasets — no full-matrix materialisation.
  - Writes obs/var/uns metadata via a temporary AnnData file and h5py copy.

  Peak memory drops from ~100–142 GB to ~chunk_size × n_genes × 12 bytes (~a few
  hundred MB). The dense sort path (`_write_sorted_dense`) already used chunked
  streaming and was unaffected.

  | Dataset | n_cells × n_genes | CSR on disk | Before | After |
  |---|---|---|---|---|
  | Feng-ts | 1,161,864 × 36,518 | ~48 GB | ~142 GB → OOM | ~500 MB streaming |
  | Replogle-GW-k562 | 1,989,578 × 8,248 | ~50 GB | ~142 GB → OOM | ~500 MB streaming |

- **`_write_sorted_sparse()` int32 overflow on datasets with >2.1 billion non-zeros**
  (`src/crispyx/data.py`):
  The indptr computation `csr.indptr[1:] + nnz_written` used scipy's default int32
  `indptr` array. When cumulative non-zeros exceeded INT32_MAX (2,147,483,647),
  NumPy 2.x raised `OverflowError: Python integer ... out of bounds for int32`.
  Feng-ts has ~6 billion non-zeros, triggering this at ~36% progress.
  Fixed by casting `csr.indptr` to int64 before the addition. The `ds_indptr` HDF5
  dataset was already int64.

  **Note**: The previous `_deseq2_style_size_factors()` streaming fix (2026-03-20)
  was a contributing safeguard but not the primary crash cause for these two datasets.
  Profiling revealed that the size factor function always early-terminated (only 8–9
  genes expressed in all cells) and fell back to the already-streaming
  `_median_of_ratios_size_factors()`, never reaching the dense allocation. The actual
  OOM occurred in the sort step, which runs before size factor computation.

### Fixed (2026-03-20 — NB-GLM DESeq2 size factor streaming)

- **`_deseq2_style_size_factors()` dense path safeguard for large datasets**
  (`src/crispyx/_size_factors.py`):
  Added a memory check (`counts_filtered_gb > 4 GB`) that switches to a streaming
  three-pass approach when the dense `(n_cells, n_all_expressed)` float64 array would
  exceed 4 GB. While empirical profiling showed that for Feng-ts and Replogle-GW-k562
  the function early-terminates (only 8–9 genes pass the "expressed in all cells"
  filter) and falls back to `_median_of_ratios_size_factors()`, this safeguard
  prevents OOM for future datasets that may have many universally-expressed genes.

  The streaming approach:
  - Pass 1 (existing): find genes expressed in all cells (chunked, unchanged).
  - Pass 2: accumulate log-sums per gene across chunks → geometric means.
  - Pass 3: compute per-cell median of ratios chunk-by-chunk.

  Peak memory: ~60 MB per chunk vs potentially 142 GB dense.
  Results are mathematically identical to the dense path.

### Tests (2026-03-20)

- **`test_deseq2_size_factors_streaming_parity`** (`tests/test_normalisation.py`):
  Verifies that the streaming DESeq2 size factor computation produces identical
  results to the dense path on synthetic test data (200 cells × 50 genes, all
  genes expressed in every cell).

### Added (2026-03-13 — NB-GLM OOM & timeout fixes)

- **`cx.pp.convert_to_csr()` — streaming CSC/dense→CSR conversion** (`src/crispyx/data.py`):
  New function analogous to `convert_to_csc()` for converting a backed h5ad file from CSC
  or dense storage to CSR without loading the full matrix into memory. Uses a two-pass
  streaming algorithm with format-aware axis selection:
  - **CSC input**: reads column-chunks (axis=1) and scatters into CSR buffers with
    sorted-row vectorised write positions.
  - **Dense input**: reads row-chunks (axis=0) and bulk-copies into CSR data arrays.
  - Pass 1 counts NNZ per row → builds CSR `indptr`; Pass 2 fills `data` and `indices`.
  - Auto-detects if input is already CSR and returns the source path unchanged (no I/O).
  - Available as `cx.pp.convert_to_csr()` (Scanpy-style namespace) and
    `from crispyx.data import convert_to_csr` (direct import).

- **`precompute_control_statistics_streaming()` — streaming NB-GLM control stats**
  (`src/crispyx/glm.py`):
  New function replacing `precompute_control_statistics()` for large datasets where the
  dense control matrix would exceed available memory. Reads control cells from disk in
  chunks of 4,096 and performs multi-pass IRLS:
  - Pass 0: computes expression counts and means per gene.
  - Passes 1–N: accumulates XᵀWX and XᵀWz per chunk for intercept-only model fitting.
  - Peak memory: O(chunk_size × n_genes) instead of O(n_control × n_genes).
  - Auto-triggers when `control_matrix_memory × 4 > 30%` of available memory.
  - Forces `freeze_control=True` in streaming mode.
  - Returns the same `ControlStatisticsCache` as the in-memory version.

- **CSC format warning in `nb_glm_test()`** (`src/crispyx/de.py`):
  Added `get_matrix_storage_format()` check at the start of `nb_glm_test()`. If CSC
  storage is detected, emits a `UserWarning` advising the user to convert to CSR first
  via `cx.pp.convert_to_csr()`, since all NB-GLM operations use row-wise access which
  is O(total_nnz) per slice on CSC.

- **`crispyx_standardize_csr` benchmark step** (`benchmarking/tools/run_benchmarks.py`):
  New `run_preprocess_csr()` function and corresponding `crispyx_standardize_csr`
  BenchmarkMethod. Converts the standardised dataset to CSR format before NB-GLM,
  with stale-cache detection. Creates a symlink when the input is already CSR.
  Auto-included via `_resolve_dependencies()` when `crispyx_de_nb_glm` or
  `crispyx_de_nb_glm_pydeseq2` is selected. Listed in `OPTIONAL_METHODS`.

### Fixed (2026-03-18 — NB-GLM cgroup OOM on Replogle-GW / Feng-ts)

- **Page cache exhausting SLURM cgroup memory** (`src/crispyx/data.py`,
  `src/crispyx/de.py`):
  On cgroup-limited systems (SLURM), kernel page cache from h5py file reads counts
  toward the memory limit.  Streaming through a 48 GB CSR file accumulated enough
  cached pages to push RSS + page cache past 128 GB → OOM kill.
  Added `drop_file_cache(path)` utility (`posix_fadvise(FADV_DONTNEED)`) and call it
  after every streaming phase in `nb_glm_test()`:
  1. After streaming control expression counts
  2. After loading control matrix (non-streaming path)
  3. After `precompute_control_statistics_streaming()` IRLS
  4. After `precompute_global_dispersion_from_path()` streaming dispersion
  5. After non-streaming global dispersion full-matrix read

  The syscall has zero CPU cost and is a no-op on non-Linux platforms.

- **Dense IRLS peak when `freeze_control` auto-enabled** (`src/crispyx/de.py`):
  When `freeze_control` was auto-detected (condition 2: worker count < 4), the control
  matrix was still loaded dense for `precompute_control_statistics()`, producing a
  4× control-size IRLS peak (20 GB for Replogle-GW with 75K control cells × 8K genes).
  Now, when `can_use_frozen_control` is True, retroactively switches to
  `use_streaming_control=True` and frees the already-loaded control matrix.  Streaming
  IRLS processes chunks of 4,096 cells, keeping peak at ~260 MB instead of 20 GB.

- **`control_matrix` leaked after frozen cache creation** (`src/crispyx/de.py`):
  After `precompute_control_statistics()` created frozen sufficient statistics, the
  local `control_matrix` variable (5 GB for Replogle-GW) was never freed.  Now
  explicitly deleted with `gc.collect()` when `freeze_control=True`.

  | Dataset | n_cells | n_control | Before | After |
  |---|---|---|---|---|
  | Replogle-GW-k562 | 1,989,578 | 75,328 | 142 GB → OOM | ~15 GB (est.) |
  | Feng-ts | 1,161,864 | 535,083 | 142 GB → OOM | ~15 GB (est.) |

### Fixed (2026-03-13 — NB-GLM OOM & timeout fixes)

- **NB-GLM OOM on 3 large datasets** (`src/crispyx/glm.py`):
  `precompute_control_statistics()` densified the full control matrix for IRLS fitting.
  For Feng-ts (535K control cells × 36K genes), this single allocation was 156 GiB —
  exceeding the 128 GB limit. The streaming replacement reads control cells in chunks,
  keeping peak memory at O(chunk_size × n_genes) ≈ 1.2 GB.

  | Dataset | Control matrix | Old behaviour | New behaviour |
  |---|---|---|---|
  | Feng-gwsf | 63.8 GB | OOM (SIGKILL) | Streaming (< 2 GB RSS) |
  | Feng-gwsnf | 82.1 GB | OOM (SLURM oom_kill) | Streaming (< 2 GB RSS) |
  | Feng-ts | 156.2 GB | OOM (MemoryError) | Streaming (< 2 GB RSS) |

- **NB-GLM timeout on CSC datasets** (`benchmarking/tools/run_benchmarks.py`):
  `standardize_dataset()` preserved the original h5ad storage format. For Frangieh
  (CSC on disk), every row-wise access (size factors, control loading, per-perturbation
  slices) was O(total_nnz) per slice. Additionally, `calculate_adaptive_qc_thresholds()`
  in `_run_single_benchmark()` also hung on CSC files due to row-chunk iteration.
  Fixed by converting to CSR immediately after standardisation and before adaptive QC.

- **Double-prefix naming in `create_benchmark_suite()`**
  (`benchmarking/tools/run_benchmarks.py`):
  When `_run_single_benchmark()` passes an already-CSR path to `create_benchmark_suite()`,
  the derived `standardized_csr_path` would get a `standardized_csr_standardized_csr_`
  prefix. Fixed by detecting already-CSR input via `get_matrix_storage_format()` and
  using the input path directly.

### Performance (2026-03-13)

- **Frangieh NB-GLM: 6 h timeout → 22.7 min (peak 2.2 GB)**:
  Previously timed out after 6 hours due to CSC storage making row-wise access
  O(total_nnz) per slice. After CSC→CSR conversion (1.0 s, already cached) + streaming
  control statistics, the full NB-GLM run (248 perturbations × 23,712 genes) completes
  in 1,363 seconds with 2.2 GB peak memory.

- **Feng-gwsf/gwsnf/ts NB-GLM: OOM → running at < 2 GB RSS**:
  All three Feng datasets previously crashed with OOM (64–156 GB control matrices).
  With streaming control statistics, all run with < 2 GB RSS on a 128 GB node.

### Tests (2026-03-13)

- **12 new unit tests for `convert_to_csr()`** (`tests/test_convert_to_csr.py`):
  Covers CSC→CSR basic correctness, already-CSR fast path (no file written), default
  output path naming, obs/var metadata preservation, row access correctness, empty
  matrix edge case, format verification, `cx.pp` API access, `chunk_size=1`, dense→CSR,
  round-trip CSR↔CSC, and dtype check.

- **10 new unit tests for streaming control statistics**
  (`tests/test_streaming_control.py`):
  Covers streaming vs in-memory parity (means, dispersions, fitted mu), automatic
  dispatch threshold, forced streaming mode, single-chunk edge case, sparse/dense
  input, chunk size independence, frozen control enforcement, gene subset consistency,
  and convergence with real-like data.

---

## [0.7.4] - 2026-02-25

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
