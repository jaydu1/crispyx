# Validation Plan for Streamlined CRISPR Pipeline vs. Scanpy

This plan documents how to verify that each major stage of the streamlined
pipeline matches an equivalent Scanpy workflow on the provided
`data/Adamson_subset.h5ad` dataset. The goal is to confirm parity for
preprocessing, filtering, and pseudo-bulk aggregation while highlighting any
expected differences in downstream differential expression (DE).

## 1. Preprocessing parity

1. Load the raw AnnData object in memory with Scanpy.
2. Run `scanpy.pp.normalize_total(target_sum=1e4)` followed by `scanpy.pp.log1p`.
3. Iterate over the backed AnnData file with
   `streamlined_crispr.data.iter_matrix_chunks` and apply
   `streamlined_crispr.data.normalize_total_block` (including the subsequent
   `np.log1p`).
4. Compare the Scanpy-normalised blocks with the streamed blocks and record the
   maximum absolute difference for both the normalised and log-transformed
   matrices. Differences beyond numerical precision would signal an
   inconsistency in library-size scaling or log handling.

## 2. Quality-control filtering

1. Run `streamlined_crispr.qc.quality_control_summary` with the tutorial
   parameters (`min_genes=100`, `min_cells_per_perturbation=50`,
   `min_cells_per_gene=100`).
2. In Scanpy, mimic the same filters by applying `sc.pp.filter_cells`, removing
   perturbations with insufficient representation (while keeping the control),
   and finally `sc.pp.filter_genes` on the remaining cells.
3. Compare the number of retained cells and genes. Inspect discrepancies in the
   cell masks to ensure any differences stem from intentional parameter
   choices.

## 3. Pseudo-bulk aggregation

1. Use `streamlined_crispr.pseudobulk.compute_average_log_expression` and
   `compute_pseudobulk_expression` on the QC-filtered dataset.
2. With the Scanpy-filtered AnnData object, compute matching statistics by
   grouping cells by perturbation label:
   - For log-normalised averages, aggregate the `log1p` matrix.
   - For pseudo-bulk, average the library-size normalised counts, apply the same
     baseline factor, and compute log-fold changes relative to the control.
3. Align the resulting DataFrames by genes and perturbations and inspect maximum
   absolute differences. Values close to numerical precision indicate matching
   implementations.

## 4. Differential expression cross-check

1. Run both Wald and Wilcoxon tests with the streamlined package.
2. Replicate comparable DE contrasts in Scanpy (e.g. `sc.tl.rank_genes_groups`
   with appropriate settings) and compare top hits and effect sizes. Because the
   statistical tests differ, only qualitative agreement is expected.

## 5. Performance measurement

1. Record wall-clock timings for each major step in both pipelines using
   `time.perf_counter`.
2. Summarise the timings in a table to highlight how the streaming approach
   compares with in-memory Scanpy execution on the sample dataset.

## 6. Notebook updates

1. Integrate a reusable comparison helper (see
   `streamlined_crispr.scanpy_validation.compare_with_scanpy`).
2. Extend the tutorial notebook to:
   - Execute the comparison on `data/Adamson_subset.h5ad`.
   - Display validation metrics (differences and counts).
   - Summarise runtime measurements.
   - Provide guidance for interpreting DE differences.
3. Ensure the notebook references the validation plan so future contributors can
   re-run the checks when modifying the pipeline.

Following this plan establishes a repeatable process for auditing the streaming
implementation against a well-understood Scanpy baseline.
