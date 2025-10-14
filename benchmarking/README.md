# Benchmarking utilities

This directory contains scripts and example outputs for comparing the
streaming CRISPR screen analysis methods provided by this project.

## Quick start

First generate the synthetic demo dataset (or point the benchmarks at one
of your own ``.h5ad`` files):

```bash
python benchmarking/generate_demo_dataset.py
```

Then run the benchmark script from the repository root to execute all
available methods against the generated dataset:

```bash
python benchmarking/run_benchmarks.py
```

The script enforces configurable CPU-time and memory limits for each
method. When a method exceeds the requested resources it is terminated
and the failure is recorded in the output table.

### Comparison groups

In addition to the streaming-only methods, the benchmark now includes
two categories of parity checks against popular single-cell analysis
tools:

1. **Quality control and preprocessing comparisons** – run the full
   streaming pipeline and a Scanpy-based in-memory workflow, recording
   agreement metrics for filtering, normalisation, and pseudobulk
   summaries.
2. **Differential expression comparisons** – contrast the streaming
   differential expression outputs with reference implementations from
   Scanpy (t-test and Wilcoxon), as well as GLM-based methods exposed via
   Pertpy (edgeR, PyDESeq2, and statsmodels). When a particular backend
   is unavailable, the benchmark reports the attempt with `NA`
   placeholders instead of failing the full run.

Key command line options:

- `--data-path`: Path to an `.h5ad` file with the same columns as the
  generated demo dataset. Any dataset with `perturbation`, `celltype`, and
  `gene_symbols` fields can be used.
- `--output-dir`: Location for generated `.h5ad` outputs, CSV summaries, and
  Markdown reports.
- `--methods`: Optional subset of method names to benchmark.
- `--time-limit`: CPU time limit per method in seconds (set to `0` to
  disable the guard).
- `--memory-limit`: Memory cap per method in GiB (set to `0` to disable
  the guard).

## Outputs

Each benchmark run writes three types of files:

1. Intermediate `.h5ad` files produced by each method (written directly to
   `<output-dir>`).
2. A CSV summary (`benchmark_results.csv`) containing timing,
   memory usage, retention statistics, and high-level result
   information for each method. Paths in the CSV are recorded relative
   to the benchmark output directory so that the report remains stable
   across machines.
3. A Markdown report (`benchmark_results.md`) grouped by category
   (streaming pipeline, differential expression, and reference
   comparisons). Each section includes only the columns that are
   relevant for that group, making it easier to scan performance and
   agreement metrics at a glance when browsing on GitHub.

Example outputs generated with the demo dataset are committed alongside this
script. They provide a ready-to-view reference when browsing the repository.
