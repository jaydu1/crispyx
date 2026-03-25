Adamson_subset benchmark (2026-03-24)
=====================================

Summary from the latest ``benchmarking/run_benchmark.sh`` run with
``benchmarking/config/Adamson_subset.yaml``.

Performance
-----------

.. list-table::
   :header-rows: 1
   :widths: 28 12 18 18

   * - method
     - status
     - elapsed_seconds
     - peak_memory_mb
   * - crispyx QC filter
     - success
     - 1.49
     - 324.59
   * - crispyx preprocess
     - success
     - 1.32
     - 299.28
   * - crispyx Wilcoxon
     - success
     - 7.93
     - 716.07

Accuracy
--------

Correlation metrics between crispyx and reference methods.
✅ > 0.95, ⚠️ 0.8–0.95, ❌ < 0.8.

.. list-table::
   :header-rows: 1
   :widths: 22 22 14 14 14 14

   * - crispyx method
     - compared to
     - Eff ρ
     - Eff ρₛ
     - log-Pval ρ
     - log-Pval ρₛ
   * - t-test
     - scanpy t-test
     - 1.000 ✅
     - 1.000 ✅
     - 1.000 ✅
     - 1.000 ✅
   * - Wilcoxon
     - scanpy Wilcoxon
     - 1.000 ✅
     - 1.000 ✅
     - 1.000 ✅
     - 1.000 ✅
   * - NB-GLM
     - pertpy NB-GLM
     - 0.999 ✅
     - 1.000 ✅
     - 0.939 ⚠️
     - 0.917 ⚠️
   * - NB-GLM
     - edgeR NB-GLM
     - 0.918 ⚠️
     - 0.924 ⚠️
     - 0.874 ⚠️
     - 0.526 ❌

Gene set overlap (effect size, top-k)
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 14 14 14

   * - crispyx method
     - compared to
     - Top-50
     - Top-100
     - Top-500
   * - t-test
     - scanpy t-test
     - 1.000 ✅
     - 1.000 ✅
     - 1.000 ✅
   * - Wilcoxon
     - scanpy Wilcoxon
     - 1.000 ✅
     - 1.000 ✅
     - 1.000 ✅
   * - NB-GLM
     - pertpy NB-GLM
     - 0.960 ✅
     - 0.960 ✅
     - 0.965 ✅
   * - NB-GLM
     - edgeR NB-GLM
     - 0.640 ⚠️
     - 0.695 ⚠️
     - 0.709 ✅

Notes
-----

* Full benchmark report (including additional comparisons and overlap plots) is in
  ``benchmarking/results/Adamson_subset/benchmark_report.md``.
