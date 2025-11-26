Adamson_subset benchmark (2025-11-26)
=====================================

Summary from rerunning ``benchmarking/run_benchmark.sh`` with ``benchmarking/config/Adamson_subset.yaml``.
All methods were executed on the preprocessed Adamson subset dataset, with errors noted where applicable.

Performance
-----------

.. list-table::
   :header-rows: 1
   :widths: 28 12 18 18

   * - method
     - status
     - elapsed_seconds
     - peak_memory_mb
   * - crispyx_de_nb_glm
     - success
     - 88.21
     - 59.578
   * - crispyx_de_t_test
     - success
     - 5.251
     - 156.551
   * - crispyx_de_wilcoxon
     - success
     - 12.728
     - 57.363
   * - crispyx_pb_avg_log
     - success
     - 4.403
     - 482.664
   * - crispyx_pb_pseudobulk
     - success
     - 3.145
     - 330.422
   * - crispyx_qc_filtered
     - success
     - 4.567
     - 273.828
   * - edger_de_glm
     - error
     - 2.316
     - n/a
   * - pertpy_de_pydeseq2
     - error
     - 4.543
     - n/a
   * - scanpy_de_t_test
     - success
     - 6.865
     - 235.176
   * - scanpy_de_wilcoxon
     - success
     - 17.324
     - 456.789
   * - scanpy_qc_filtered
     - success
     - 2.281
     - 178.633

Performance comparison
----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 16 16 16 17

   * - comparison
     - crispyx_time_s
     - other_time_s
     - time_pct
     - mem_pct
   * - crispyx_qc_filtered vs scanpy_qc_filtered
     - 4.567
     - 2.281
     - 200.192
     - 153.291
   * - crispyx_de_t_test vs scanpy_de_t_test
     - 5.251
     - 6.865
     - 76.499
     - 66.568
   * - crispyx_de_wilcoxon vs scanpy_de_wilcoxon
     - 12.728
     - 17.324
     - 73.475
     - 12.558

Accuracy
--------

.. list-table::
   :header-rows: 1
   :widths: 38 14 14 16 18

   * - comparison
     - effect_max_abs_diff
     - effect_pearson_corr
     - statistic_pearson_corr
     - statistic_top_k_overlap
   * - crispyx_qc_filtered vs scanpy_qc_filtered
     - 0.0
     - n/a
     - n/a
     - n/a
   * - crispyx_de_t_test vs scanpy_de_t_test
     - 24.025
     - 0.362
     - 1.0
     - 1.0
   * - crispyx_de_wilcoxon vs scanpy_de_wilcoxon
     - 24.029
     - 0.346
     - 0.975
     - 0.94

Notes
-----

* ``edger_de_glm`` and ``pertpy_de_pydeseq2`` errored on this dataset; refer to the
  full log files in ``benchmarking/logs/20251126_064051_*.log`` for stack traces.
* Cell and gene retention counts are present in the QC rows of the full ``results.md``
  artifact under ``benchmarking/results/Adamson_subset/``.
