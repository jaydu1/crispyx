Benchmarking
============

The benchmarking toolkit automates comparisons between the streaming
implementations in this project. Generate the synthetic demo dataset with
``python benchmarking/generate_demo_dataset.py`` (or supply your own data) and
run the benchmark suite against any compatible ``.h5ad`` file that exposes
``perturbation`` and ``gene_symbols`` columns.

Running benchmarks
------------------

.. code-block:: bash

   python benchmarking/run_benchmarks.py \
       --data-path data/demo_benchmark.h5ad \
       --time-limit 300 \
       --memory-limit 4.0

Outputs
-------

* Resource usage measurements stored in ``benchmarking/results/benchmark_results.csv``.
* A GitHub-friendly table at ``benchmarking/results/benchmark_results.md``.
* Intermediate AnnData files written directly to ``benchmarking/results`` (or
  any directory provided via ``--output-dir``).
* Comprehensive differential expression parity metrics: max absolute
  differences, Pearson/Spearman correlations, top-``k`` overlaps (``k=50`` by
  default), and AUROC values when ground-truth labels are present. These are
  captured in both the CSV and Markdown summaries so ranking agreement is easy
  to audit alongside effect size deviations.

The script accepts additional options to benchmark a subset of methods or to
redirect outputs to a different directory. Refer to ``benchmarking/README.md``
for further details.

Available NB-GLM benchmark methods
----------------------------------

The benchmark suite includes three NB-GLM variants to compare fitting strategies:

* ``crispyx_de_nb_glm``: Independent fitting (each perturbation fit separately)
* ``crispyx_de_nb_glm_joint``: Joint fitting with shared intercept from control cells
* ``crispyx_de_nb_glm_joint_shared_disp``: Joint fitting with shared intercept AND shared dispersion

These allow systematic comparison of accuracy and runtime trade-offs between
fitting strategies on your dataset.

Latest Adamson_subset benchmark
-------------------------------

For the most recent Adamson_subset run (2025-11-26), see :doc:`Adamson_subset_results`.
