Usage Guide
===========

The core workflow mirrors the steps demonstrated in the tutorial notebook.
Each operation streams data from disk so that large ``.h5ad`` files can be
processed on commodity hardware.

Setting up
----------

Install the project in editable mode with optional dependencies:

.. code-block:: bash

   pip install -e .[test]

Loading data
------------

Use :func:`streamlined_crispr.data.read_backed` to open a dataset without
materialising the expression matrix. A synthetic demo dataset can be generated
via ``python benchmarking/generate_demo_dataset.py`` and stored at
``data/demo_benchmark.h5ad``.

Quality control
---------------

Run :func:`streamlined_crispr.qc.quality_control_summary` to filter cells,
perturbations, and genes. The function returns masks describing the retained
entries and writes a filtered AnnData file to disk.

Effect size estimation
----------------------

Two complementary estimators are provided:

* :func:`streamlined_crispr.pseudobulk.compute_average_log_expression`
* :func:`streamlined_crispr.pseudobulk.compute_pseudobulk_expression`

Each operates on the filtered dataset and produces an AnnData artifact
containing the effect sizes per perturbation.

Differential expression
-----------------------

Apply :func:`streamlined_crispr.de.wald_test` or
:func:`streamlined_crispr.de.wilcoxon_test` to compare perturbations against the
control population. The results can be converted into tidy DataFrames for
summary tables or visualisation.

Benchmarking
------------

The ``benchmarking`` directory ships with a reusable script that measures time
and memory usage across the main methods. Execute
``python benchmarking/run_benchmarks.py`` to generate CSV and Markdown reports
for any compatible dataset.
