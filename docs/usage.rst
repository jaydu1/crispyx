Usage Guide
===========

The core workflow mirrors the steps demonstrated in the tutorial notebook and
now adopts a Scanpy-style API so existing notebooks can migrate with minimal
changes. Each operation streams data from disk so large ``.h5ad`` files can be
processed on commodity hardware.

Quick start
-----------

.. code-block:: python

   import streamlined_crispr as scr

   adata_ro = scr.read_h5ad_ondisk("data/demo_benchmark.h5ad")
   qc = scr.pp.qc_summary(
       adata_ro,
       perturbation_column="perturbation",
       min_genes=100,
       min_cells_per_perturbation=15,
       min_cells_per_gene=10,
   )
   avg = scr.pb.average_log_expression(
       qc.filtered,
       perturbation_column="perturbation",
   )
   de = scr.tl.rank_genes_groups(
       qc.filtered,
       perturbation_column="perturbation",
       method="wilcoxon",
   )
   qc.filtered.close()
   adata_ro.close()

Setting up
----------

Install the project in editable mode with optional dependencies:

.. code-block:: bash

   pip install -e .[test]

Loading data
------------

Use :func:`streamlined_crispr.read_h5ad_ondisk` to open a dataset without
materialising the expression matrix and print the first few rows of metadata.
A synthetic demo dataset can be generated via
``python benchmarking/generate_demo_dataset.py`` and stored at
``data/demo_benchmark.h5ad``. The returned :class:`scr.AnnData` object keeps a
backed AnnData handle alive lazily and automatically closes it when the wrapper
is garbage collected. Explicitly call ``adata.close()`` to release the file as
soon as you finish the preview.

Quality control
---------------

Call :func:`streamlined_crispr.pp.qc_summary` to filter cells, perturbations,
and genes. The function returns masks describing the retained entries and
writes a filtered AnnData file to disk, exposing the backed result via the
``filtered`` attribute as a :class:`scr.AnnData` wrapper. When ``control_label``
is omitted, perturbations containing strings such as ``ctrl`` or ``nontarget``
are chosen automatically and logged for reproducibility. Likewise, omitting
``gene_name_column`` falls back to ``adata.var_names`` with a helpful message.
Individual helpers such as :func:`streamlined_crispr.pp.filter_cells` are also
available for customised pipelines.

Effect size estimation
----------------------

Two complementary estimators are exposed through :mod:`streamlined_crispr.pb`:

* :func:`streamlined_crispr.pb.average_log_expression`
* :func:`streamlined_crispr.pb.pseudobulk`

Each operates on the filtered dataset and produces an AnnData artifact
containing the effect sizes per perturbation. The control label inference and
gene name fallbacks described above apply here as well, so these functions can
operate with minimal boilerplate on well-annotated datasets. Passing a
``scr.AnnData`` instance from the QC step avoids reopening the file path
manually.

Differential expression
-----------------------

Invoke :func:`streamlined_crispr.tl.rank_genes_groups` to compare perturbations
against the control population while matching the familiar
:func:`scanpy.tl.rank_genes_groups` interface. Choose ``method="wilcoxon"`` (the
default) for a Mann-Whitney U test, ``method="wald"`` for the streaming Wald
test, or ``method="nb_glm"`` to fit the negative binomial GLM that supports
covariates. The helper reuses the automatic control inference so a missing
``control_label`` triggers the same adaptive search used earlier, and the
returned :class:`scr.AnnData` wrapper keeps the differential expression output
available for downstream inspection.

Benchmarking
------------

The ``benchmarking`` directory ships with a reusable script that measures time
and memory usage across the main methods. Execute
``python benchmarking/run_benchmarks.py`` to generate CSV and Markdown reports
for any compatible dataset.
