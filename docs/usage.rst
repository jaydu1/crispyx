Usage Guide
===========

The core workflow mirrors the steps demonstrated in the tutorial notebook and
now adopts a Scanpy-style API so existing notebooks can migrate with minimal
changes. Each operation streams data from disk so large ``.h5ad`` files can be
processed on commodity hardware.

Quick start
-----------

.. code-block:: python

   import crispyx as cx

   adata_ro = cx.read_h5ad_ondisk("data/demo_benchmark.h5ad")
   adata_ro = cx.pp.qc_summary(
       adata_ro,
       perturbation_column="perturbation",
       min_genes=100,
       min_cells_per_perturbation=15,
       min_cells_per_gene=10,
   )
   adata_pb = cx.pb.average_log_expression(
       adata_ro,
       perturbation_column="perturbation",
   )
   adata_ro = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       method="wilcoxon",
   )
   print(adata_ro.uns["rank_genes_groups"])  # preview without loading everything
   de_results = adata_ro.uns["rank_genes_groups"].load()
   de_full = de_results["full"]
   var_table = adata_ro.var.load()

Setting up
----------

Install the project in editable mode with optional dependencies:

.. code-block:: bash

   pip install -e .[test]

Loading data
------------

Use :func:`crispyx.read_h5ad_ondisk` to open a dataset without
materialising the expression matrix and print the first few rows of metadata.
A synthetic demo dataset can be generated via
``python benchmarking/generate_demo_dataset.py`` and stored at
``data/demo_benchmark.h5ad``. The returned :class:`cx.AnnData` object keeps a
backed AnnData handle alive lazily and automatically closes it when the wrapper
is garbage collected. Explicitly call ``adata.close()`` to release the file as
soon as you finish the preview.

Quality control
---------------

Call :func:`crispyx.pp.qc_summary` to filter cells, perturbations,
and genes. The function writes a filtered AnnData file to disk and returns a
new :class:`cx.AnnData` view pointing at the result so the next step can reuse
the same handle without reopening the path. When ``control_label`` is omitted,
perturbations containing strings such as ``ctrl`` or ``nontarget`` are chosen
automatically and logged for reproducibility. Likewise, omitting
``gene_name_column`` falls back to ``adata.var_names`` with a helpful message.
Individual helpers such as :func:`crispyx.pp.filter_cells` are also
available for customised pipelines.

Effect size estimation
----------------------

Two complementary estimators are exposed through :mod:`crispyx.pb`:

* :func:`crispyx.pb.average_log_expression`
* :func:`crispyx.pb.pseudobulk`

Each operates on the filtered dataset and produces an AnnData artifact
containing the effect sizes per perturbation. The control label inference and
gene name fallbacks described above apply here as well, so these functions can
operate with minimal boilerplate on well-annotated datasets. Passing a
``cx.AnnData`` instance from the QC step avoids reopening the file path
manually, and the returned wrappers expose ``.obs``/``.var`` tables with
``.load()`` helpers for materialising the full metadata only when requested.

Differential expression
-----------------------

Invoke :func:`crispyx.tl.rank_genes_groups` to compare perturbations
against the control population while matching the familiar
:func:`scanpy.tl.rank_genes_groups` interface. Choose ``method="wilcoxon"`` (the
default) for a Mann-Whitney U test, ``method="wald"`` for the streaming Wald
test, or ``method="nb_glm"`` to fit the negative binomial GLM that supports
covariates. The helper reuses the automatic control inference so a missing
``control_label`` triggers the same adaptive search used earlier, and the
returned :class:`cx.AnnData` wrapper stores previews of the results in
``.uns`` so printing ``adata.uns["rank_genes_groups"]`` shows the top genes
per perturbation while ``.load()`` retrieves the full tables on demand.

NB-GLM fitting methods
~~~~~~~~~~~~~~~~~~~~~~

The negative binomial GLM (``method="nb_glm"``) supports two fitting strategies
via the ``fit_method`` parameter:

* ``fit_method="independent"`` (default): Each perturbation is fit separately
  against the control population. This is faster and suitable when you have
  few perturbations or expect heterogeneous effects.

* ``fit_method="joint"``: Estimates a global intercept (baseline expression)
  from control cells, then fits perturbation effects relative to this shared
  baseline. This provides more stable estimates when you have many perturbations
  and expect similar baseline expression across conditions.

Additionally, the ``share_dispersion`` parameter controls dispersion estimation:

* ``share_dispersion=False`` (default): Dispersion is estimated separately for
  each perturbation comparison.

* ``share_dispersion=True``: Dispersion is estimated once using all cells
  (similar to PyDESeq2's approach). This is more stable when sample sizes are
  small or when you expect homogeneous dispersion across perturbations.

.. code-block:: python

   # Independent fitting (default, faster)
   adata_de = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       method="nb_glm",
   )

   # Joint fitting with shared intercept
   adata_de = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       method="nb_glm",
       fit_method="joint",
   )

   # Joint fitting with shared intercept AND shared dispersion
   adata_de = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       method="nb_glm",
       fit_method="joint",
       share_dispersion=True,
   )

Benchmarking
------------

The ``benchmarking`` directory ships with a reusable script that measures time
and memory usage across the main methods. Execute
``python benchmarking/run_benchmarks.py`` to generate CSV and Markdown reports
for any compatible dataset.
