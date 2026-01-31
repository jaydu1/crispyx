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

Plotting
--------

crispyx provides Scanpy-style plotting helpers under ``cx.pl`` that work with
on-disk results. The plotting functions materialise only the metadata needed
for plotting, keeping the expression matrix on disk.

.. code-block:: python

   # Rank genes groups plot (Scanpy-style)
   cx.pl.rank_genes_groups(adata_de, n_genes=20, sharey=False)

   # Convert DE results into a tidy DataFrame for custom plots
   df = cx.pl.rank_genes_groups_df(adata_de, group="perturbation_A", n_genes=200)

   # Volcano and top-genes plots
   cx.pl.volcano(de_df=df, group="perturbation_A")
   cx.pl.top_genes_bar(de_df=df, group="perturbation_A", topn=15)

   # MA plot using raw counts or normalized log1p means
   cx.pl.ma(
       data=adata_ro,  # raw counts
       de_result=adata_de,
       group="perturbation_A",
       reference="control",
       perturbation_column="perturbation",
       mean_mode="raw",  # or "log1p"
   )

   # QC plotting (composition + summary distributions)
   qc = cx.pp.qc_summary(
       adata_ro,
       perturbation_column="perturbation",
       min_genes=100,
       min_cells_per_perturbation=15,
       min_cells_per_gene=10,
   )
   cx.pl.qc_perturbation_counts(
       data=adata_ro,
       perturbation_column="perturbation",
       cell_mask=qc.cell_mask,
   )
   cx.pl.qc_summary(qc, min_genes=100, min_cells_per_gene=10)

NB-GLM options
~~~~~~~~~~~~~~

The negative binomial GLM (``method="nb_glm"``) supports several options:

* **LFC shrinkage**: For improved accuracy, apply adaptive Cauchy prior 
  shrinkage to log-fold changes using ``shrink_lfc()`` after running 
  ``nb_glm_test()``. This preserves large effects while shrinking 
  small/uncertain effects toward zero.

* **Dispersion sharing** (``share_dispersion=True``): Estimate dispersion once 
  using all cells (similar to PyDESeq2's approach). This provides more stable 
  estimates when sample sizes are small or when you expect homogeneous 
  dispersion across perturbations.

* **Scanpy format** (``scanpy_format=True``): Write Scanpy-compatible 
  ``uns["rank_genes_groups"]`` structure for interoperability with 
  ``sc.get.rank_genes_groups_df()`` and similar utilities. This option is 
  available for ``t_test()``, ``wilcoxon_test()``, and ``nb_glm_test()``.
  Default is ``False`` for performance.

.. code-block:: python

   # Basic NB-GLM (faster, per-perturbation dispersion)
   adata_de = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       method="nb_glm",
   )

   # NB-GLM with shared dispersion
   adata_de = cx.tl.rank_genes_groups(
       adata_ro,
       perturbation_column="perturbation",
       method="nb_glm",
       share_dispersion=True,
   )

LFC shrinkage
~~~~~~~~~~~~~

For more accurate log-fold change estimates, apply apeGLM shrinkage after
running the NB-GLM test. This two-step workflow matches DESeq2/PyDESeq2
best practices:

.. code-block:: python

   # Step 1: Run NB-GLM test
   result = cx.nb_glm_test(
       adata_ro,
       perturbation_column="perturbation",
   )
   
   # Step 2: Apply LFC shrinkage
   shrunk = cx.shrink_lfc(
       result.result_path,
       prior_scale_mode="global",  # or "per_comparison"
   )

The ``prior_scale_mode`` parameter controls how the shrinkage prior is estimated:

* ``"global"`` (default): Estimate a single prior scale from all comparisons.
  Recommended for CRISPR screens with many perturbations.
* ``"per_comparison"``: Estimate prior scale separately for each perturbation.
  May be more accurate when effect sizes vary substantially across perturbations.

You can also use ``cx.tl.shrink_lfc()`` for API consistency with other tools:

.. code-block:: python

   shrunk = cx.tl.shrink_lfc(
       result.result_path,
       prior_scale_mode="global",
   )

Resume and checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~

Long-running differential expression analyses can be resumed after interruption
using the ``resume`` and ``checkpoint_interval`` parameters:

.. code-block:: python

   # Enable checkpointing during long runs
   result = cx.nb_glm_test(
       adata_ro,
       perturbation_column="perturbation",
       resume=True,
       checkpoint_interval=10,  # Save progress every 10 perturbations
   )

If interrupted, simply re-run the same command - completed perturbations will
be skipped automatically. The checkpoint file ``<output>_progress.json`` is
written atomically to prevent corruption.

Benchmarking
------------

The ``benchmarking`` directory ships with a reusable script that measures time
and memory usage across the main methods. Execute
``python benchmarking/run_benchmarks.py`` to generate CSV and Markdown reports
for any compatible dataset.
