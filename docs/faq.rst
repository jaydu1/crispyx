FAQ & Troubleshooting
=====================

Common issues
-------------

``MemoryError`` or out-of-memory kills
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``memory_limit_gb`` to match your available RAM:

.. code-block:: python

   result = cx.tl.rank_genes_groups(
       adata,
       perturbation_column="perturbation",
       method="nb_glm",
       memory_limit_gb=32,  # set to your SLURM --mem value
   )

For very large datasets, consider:

* Converting to CSC before Wilcoxon (see :doc:`usage`).
* Converting to CSR before NB-GLM.
* Using ``freeze_control=True`` for datasets with >100K control cells.

When should I use CSC vs CSR format?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **CSC** (Compressed Sparse Column): Use for Wilcoxon rank-sum tests, which
  iterate over gene columns. Convert with :func:`crispyx.convert_to_csc`.
* **CSR** (Compressed Sparse Row): Use for NB-GLM, size factors, and
  operations that iterate over cells (rows). Convert with
  :func:`crispyx.convert_to_csr`.

The benchmark pipeline handles this automatically, but manual workflows
should convert before calling DE functions.

``tomllib`` / ``tomli`` import errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If building docs on Python 3.10, install the backport:

.. code-block:: bash

   pip install tomli

Python 3.11+ includes ``tomllib`` in the standard library.

Control label not detected
~~~~~~~~~~~~~~~~~~~~~~~~~~

crispyx auto-detects control labels (``ctrl``, ``NTC``, ``scramble``, etc.).
If your dataset uses a non-standard label, pass it explicitly:

.. code-block:: python

   adata = cx.pp.qc_summary(
       adata,
       perturbation_column="perturbation",
       control_label="my_control_name",
   )

Or use :func:`crispyx.normalise_perturbation_labels` to canonicalise labels
before analysis.

``UserWarning: CSC storage detected`` during NB-GLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NB-GLM requires CSR format. Convert first:

.. code-block:: python

   adata_csr = cx.pp.convert_to_csr(adata, output_dir="results/")
   result = cx.nb_glm_test(adata_csr, perturbation_column="perturbation")

HPC / SLURM tips
~~~~~~~~~~~~~~~~~

* Set ``memory_limit_gb`` to your SLURM ``--mem`` allocation.
* Use ``resume=True`` and ``checkpoint_interval=10`` for long jobs that may
  be preempted.
* ``drop_file_cache()`` is called automatically to prevent cgroup-cached
  pages from counting toward memory limits.
* See ``benchmarking/singularity/`` for SLURM submission scripts.

Performance tips
----------------

* **Pre-convert matrix formats** before DE: CSC for Wilcoxon, CSR for NB-GLM.
  This avoids O(total_nnz × n_chunks) scans.
* **Use ``freeze_control=True``** for datasets with >100K control cells to
  reduce per-worker memory from ~32 GB to <1 GB.
* **Increase ``n_jobs``** for multi-core NB-GLM on machines with sufficient
  RAM.
* **Use adaptive chunk sizes** (the default): let crispyx calculate optimal
  chunk sizes based on your ``memory_limit_gb``.
