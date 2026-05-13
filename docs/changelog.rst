Changelog
=========

Version 0.0.4
-------------

*Released 2026-05-13.*

* **Decoupled per-condition pct thresholds** – ``min_pct_both`` is replaced
  by independent ``min_pct_ctrl`` (default ``0.01``) and ``min_pct_pert``
  (default ``0.002``) parameters across all three DE test functions
  (``t_test``, ``wilcoxon_test``, ``nb_glm_test``) and the internal
  ``_low_expr_in_both_mask`` helper.  The lower ``min_pct_pert`` default
  prevents over-filtering genes induced from near-zero baseline
  (e.g. transcription-factor target genes).  The old ``min_pct_both``
  kwarg is retained as a deprecated alias that overrides both new params and
  emits a ``DeprecationWarning``; it will be removed in a future release.

* **Dual-condition pert filter with enabled ``min_mean_pert``** – The
  perturbed-side filter now always applies a dual condition:
  ``(pct_p < min_pct_pert) AND (mean_p < min_mean_pert)``.  The default
  ``min_mean_pert`` is raised from ``0.0`` (v0.0.3) to ``0.005`` so that
  genes with very few but high-count expressing cells (possible doublets or
  ambient RNA) are correctly excluded.  Existing code can restore the
  v0.0.3 behaviour by passing ``min_mean_pert=0.0``.

* **NaN initialisation for filtered-gene p-values (Wilcoxon)** – The
  standard single-pass Wilcoxon path previously initialised the chunk
  p-value array with ``np.ones`` (p=1.0) rather than ``np.nan``, causing
  filtered genes to appear as nominally non-significant rather than missing.
  The array is now initialised with ``np.full(..., np.nan)``, consistent
  with the streaming path and with ``t_test`` / ``nb_glm_test``.

Version 0.0.3
-------------

*Released 2026-05-13.*

* **Auto-reload for DE results** – ``wilcoxon_test``, ``t_test``, and
  ``nb_glm_test`` now accept a ``force: bool = False`` parameter.  When
  ``False`` (default) and the expected output ``.h5ad`` file already exists on
  disk, the functions load and return the saved result instead of rerunning the
  analysis.  Set ``force=True`` to rerun unconditionally and overwrite the
  existing file.  Combined with ``verbose=True``, a notice is printed to
  stdout identifying the reloaded file path.

* **Fixed ``RecursionError`` when pickling DE results** – ``AnnData.__getattr__``
  now guards against access before ``__init__`` has run (e.g. during
  ``pickle.load``), eliminating infinite recursion.  ``AnnData`` gains
  ``__getstate__`` / ``__setstate__`` so only the file path and access mode are
  serialised; the HDF5 handle is reopened lazily after unpickling.
  ``RankGenesGroupsResult`` and ``DifferentialExpressionResult`` likewise gain
  ``__getstate__`` / ``__setstate__`` that exclude the ``AnnData`` handle and
  group cache from the pickle payload, allowing round-trip serialisation with
  ``pickle.dumps`` / ``pickle.loads``.

* **Asymmetric low-expression filter** – DE tests (t-test, Wilcoxon, NB-GLM)
  now accept a ``min_mean_pert`` parameter (default ``0.0``). With the
  default, the mean-expression check is applied only to the *control* group;
  the perturbed group is filtered on fraction-of-expressing-cells
  (``min_pct_both``) alone. This prevents the filter from discarding genes
  that are induced from near-zero baseline expression, which is common in
  unbalanced CRISPR-screen comparisons. To reproduce the v0.0.2 behaviour
  pass ``min_mean_pert=min_mean_ctrl`` (e.g. ``min_mean_pert=0.05``).

Version 0.0.2
-------------

*Released 2026-04-28.*

* **Per-condition low-expression filter for DE tests** – t-test, Wilcoxon, and
  NB-GLM now accept ``min_pct_both`` (default ``0.01``) and ``min_mean_both``
  (default ``0.05``) parameters. A gene is excluded from a perturbation
  comparison (reported as NaN in ``pvalue`` / ``effect`` / ``logfoldchanges``)
  when the fraction of expressing cells *and* the mean expression are both
  below the respective thresholds in *both* the perturbation and control
  groups. Setting both thresholds to ``0.0`` recovers the 0.0.1 behaviour
  exactly. ``pts`` and mean expression values are always retained.

Version 0.0.1
-------------

*Initial release.*

* Streaming QC and preprocessing (filter cells, perturbations, genes;
  normalize and log-transform without loading the full matrix)
* Pseudo-bulk aggregation: average log expression and pseudo-bulk count
  matrices
* Differential expression: t-test, Wilcoxon rank-sum, NB-GLM with apeGLM
  LFC shrinkage, multi-core support, and adaptive memory management
* Dimension reduction: memory-efficient PCA and KNN graph construction on
  backed data
* Scanpy-compatible API and plotting: ``cx.pp``, ``cx.pb``, ``cx.tl``,
  ``cx.pl`` namespaces; rank genes plots, volcano, MA, PCA, UMAP, QC
  summaries, and overlap heatmaps
* Data preparation utilities: edit backed metadata, standardise gene names,
  normalise perturbation labels, auto-detect metadata columns
* HPC support: resume/checkpoint for long-running jobs, configurable
  ``memory_limit_gb``, Docker and Singularity support
* Benchmarking suite across 12 CRISPR screen datasets
