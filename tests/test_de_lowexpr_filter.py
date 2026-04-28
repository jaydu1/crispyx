"""Tests for the per-condition low-expression filter applied to DE tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import pytest

import crispyx as cx
from crispyx._statistics import _low_expr_in_both_mask


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------

def test_low_expr_mask_disabled_by_zero_thresholds():
    n_genes = 6
    expr_p = np.array([0, 0, 5, 5, 0, 5])
    expr_c = np.array([0, 0, 0, 5, 5, 5])
    mean_p = np.zeros(n_genes)
    mean_c = np.zeros(n_genes)
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr_p,
        control_expr_counts=expr_c,
        pert_mean=mean_p,
        control_mean=mean_c,
        n_pert_cells=10,
        n_control_cells=10,
        min_pct_both=0.0,
        min_mean_both=0.0,
    )
    assert mask.dtype == bool
    assert not mask.any(), "Filter must be a no-op when both thresholds are 0"


def test_low_expr_mask_drops_only_jointly_low_genes():
    # 4 genes, 100 cells per group.
    expr_p = np.array([0, 0,   50, 0])
    expr_c = np.array([0, 50,  0,  0])
    mean_p = np.array([0.0, 0.0, 1.0, 0.0])
    mean_c = np.array([0.0, 1.0, 0.0, 0.0])
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr_p,
        control_expr_counts=expr_c,
        pert_mean=mean_p,
        control_mean=mean_c,
        n_pert_cells=100,
        n_control_cells=100,
        min_pct_both=0.01,
        min_mean_both=0.05,
    )
    # Gene 0: zero everywhere -> drop.
    # Gene 1: expressed in control -> keep.
    # Gene 2: expressed in pert    -> keep.
    # Gene 3: zero everywhere -> drop.
    assert mask.tolist() == [True, False, False, True]


def test_low_expr_mask_requires_both_metrics_to_drop():
    # Many cells (passes pct) but mean below threshold:
    # should still pass (only one metric fails) -> NOT dropped.
    expr_p = np.array([20, 0])
    expr_c = np.array([20, 0])
    mean_p = np.array([0.001, 0.0])  # below mean threshold
    mean_c = np.array([0.001, 0.0])
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr_p,
        control_expr_counts=expr_c,
        pert_mean=mean_p,
        control_mean=mean_c,
        n_pert_cells=100,
        n_control_cells=100,
        min_pct_both=0.01,
        min_mean_both=0.05,
    )
    # Gene 0: pct = 0.2 (passes), mean below -> low_p/low_c is False (pct passes)
    #   -> NOT dropped.
    # Gene 1: zero everywhere -> dropped.
    assert mask.tolist() == [False, True]


def test_low_expr_mask_handles_empty_groups():
    expr = np.array([0, 5])
    mean = np.array([0.0, 1.0])
    mask = _low_expr_in_both_mask(
        pert_expr_counts=expr,
        control_expr_counts=expr,
        pert_mean=mean,
        control_mean=mean,
        n_pert_cells=0,
        n_control_cells=10,
    )
    assert not mask.any()


# ---------------------------------------------------------------------------
# End-to-end tests on tiny datasets
# ---------------------------------------------------------------------------

def _make_dataset(tmp_path: Path, *, log_normalise: bool):
    """Build a small AnnData where one gene is jointly silent in BOTH groups."""

    rng = np.random.default_rng(0)
    n_ctrl = 80
    n_pert = 80
    n_cells = n_ctrl + n_pert
    n_genes = 5

    # Gene 0: well expressed in both groups
    # Gene 1: differentially expressed (high in pert, low in ctrl)
    # Gene 2: differentially expressed (high in ctrl, low in pert)
    # Gene 3: SILENT in both groups (should be filtered out)
    # Gene 4: low-but-real signal in both groups
    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    counts[:n_ctrl, 0] = rng.poisson(20, n_ctrl)
    counts[n_ctrl:, 0] = rng.poisson(20, n_pert)
    counts[:n_ctrl, 1] = rng.poisson(2, n_ctrl)
    counts[n_ctrl:, 1] = rng.poisson(15, n_pert)
    counts[:n_ctrl, 2] = rng.poisson(15, n_ctrl)
    counts[n_ctrl:, 2] = rng.poisson(2, n_pert)
    # gene 3: leave all zeros (silent in both)
    counts[:n_ctrl, 4] = rng.poisson(5, n_ctrl)
    counts[n_ctrl:, 4] = rng.poisson(5, n_pert)

    obs = pd.DataFrame(
        {"perturbation": ["ctrl"] * n_ctrl + ["KO1"] * n_pert},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"gene_symbol": [f"gene{i}" for i in range(n_genes)]},
        index=[f"gene{i}" for i in range(n_genes)],
    )
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    if log_normalise:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        adata.X = sp.csr_matrix(adata.X)
    path = tmp_path / ("ds_log.h5ad" if log_normalise else "ds_raw.h5ad")
    adata.write(path)
    return path


def test_t_test_excludes_jointly_silent_gene(tmp_path):
    path = _make_dataset(tmp_path, log_normalise=True)
    res = cx.t_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_both=0.01,
        min_mean_both=0.05,
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    lfc = np.asarray(res.logfoldchanges[pert_idx])
    # Gene 3 is silent in both groups -> NaN in pvalue and logfc
    assert np.isnan(pvals[3]), f"Expected NaN for gene 3, got {pvals[3]}"
    assert np.isnan(lfc[3]), f"Expected NaN logfc for gene 3, got {lfc[3]}"
    # Other genes should have finite p-values
    finite_other = np.isfinite(pvals[[0, 1, 2, 4]])
    assert finite_other.all(), f"Non-silent genes should have finite p-values: {pvals}"


def test_wilcoxon_excludes_jointly_silent_gene(tmp_path):
    path = _make_dataset(tmp_path, log_normalise=True)
    res = cx.wilcoxon_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_both=0.01,
        min_mean_both=0.05,
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    assert np.isnan(pvals[3]), f"Expected NaN for gene 3, got {pvals[3]}"
    finite_other = np.isfinite(pvals[[0, 1, 2, 4]])
    assert finite_other.all()


def test_t_test_disabled_filter_recovers_legacy_behaviour(tmp_path):
    """With both thresholds at 0 the filter is inert."""
    path = _make_dataset(tmp_path, log_normalise=True)
    res = cx.t_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_both=0.0,
        min_mean_both=0.0,
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    # With filter off, no NaN should appear from low-expression masking.
    # Gene 3 has zero variance in both groups -> SE=0 -> still NaN by the
    # original valid-mask logic (or 1.0 in legacy code). Allow either.
    assert np.isnan(pvals[3]) or pvals[3] == 1.0


def test_nb_glm_excludes_jointly_silent_gene(tmp_path):
    path = _make_dataset(tmp_path, log_normalise=False)  # raw counts
    res = cx.nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_both=0.01,
        min_mean_both=0.05,
        n_jobs=1,
        verbose=False,
    )
    pert_idx = res.groups.index("KO1")
    pvals = np.asarray(res.pvalues[pert_idx])
    assert np.isnan(pvals[3]), f"Expected NaN for gene 3, got {pvals[3]}"


def test_filter_thresholds_change_filtering(tmp_path):
    """A tight threshold drops more genes than a loose one."""
    path = _make_dataset(tmp_path, log_normalise=True)
    loose = cx.t_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_both=0.0,
        min_mean_both=0.0,
    )
    # Strict: pct<0.99 AND mean<1e6 in both groups effectively drops every
    # gene whose pct is below 99% in both groups.
    strict = cx.t_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        min_pct_both=0.99,
        min_mean_both=1e6,
    )
    pert_idx = strict.groups.index("KO1")
    n_strict_nan = int(np.isnan(strict.pvalues[pert_idx]).sum())
    n_loose_nan = int(np.isnan(loose.pvalues[pert_idx]).sum())
    assert n_strict_nan > n_loose_nan, (
        f"Strict threshold should NaN more genes (loose={n_loose_nan}, strict={n_strict_nan})"
    )
