"""Regression tests to ensure streaming normalisation matches Scanpy."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
    
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import h5py
from scipy.stats import norm, rankdata

from streamlined_crispr.data import ensure_gene_symbol_column, normalize_total_block
from streamlined_crispr.de import _tie_correction, wald_test, wilcoxon_test
from streamlined_crispr.pseudobulk import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
)


@pytest.fixture
def small_adata(tmp_path):
    counts = np.array(
        [
            [1, 0, 3, 0],
            [0, 2, 1, 0],
            [4, 0, 0, 1],
            [3, 1, 0, 0],
            [0, 0, 2, 2],
            [1, 0, 1, 3],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "perturbation": ["ctrl", "ctrl", "A", "A", "B", "B"],
        },
        index=[f"cell_{i}" for i in range(counts.shape[0])],
    )
    var = pd.DataFrame(
        {"gene_symbols": ["G1", "G2", "G3", "G4"]},
        index=[f"g{i}" for i in range(counts.shape[1])],
    )
    adata = ad.AnnData(counts, obs=obs, var=var)
    path = tmp_path / "small.h5ad"
    adata.write(path)
    return path, adata


def test_average_log_expression_matches_scanpy(small_adata, tmp_path):
    path, adata = small_adata
    result = compute_average_log_expression(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        chunk_size=2,
        output_dir=tmp_path,
    )

    output_path = tmp_path / "small_avg_log_effects.h5ad"
    assert output_path.exists()

    result_mem = result.to_memory()
    result_df = pd.DataFrame(
        result_mem.X, index=result_mem.obs.index, columns=result_mem.var_names
    )

    sc_adata = adata.copy()
    sc.pp.normalize_total(sc_adata, target_sum=1e4)
    sc.pp.log1p(sc_adata)

    ctrl_mask = sc_adata.obs["perturbation"] == "ctrl"
    ctrl_mean = np.asarray(sc_adata[ctrl_mask].X).mean(axis=0)
    expected = {}
    for label in result_df.index:
        mask = sc_adata.obs["perturbation"] == label
        mean = np.asarray(sc_adata[mask].X).mean(axis=0)
        expected[label] = mean - ctrl_mean
    expected_df = pd.DataFrame(expected).T
    expected_df.columns = sc_adata.var["gene_symbols"].to_list()
    expected_df = expected_df.loc[result_df.index, result_df.columns]

    np.testing.assert_allclose(result_df.to_numpy(), expected_df.to_numpy(), rtol=1e-8, atol=1e-8)
    result.close()


def test_pseudobulk_expression_matches_scanpy(small_adata, tmp_path):
    path, adata = small_adata
    result = compute_pseudobulk_expression(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        chunk_size=2,
        output_dir=tmp_path,
    )

    output_path = tmp_path / "small_pseudobulk_effects.h5ad"
    assert output_path.exists()

    result_mem = result.to_memory()
    result_df = pd.DataFrame(
        result_mem.X, index=result_mem.obs.index, columns=result_mem.var_names
    )

    sc_adata = adata.copy()
    sc.pp.normalize_total(sc_adata, target_sum=1e4)

    ctrl_mask = sc_adata.obs["perturbation"] == "ctrl"
    ctrl_mean = np.asarray(sc_adata[ctrl_mask].X).mean(axis=0)
    expected = {}
    for label in result_df.index:
        mask = sc_adata.obs["perturbation"] == label
        mean = np.asarray(sc_adata[mask].X).mean(axis=0)
        expected[label] = np.log1p(mean) - np.log1p(ctrl_mean)
    expected_df = pd.DataFrame(expected).T
    expected_df.columns = sc_adata.var["gene_symbols"].to_list()
    expected_df = expected_df.loc[result_df.index, result_df.columns]

    np.testing.assert_allclose(result_df.to_numpy(), expected_df.to_numpy(), rtol=1e-8, atol=1e-8)
    result.close()


def test_wald_test_matches_scanpy(small_adata, tmp_path):
    path, adata = small_adata
    results = wald_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        chunk_size=2,
        output_dir=tmp_path,
    )

    output_path = tmp_path / "small_wald_de.h5ad"
    assert output_path.exists()

    sc_adata = adata.copy()
    sc.pp.normalize_total(sc_adata, target_sum=1e4)
    sc.pp.log1p(sc_adata)

    ctrl_mask = sc_adata.obs["perturbation"] == "ctrl"
    ctrl_data = np.asarray(sc_adata[ctrl_mask].X)
    ctrl_mean = ctrl_data.mean(axis=0)
    if ctrl_data.shape[0] > 1:
        ctrl_var = ctrl_data.var(axis=0, ddof=1)
    else:
        ctrl_var = np.zeros_like(ctrl_mean)
    ctrl_expr = np.asarray((adata[ctrl_mask].X > 0).sum(axis=0)).ravel()

    for label, result in results.items():
        mask = sc_adata.obs["perturbation"] == label
        pert_data = np.asarray(sc_adata[mask].X)
        mean = pert_data.mean(axis=0)
        if pert_data.shape[0] > 1:
            var = pert_data.var(axis=0, ddof=1)
        else:
            var = np.zeros_like(mean)
        effect = mean - ctrl_mean
        se = np.sqrt(ctrl_var / ctrl_data.shape[0] + var / pert_data.shape[0])
        pert_expr = np.asarray((adata[mask].X > 0).sum(axis=0)).ravel()
        total_expr = ctrl_expr + pert_expr
        valid = se > 0
        z = np.zeros_like(effect)
        pvalue = np.ones_like(effect)
        z[valid] = effect[valid] / se[valid]
        pvalue[valid] = 2 * norm.sf(np.abs(z[valid]))

        np.testing.assert_allclose(result.effect_size, effect, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(result.statistic, z, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(result.pvalue, pvalue, rtol=1e-8, atol=1e-8)
        assert result.result_path == output_path


def test_wilcoxon_test_matches_scanpy(small_adata, tmp_path):
    path, adata = small_adata
    results = wilcoxon_test(
        path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        chunk_size=2,
        output_dir=tmp_path,
    )

    output_path = tmp_path / "small_wilcoxon_de.h5ad"
    assert output_path.exists()

    raw = np.asarray(adata.X)
    norm_data, _ = normalize_total_block(raw)
    log_data = np.log1p(norm_data)
    ctrl_mask = adata.obs["perturbation"] == "ctrl"
    control_values = log_data[ctrl_mask]
    control_n = float(control_values.shape[0])

    for label, result in results.items():
        mask = adata.obs["perturbation"] == label
        pert_values = log_data[mask]
        pert_n = float(pert_values.shape[0])

        effects = []
        stats = []
        pvals = []
        for gene in range(adata.n_vars):
            combined = np.concatenate([pert_values[:, gene], control_values[:, gene]])[:, None]
            ranks = rankdata(combined, axis=0)
            rank_sum = float(ranks[: pert_values.shape[0]].sum())
            tie = float(_tie_correction(ranks)[0])
            expected = pert_n * (pert_n + control_n + 1.0) / 2.0
            std = float(np.sqrt(tie * pert_n * control_n * (pert_n + control_n + 1.0) / 12.0))
            u_stat = float(rank_sum - pert_n * (pert_n + 1.0) / 2.0)
            effect = u_stat / (control_n * pert_n) - 0.5
            effects.append(effect)
            if std == 0 or np.isnan(std):
                stats.append(0.0)
                pvals.append(1.0)
            else:
                z = (rank_sum - expected) / std
                stats.append(float(z))
                pvals.append(float(2 * norm.sf(abs(z))))

        np.testing.assert_allclose(result.effect_size, np.asarray(effects), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(result.statistic, np.asarray(stats), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(result.pvalue, np.asarray(pvals), atol=1e-9, rtol=1e-7)
        assert result.result_path == output_path


def test_ensure_gene_symbol_column_fallback_to_index():
    counts = np.zeros((2, 2))
    obs = pd.DataFrame(index=["cell1", "cell2"])
    var = pd.DataFrame(
        {"Ensembl_ID": ["ENS0001", "ENS0002"]},
        index=["GeneA", "GeneB"],
    )
    adata = ad.AnnData(counts, obs=obs, var=var)

    names = ensure_gene_symbol_column(adata, "gene_symbols")

    assert list(names) == ["GeneA", "GeneB"]
