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
import pytest
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, norm
import scanpy as sc

from streamlined_crispr import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
    quality_control_summary,
    wald_test,
    wilcoxon_test,
)
from streamlined_crispr.data import iter_matrix_chunks, normalize_total_block

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Adamson_subset.h5ad"
PERTURBATION_COLUMN = "batch_group"
MIN_GENES = 100
MIN_CELLS_PER_PERTURBATION = 15
MIN_CELLS_PER_GENE = 10
CHUNK_SIZE = 128


@pytest.fixture(scope="module")
def subset_dataset(tmp_path_factory):
    if not DATA_PATH.exists():
        pytest.skip(f"Sample dataset not found at {DATA_PATH}")

    adata = sc.read_h5ad(DATA_PATH)
    # Work on a small but representative subset to keep the test quick.
    n_obs = min(200, adata.n_obs)
    if n_obs < adata.n_obs:
        rng = np.random.default_rng(0)
        obs_indices = np.sort(rng.choice(adata.n_obs, size=n_obs, replace=False))
    else:
        obs_indices = np.arange(n_obs)
    var_slice = slice(0, min(500, adata.n_vars))
    subset = adata[obs_indices, var_slice].copy()

    if "Batch" in subset.obs:
        perturbation_source = subset.obs["Batch"]
    elif "batch" in subset.obs:
        perturbation_source = subset.obs["batch"]
    elif "perturbation" in subset.obs:
        perturbation_source = subset.obs["perturbation"]
    else:
        available = ", ".join(subset.obs.columns)
        raise KeyError(
            "No column available to derive perturbation labels. "
            f"Available columns: {available}"
        )

    subset.obs[PERTURBATION_COLUMN] = perturbation_source.astype(str)
    control_label = subset.obs[PERTURBATION_COLUMN].value_counts().idxmax()
    subset_path = tmp_path_factory.mktemp("scanpy_parity") / "subset.h5ad"
    subset.write(subset_path)
    return subset, subset_path, control_label


def _to_dense(matrix):
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def test_quality_control_matches_scanpy(subset_dataset, tmp_path):
    subset, subset_path, control_label = subset_dataset

    qc_result = quality_control_summary(
        subset_path,
        min_genes=MIN_GENES,
        min_cells_per_perturbation=MIN_CELLS_PER_PERTURBATION,
        min_cells_per_gene=MIN_CELLS_PER_GENE,
        perturbation_column=PERTURBATION_COLUMN,
        control_label=control_label,
        chunk_size=CHUNK_SIZE,
        output_dir=tmp_path,
        data_name="scanpy_parity",
    )

    raw = _to_dense(subset.X)
    cell_gene_counts = np.count_nonzero(raw, axis=1)
    scanpy_cell_mask = cell_gene_counts >= MIN_GENES

    labels = subset.obs[PERTURBATION_COLUMN].to_numpy(dtype=str)
    counts = pd.Series(labels[scanpy_cell_mask]).value_counts()
    keep_labels = set(counts[counts >= MIN_CELLS_PER_PERTURBATION].index)
    keep_labels.add(control_label)
    scanpy_cell_mask &= np.isin(labels, list(keep_labels))

    filtered_raw = raw[scanpy_cell_mask]
    gene_cell_counts = np.count_nonzero(filtered_raw, axis=0)
    scanpy_gene_mask = gene_cell_counts >= MIN_CELLS_PER_GENE

    np.testing.assert_array_equal(qc_result.cell_mask, scanpy_cell_mask)
    np.testing.assert_array_equal(qc_result.gene_mask, scanpy_gene_mask)
    assert qc_result.filtered_path.exists()


def test_normalisation_and_pseudobulk_match_scanpy(subset_dataset, tmp_path):
    subset, subset_path, control_label = subset_dataset

    qc_result = quality_control_summary(
        subset_path,
        min_genes=MIN_GENES,
        min_cells_per_perturbation=MIN_CELLS_PER_PERTURBATION,
        min_cells_per_gene=MIN_CELLS_PER_GENE,
        perturbation_column=PERTURBATION_COLUMN,
        control_label=control_label,
        chunk_size=CHUNK_SIZE,
        output_dir=tmp_path,
        data_name="scanpy_parity",
    )

    filtered = sc.read_h5ad(qc_result.filtered_path)

    normalised_blocks = []
    log_blocks = []
    for _, block in iter_matrix_chunks(filtered, axis=0, chunk_size=CHUNK_SIZE):
        normalised, _ = normalize_total_block(block)
        normalised_blocks.append(normalised)
        log_blocks.append(np.log1p(normalised))
    toolkit_norm = np.vstack(normalised_blocks)
    toolkit_log = np.vstack(log_blocks)

    scanpy_filtered = filtered.copy()
    sc.pp.normalize_total(scanpy_filtered, target_sum=1e4)
    scanpy_norm = _to_dense(scanpy_filtered.X)
    sc.pp.log1p(scanpy_filtered)
    scanpy_log = _to_dense(scanpy_filtered.X)

    np.testing.assert_allclose(toolkit_norm, scanpy_norm, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(toolkit_log, scanpy_log, atol=1e-8, rtol=1e-6)

    labels = filtered.obs[PERTURBATION_COLUMN].to_numpy(dtype=str)
    control_mask = labels == control_label
    control_mean_log = toolkit_log[control_mask].mean(axis=0)
    control_bulk = np.log1p(toolkit_norm[control_mask].mean(axis=0))

    avg_effects = compute_average_log_expression(
        qc_result.filtered_path,
        perturbation_column=PERTURBATION_COLUMN,
        control_label=control_label,
        gene_name_column="gene_symbols",
        chunk_size=CHUNK_SIZE,
        output_dir=tmp_path,
        data_name="scanpy_parity",
    )
    pseudo_effects = compute_pseudobulk_expression(
        qc_result.filtered_path,
        perturbation_column=PERTURBATION_COLUMN,
        control_label=control_label,
        gene_name_column="gene_symbols",
        chunk_size=CHUNK_SIZE,
        output_dir=tmp_path,
        data_name="scanpy_parity",
    )

    expected_avg = {}
    expected_pseudo = {}
    for label in avg_effects.index:
        mask = labels == label
        mean_log = toolkit_log[mask].mean(axis=0)
        expected_avg[label] = mean_log - control_mean_log
        bulk = np.log1p(toolkit_norm[mask].mean(axis=0))
        expected_pseudo[label] = bulk - control_bulk

    for label in avg_effects.index:
        np.testing.assert_allclose(
            avg_effects.loc[label].to_numpy(), expected_avg[label], atol=1e-8, rtol=1e-6
        )
        np.testing.assert_allclose(
            pseudo_effects.loc[label].to_numpy(), expected_pseudo[label], atol=1e-8, rtol=1e-6
        )


def test_differential_expression_matches_scanpy(subset_dataset, tmp_path):
    subset, subset_path, control_label = subset_dataset

    qc_result = quality_control_summary(
        subset_path,
        min_genes=MIN_GENES,
        min_cells_per_perturbation=MIN_CELLS_PER_PERTURBATION,
        min_cells_per_gene=MIN_CELLS_PER_GENE,
        perturbation_column=PERTURBATION_COLUMN,
        control_label=control_label,
        chunk_size=CHUNK_SIZE,
        output_dir=tmp_path,
        data_name="scanpy_parity",
    )

    filtered = sc.read_h5ad(qc_result.filtered_path)
    labels = filtered.obs[PERTURBATION_COLUMN].to_numpy(dtype=str)
    toolkit_norm = []
    toolkit_log = []
    for _, block in iter_matrix_chunks(filtered, axis=0, chunk_size=CHUNK_SIZE):
        normalised, _ = normalize_total_block(block)
        toolkit_norm.append(normalised)
        toolkit_log.append(np.log1p(normalised))
    toolkit_norm = np.vstack(toolkit_norm)
    toolkit_log = np.vstack(toolkit_log)

    raw = _to_dense(filtered.X)
    control_mask = labels == control_label
    control_n = int(control_mask.sum())
    control_log = toolkit_log[control_mask]
    control_mean = control_log.mean(axis=0)
    control_var = np.zeros_like(control_mean)
    if control_n > 1:
        control_var = control_log.var(axis=0, ddof=1)

    expr_control = np.count_nonzero(raw[control_mask], axis=0)

    expected_wald = {}
    expected_wald_z = {}
    expected_wald_p = {}

    for label in np.unique(labels):
        if label == control_label:
            continue
        mask = labels == label
        group_log = toolkit_log[mask]
        n_cells = int(mask.sum())
        group_mean = group_log.mean(axis=0)
        group_var = np.zeros_like(group_mean)
        if n_cells > 1:
            group_var = group_log.var(axis=0, ddof=1)
        effect = group_mean - control_mean
        se = np.sqrt(control_var / max(control_n, 1) + group_var / max(n_cells, 1))
        expr_group = np.count_nonzero(raw[mask], axis=0)
        total_expr = expr_control + expr_group
        valid = (se > 0) & (total_expr >= 0)
        z = np.zeros_like(effect)
        p = np.ones_like(effect)
        z[valid] = effect[valid] / se[valid]
        p[valid] = 2 * norm.sf(np.abs(z[valid]))
        expected_wald[label] = effect
        expected_wald_z[label] = z
        expected_wald_p[label] = p

    wald_results = wald_test(
        qc_result.filtered_path,
        perturbation_column=PERTURBATION_COLUMN,
        control_label=control_label,
        gene_name_column="gene_symbols",
        chunk_size=CHUNK_SIZE,
        output_dir=tmp_path,
        data_name="scanpy_parity",
    )

    for label, expected in expected_wald.items():
        result = wald_results[label]
        np.testing.assert_allclose(result.effect_size, expected, atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(result.statistic, expected_wald_z[label], atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(result.pvalue, expected_wald_p[label], atol=1e-8, rtol=1e-6)

    wilcoxon_results = wilcoxon_test(
        qc_result.filtered_path,
        perturbation_column=PERTURBATION_COLUMN,
        control_label=control_label,
        gene_name_column="gene_symbols",
        chunk_size=64,
        output_dir=tmp_path,
        data_name="scanpy_parity",
    )

    control_values = toolkit_norm[control_mask]
    control_expr = np.count_nonzero(control_values, axis=0)

    for label, result in wilcoxon_results.items():
        mask = labels == label
        pert_values = toolkit_norm[mask]
        pert_expr = np.count_nonzero(pert_values, axis=0)
        total_expr = control_expr + pert_expr
        expected_effect = np.zeros_like(result.effect_size)
        expected_stat = np.zeros_like(result.statistic)
        expected_pvalue = np.ones_like(result.pvalue)
        for gene_idx in range(toolkit_norm.shape[1]):
            if total_expr[gene_idx] < 0:
                continue
            res = mannwhitneyu(
                control_values[:, gene_idx],
                pert_values[:, gene_idx],
                alternative="two-sided",
                method="auto",
            )
            expected_stat[gene_idx] = res.statistic
            expected_pvalue[gene_idx] = res.pvalue
            expected_effect[gene_idx] = (
                res.statistic / (control_values.shape[0] * pert_values.shape[0]) - 0.5
            )
        np.testing.assert_allclose(result.effect_size, expected_effect, atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(result.statistic, expected_stat, atol=1e-8, rtol=1e-6)
        np.testing.assert_allclose(result.pvalue, expected_pvalue, atol=1e-8, rtol=1e-6)
