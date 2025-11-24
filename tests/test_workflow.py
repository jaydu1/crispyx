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
import h5py

import crispyx as cx

from crispyx import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
    quality_control_summary,
    t_test,
    wilcoxon_test,
)
from crispyx.data import normalize_total_block


def create_test_dataset(tmp_path):
    x = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 3, 0],
            [0, 1, 0, 0],
            [2, 1, 0, 4],
            [0, 0, 0, 5],
            [1, 0, 0, 6],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {"perturbation": ["ctrl", "ctrl", "KO1", "KO1", "KO2", "KO2"]},
        index=[f"cell_{i}" for i in range(x.shape[0])],
    )
    var = pd.DataFrame({"gene_symbol": [f"gene{i}" for i in range(x.shape[1])]})
    var.index = var["gene_symbol"]
    adata = ad.AnnData(x, obs=obs, var=var)
    path = tmp_path / "test.h5ad"
    adata.write(path)
    return path, adata


def create_sparse_test_dataset(tmp_path):
    dense_path, dense = create_test_dataset(tmp_path)
    sparse = ad.AnnData(sp.csr_matrix(dense.X), obs=dense.obs.copy(), var=dense.var.copy())
    sparse_path = tmp_path / "test_sparse.h5ad"
    sparse.write(sparse_path)
    return sparse_path, sparse


def _normalize_total(matrix: np.ndarray, *, target_sum: float = 1e4) -> np.ndarray:
    """Replicate the package's library-size normalisation."""

    matrix = np.asarray(matrix, dtype=float)
    library_size = matrix.sum(axis=1, keepdims=True)
    scale = np.divide(
        float(target_sum),
        library_size,
        out=np.zeros_like(library_size, dtype=float),
        where=library_size > 0,
    )
    return matrix * scale


def test_quality_control_writes_filtered_dataset(tmp_path):
    path, adata = create_test_dataset(tmp_path)
    result = quality_control_summary(
        path,
        min_genes=1,
        min_cells_per_perturbation=2,
        min_cells_per_gene=1,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbol",
        output_dir=tmp_path,
        data_name="qc_test",
    )
    assert isinstance(result.filtered, cx.AnnData)
    assert result.filtered_path.exists()
    filtered = result.filtered.to_memory()
    assert filtered.n_obs == int(result.cell_mask.sum())
    assert filtered.n_vars == int(result.gene_mask.sum())
    assert filtered.var["gene_symbols"].tolist() == adata.var["gene_symbol"].tolist()
    result.filtered.close()


def test_quality_control_sparse_roundtrip(tmp_path):
    path, adata = create_sparse_test_dataset(tmp_path)
    result = quality_control_summary(
        path,
        min_genes=1,
        min_cells_per_perturbation=2,
        min_cells_per_gene=1,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbol",
        output_dir=tmp_path,
        data_name="qc_sparse",
    )
    filtered = result.filtered.to_memory()
    expected = adata[result.cell_mask, result.gene_mask]
    np.testing.assert_array_equal(filtered.X.toarray(), expected.X.toarray())
    with h5py.File(result.filtered_path) as handle:
        assert handle["X"].attrs["encoding-type"] == b"csr_matrix"
    result.filtered.close()


def test_gene_symbol_validation(tmp_path):
    x = np.ones((5, 3))
    obs = pd.DataFrame({"perturbation": ["ctrl"] * 5})
    obs.index = [f"cell_{i}" for i in range(obs.shape[0])]
    var = pd.DataFrame(index=["ENSG000001", "ENSG000002", "ENSG000003"])
    adata = ad.AnnData(x, obs=obs, var=var)
    path = tmp_path / "invalid.h5ad"
    adata.write(path)
    try:
        quality_control_summary(
            path,
            min_genes=1,
            min_cells_per_perturbation=1,
            min_cells_per_gene=1,
            perturbation_column="perturbation",
            control_label="ctrl",
        )
    except ValueError as exc:
        assert "Ensembl" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for Ensembl-style identifiers")


def test_downstream_effect_outputs(tmp_path):
    path, _ = create_test_dataset(tmp_path)
    qc_result = quality_control_summary(
        path,
        min_genes=1,
        min_cells_per_perturbation=2,
        min_cells_per_gene=1,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbol",
        output_dir=tmp_path,
        data_name="de",
    )
    avg = compute_average_log_expression(
        qc_result.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="avg_effects",
    )
    pseudo = compute_pseudobulk_expression(
        qc_result.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="pseudo_effects",
    )
    wald = t_test(
        qc_result.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="t_test",
    )
    wilcoxon = wilcoxon_test(
        qc_result.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wilcoxon",
    )

    assert isinstance(avg, cx.AnnData)
    assert isinstance(pseudo, cx.AnnData)
    avg_mem = avg.to_memory()
    pseudo_mem = pseudo.to_memory()
    avg_df = pd.DataFrame(avg_mem.X, index=avg_mem.obs.index, columns=avg_mem.var_names)
    pseudo_df = pd.DataFrame(
        pseudo_mem.X, index=pseudo_mem.obs.index, columns=pseudo_mem.var_names
    )
    assert set(avg_df.index) == {"KO1", "KO2"}
    assert set(pseudo_df.index) == {"KO1", "KO2"}
    assert set(wald.keys()) == {"KO1", "KO2"}
    assert set(wilcoxon.keys()) == {"KO1", "KO2"}

    filtered = qc_result.filtered.to_memory()
    ctrl_mask = (filtered.obs["perturbation"] == "ctrl").to_numpy()
    ko1_mask = (filtered.obs["perturbation"] == "KO1").to_numpy()
    normalised, _ = normalize_total_block(filtered.X)
    log_block = np.log1p(normalised)
    ctrl = log_block[ctrl_mask, 0]
    ko1 = log_block[ko1_mask, 0]
    expected = ko1.mean() - ctrl.mean()
    assert np.isclose(avg_df.loc["KO1", "gene0"], expected)

    assert (tmp_path / "crispyx_avg_effects_avg_log_effects.h5ad").exists()
    assert (tmp_path / "crispyx_pseudo_effects_pseudobulk_effects.h5ad").exists()
    assert (tmp_path / "crispyx_t_test.h5ad").exists()
    assert (tmp_path / "crispyx_wilcoxon.h5ad").exists()

    ko1_result = wald["KO1"]
    assert ko1_result.effect_size.shape[0] == 4
    assert ko1_result.method == "t_test"
    ko2_result = wilcoxon["KO2"]
    assert ko2_result.pvalue.shape[0] == 4
    assert ko2_result.method == "wilcoxon"
    avg.close()
    pseudo.close()
    qc_result.filtered.close()


def test_scanpy_style_namespaces_match_direct(tmp_path):
    path, _ = create_test_dataset(tmp_path)
    adata_ro = ad.read_h5ad(path, backed="r")
    try:
        qc_wrapped = cx.pp.qc_summary(
            adata_ro,
            min_genes=1,
            min_cells_per_perturbation=2,
            min_cells_per_gene=1,
            perturbation_column="perturbation",
            control_label="ctrl",
            gene_name_column="gene_symbol",
            output_dir=tmp_path,
            data_name="wrapped",
        )
    finally:
        adata_ro.file.close()

    qc_direct = quality_control_summary(
        path,
        min_genes=1,
        min_cells_per_perturbation=2,
        min_cells_per_gene=1,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbol",
        output_dir=tmp_path,
        data_name="direct",
    )

    assert isinstance(qc_wrapped, cx.AnnData)
    qc_wrapped_mem = qc_wrapped.to_memory()
    filtered_direct = qc_direct.filtered.to_memory()
    pd.testing.assert_index_equal(qc_wrapped_mem.obs.index, filtered_direct.obs.index)
    pd.testing.assert_index_equal(qc_wrapped_mem.var_names, filtered_direct.var_names)

    avg_wrapped = cx.pb.average_log_expression(
        qc_wrapped,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wrapped_avg",
    )
    avg_direct = compute_average_log_expression(
        qc_direct.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="direct_avg",
    )
    avg_wrapped_mem = avg_wrapped.to_memory()
    avg_direct_mem = avg_direct.to_memory()
    np.testing.assert_allclose(avg_wrapped_mem.X, avg_direct_mem.X)
    pd.testing.assert_index_equal(avg_wrapped_mem.obs.index, avg_direct_mem.obs.index)
    pd.testing.assert_index_equal(avg_wrapped_mem.var_names, avg_direct_mem.var_names)

    pseudo_wrapped = cx.pb.pseudobulk(
        qc_wrapped,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wrapped_pseudo",
    )
    pseudo_direct = compute_pseudobulk_expression(
        qc_direct.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="direct_pseudo",
    )
    pseudo_wrapped_mem = pseudo_wrapped.to_memory()
    pseudo_direct_mem = pseudo_direct.to_memory()
    np.testing.assert_allclose(pseudo_wrapped_mem.X, pseudo_direct_mem.X)
    pd.testing.assert_index_equal(
        pseudo_wrapped_mem.obs.index, pseudo_direct_mem.obs.index
    )
    pd.testing.assert_index_equal(
        pseudo_wrapped_mem.var_names, pseudo_direct_mem.var_names
    )

    wald_wrapped = cx.tl.rank_genes_groups(
        qc_wrapped,
        perturbation_column="perturbation",
        method="t-test",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wrapped_wald",
        min_cells_expressed=0,
    )
    wald_direct = t_test(
        qc_direct.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="direct_wald",
        min_cells_expressed=0,
    )
    wald_wrapped_uns = wald_wrapped.uns["rank_genes_groups"].load()
    wald_wrapped_full = wald_wrapped_uns["full"]
    wrapped_groups = list(wald_wrapped_uns["names"].dtype.names)
    direct_effect = np.vstack([wald_direct[group].effect_size for group in wrapped_groups])
    direct_stat = np.vstack([wald_direct[group].statistic for group in wrapped_groups])
    direct_p = np.vstack([wald_direct[group].pvalue for group in wrapped_groups])
    np.testing.assert_allclose(wald_wrapped_full["logfoldchanges"], direct_effect)
    np.testing.assert_allclose(wald_wrapped_full["scores"], direct_stat)
    np.testing.assert_allclose(wald_wrapped_full["pvals"], direct_p)

    wilcoxon_wrapped = cx.tl.rank_genes_groups(
        qc_wrapped,
        perturbation_column="perturbation",
        method="wilcoxon",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wrapped_wilcoxon",
        min_cells_expressed=0,
    )
    wilcoxon_direct = wilcoxon_test(
        qc_direct.filtered,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="direct_wilcoxon",
        min_cells_expressed=0,
    )
    wilcoxon_wrapped_uns = wilcoxon_wrapped.uns["rank_genes_groups"].load()
    wrapped_groups = list(wilcoxon_wrapped_uns["names"].dtype.names)
    direct_order = wilcoxon_direct.to_full_order_dict()
    full_wrapped = wilcoxon_wrapped_uns["full"]
    np.testing.assert_allclose(full_wrapped["scores"], direct_order["scores"])
    np.testing.assert_allclose(full_wrapped["pvals"], direct_order["pvals"])
    assert set(wrapped_groups) == set(wilcoxon_direct.groups)
    avg_wrapped.close()
    avg_direct.close()
    pseudo_wrapped.close()
    pseudo_direct.close()
    wald_wrapped.close()
    wilcoxon_wrapped.close()
    qc_wrapped.close()
    qc_direct.filtered.close()
