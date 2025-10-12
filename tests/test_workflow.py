from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _require(module_name: str):
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(
            f"Missing dependency '{module_name}'. Install project test extras via 'pip install .[test]'"
        ) from exc


np = _require("numpy")
pd = _require("pandas")
ad = _require("anndata")
sp = _require("scipy.sparse")
h5py = _require("h5py")

from streamlined_crispr import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
    quality_control_summary,
    wald_test,
    wilcoxon_test,
)


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
    obs = pd.DataFrame({"perturbation": ["ctrl", "ctrl", "KO1", "KO1", "KO2", "KO2"]})
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
    assert result.filtered_path.exists()
    filtered = ad.read_h5ad(result.filtered_path)
    assert filtered.n_obs == int(result.cell_mask.sum())
    assert filtered.n_vars == int(result.gene_mask.sum())
    assert filtered.var["gene_symbols"].tolist() == adata.var["gene_symbol"].tolist()


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
    filtered = ad.read_h5ad(result.filtered_path)
    expected = adata[result.cell_mask, result.gene_mask]
    np.testing.assert_array_equal(filtered.X.toarray(), expected.X.toarray())
    with h5py.File(result.filtered_path) as handle:
        assert handle["X"].attrs["encoding-type"] == b"csr_matrix"


def test_gene_symbol_validation(tmp_path):
    x = np.ones((5, 3))
    obs = pd.DataFrame({"perturbation": ["ctrl"] * 5})
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


def test_pseudobulk_outputs_and_files(tmp_path):
    path, adata = create_test_dataset(tmp_path)
    qc_result = quality_control_summary(
        path,
        min_genes=1,
        min_cells_per_perturbation=2,
        min_cells_per_gene=1,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbol",
        output_dir=tmp_path,
        data_name="pseudobulk",
    )
    avg = compute_average_log_expression(
        qc_result.filtered_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="avg_effects",
    )
    pseudo = compute_pseudobulk_expression(
        qc_result.filtered_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="pseudo_effects",
    )
    assert set(avg.index) == {"KO1", "KO2"}
    assert set(pseudo.index) == {"KO1", "KO2"}
    filtered = ad.read_h5ad(qc_result.filtered_path)
    ctrl_mask = (filtered.obs["perturbation"] == "ctrl").to_numpy()
    ko1_mask = (filtered.obs["perturbation"] == "KO1").to_numpy()
    matrix = filtered.X.toarray() if sp.issparse(filtered.X) else np.asarray(filtered.X)
    normalised = _normalize_total(matrix)
    ctrl = np.log1p(normalised[ctrl_mask, 0])
    ko1 = np.log1p(normalised[ko1_mask, 0])
    expected = ko1.mean() - ctrl.mean()
    assert np.isclose(avg.loc["KO1", "gene0"], expected)
    assert (tmp_path / "avg_effects_avg_log_effects.h5ad").exists()
    assert (tmp_path / "pseudo_effects_pseudobulk_effects.h5ad").exists()


def test_differential_expression_outputs(tmp_path):
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
    wald = wald_test(
        qc_result.filtered_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wald",
    )
    wilcoxon = wilcoxon_test(
        qc_result.filtered_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbols",
        output_dir=tmp_path,
        data_name="wilcoxon",
    )
    assert set(wald.keys()) == {"KO1", "KO2"}
    assert set(wilcoxon.keys()) == {"KO1", "KO2"}
    assert (tmp_path / "wald_wald_de.h5ad").exists()
    assert (tmp_path / "wilcoxon_wilcoxon_de.h5ad").exists()
    ko1_result = wald["KO1"]
    assert ko1_result.effect_size.shape[0] == 4
    assert ko1_result.method == "wald"
    ko2_result = wilcoxon["KO2"]
    assert ko2_result.pvalue.shape[0] == 4
    assert ko2_result.method == "wilcoxon"

