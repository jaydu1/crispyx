"""Tests for QC strategy parity - all strategies should produce identical results."""

from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


# Test datasets with different storage formats
TEST_DATASETS = {
    "csr_small": {
        "path": Path("data/Adamson_subset.h5ad"),
        "perturbation_column": "perturbation",
        "expected_format": "csr",
    },
    "csc_medium": {
        "path": Path("data/Tian-crispra.h5ad"),
        "perturbation_column": "perturbation",
        "expected_format": "csc",
    },
}

# QC parameters used for testing
QC_PARAMS = {
    "min_genes": 100,
    "min_cells_per_perturbation": 50,
    "min_cells_per_gene": 100,
}


@pytest.fixture
def tmp_output_dir():
    """Create a temporary directory for test outputs."""
    tmp_dir = tempfile.mkdtemp(prefix="crispyx_qc_test_")
    yield Path(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_get_matrix_storage_format():
    """Test storage format detection function."""
    from crispyx.data import get_matrix_storage_format
    
    for name, config in TEST_DATASETS.items():
        if not config["path"].exists():
            pytest.skip(f"Dataset {config['path']} not found")
        
        detected_format = get_matrix_storage_format(config["path"])
        assert detected_format == config["expected_format"], (
            f"{name}: expected {config['expected_format']}, got {detected_format}"
        )


def test_qc_in_memory_basic(tmp_output_dir):
    """Test that in-memory QC runs without errors on small dataset."""
    from crispyx.qc import _qc_in_memory
    from crispyx.data import read_backed, resolve_control_label
    
    dataset = TEST_DATASETS["csr_small"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    output_path = tmp_output_dir / "in_memory.h5ad"
    result = _qc_in_memory(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        control_label=control_label,
        gene_name_column=None,
        output_path=output_path,
        **QC_PARAMS,
    )
    
    assert result.cell_mask.sum() > 0, "No cells passed filter"
    assert result.gene_mask.sum() > 0, "No genes passed filter"
    assert output_path.exists(), "Output file not created"
    
    # Verify output file is readable
    import anndata as ad
    adata = ad.read_h5ad(output_path)
    assert adata.n_obs == result.cell_mask.sum()
    assert adata.n_vars == result.gene_mask.sum()


def test_qc_column_oriented_basic(tmp_output_dir):
    """Test that column-oriented QC runs without errors on CSC dataset."""
    from crispyx.qc import _qc_column_oriented
    from crispyx.data import read_backed, resolve_control_label
    
    dataset = TEST_DATASETS["csc_medium"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    output_path = tmp_output_dir / "column_oriented.h5ad"
    result = _qc_column_oriented(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        control_label=control_label,
        gene_name_column=None,
        chunk_size=1024,
        output_path=output_path,
        **QC_PARAMS,
    )
    
    assert result.cell_mask.sum() > 0, "No cells passed filter"
    assert result.gene_mask.sum() > 0, "No genes passed filter"
    assert output_path.exists(), "Output file not created"


def test_qc_row_oriented_basic(tmp_output_dir):
    """Test that row-oriented QC runs without errors on CSR dataset."""
    from crispyx.qc import _qc_row_oriented
    from crispyx.data import read_backed, resolve_control_label
    
    dataset = TEST_DATASETS["csr_small"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    output_path = tmp_output_dir / "row_oriented.h5ad"
    result = _qc_row_oriented(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        control_label=control_label,
        gene_name_column=None,
        chunk_size=1024,
        output_path=output_path,
        cache_mode="memmap",
        delta_threshold=0.3,
        **QC_PARAMS,
    )
    
    assert result.cell_mask.sum() > 0, "No cells passed filter"
    assert result.gene_mask.sum() > 0, "No genes passed filter"
    assert output_path.exists(), "Output file not created"


def test_qc_strategy_parity_csr(tmp_output_dir):
    """Verify all QC strategies produce identical results on CSR dataset."""
    from crispyx.qc import _qc_in_memory, _qc_column_oriented, _qc_row_oriented
    from crispyx.data import read_backed, resolve_control_label
    
    dataset = TEST_DATASETS["csr_small"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    common_kwargs = {
        "perturbation_column": dataset["perturbation_column"],
        "control_label": control_label,
        "gene_name_column": None,
        **QC_PARAMS,
    }
    
    # Run all three strategies
    result_memory = _qc_in_memory(
        dataset["path"],
        output_path=tmp_output_dir / "memory.h5ad",
        **common_kwargs,
    )
    
    result_column = _qc_column_oriented(
        dataset["path"],
        output_path=tmp_output_dir / "column.h5ad",
        chunk_size=1024,
        **common_kwargs,
    )
    
    result_row = _qc_row_oriented(
        dataset["path"],
        output_path=tmp_output_dir / "row.h5ad",
        chunk_size=1024,
        cache_mode="memmap",
        delta_threshold=0.3,
        **common_kwargs,
    )
    
    # Verify cell masks are identical
    assert np.array_equal(result_memory.cell_mask, result_column.cell_mask), (
        f"Cell mask mismatch (in-memory vs column): "
        f"{result_memory.cell_mask.sum()} vs {result_column.cell_mask.sum()}"
    )
    assert np.array_equal(result_memory.cell_mask, result_row.cell_mask), (
        f"Cell mask mismatch (in-memory vs row): "
        f"{result_memory.cell_mask.sum()} vs {result_row.cell_mask.sum()}"
    )
    
    # Verify gene masks are identical
    assert np.array_equal(result_memory.gene_mask, result_column.gene_mask), (
        f"Gene mask mismatch (in-memory vs column): "
        f"{result_memory.gene_mask.sum()} vs {result_column.gene_mask.sum()}"
    )
    assert np.array_equal(result_memory.gene_mask, result_row.gene_mask), (
        f"Gene mask mismatch (in-memory vs row): "
        f"{result_memory.gene_mask.sum()} vs {result_row.gene_mask.sum()}"
    )
    
    print(f"✓ CSR parity: cells={result_memory.cell_mask.sum()}, genes={result_memory.gene_mask.sum()}")


def test_qc_strategy_parity_csc(tmp_output_dir):
    """Verify all QC strategies produce identical results on CSC dataset."""
    from crispyx.qc import _qc_in_memory, _qc_column_oriented, _qc_row_oriented
    from crispyx.data import read_backed, resolve_control_label
    
    dataset = TEST_DATASETS["csc_medium"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    common_kwargs = {
        "perturbation_column": dataset["perturbation_column"],
        "control_label": control_label,
        "gene_name_column": None,
        **QC_PARAMS,
    }
    
    # Run all three strategies
    result_memory = _qc_in_memory(
        dataset["path"],
        output_path=tmp_output_dir / "memory.h5ad",
        **common_kwargs,
    )
    
    result_column = _qc_column_oriented(
        dataset["path"],
        output_path=tmp_output_dir / "column.h5ad",
        chunk_size=1024,
        **common_kwargs,
    )
    
    result_row = _qc_row_oriented(
        dataset["path"],
        output_path=tmp_output_dir / "row.h5ad",
        chunk_size=1024,
        cache_mode="memmap",
        delta_threshold=0.3,
        **common_kwargs,
    )
    
    # Verify cell masks are identical
    assert np.array_equal(result_memory.cell_mask, result_column.cell_mask), (
        f"Cell mask mismatch (in-memory vs column): "
        f"{result_memory.cell_mask.sum()} vs {result_column.cell_mask.sum()}"
    )
    assert np.array_equal(result_memory.cell_mask, result_row.cell_mask), (
        f"Cell mask mismatch (in-memory vs row): "
        f"{result_memory.cell_mask.sum()} vs {result_row.cell_mask.sum()}"
    )
    
    # Verify gene masks are identical
    assert np.array_equal(result_memory.gene_mask, result_column.gene_mask), (
        f"Gene mask mismatch (in-memory vs column): "
        f"{result_memory.gene_mask.sum()} vs {result_column.gene_mask.sum()}"
    )
    assert np.array_equal(result_memory.gene_mask, result_row.gene_mask), (
        f"Gene mask mismatch (in-memory vs row): "
        f"{result_memory.gene_mask.sum()} vs {result_row.gene_mask.sum()}"
    )
    
    print(f"✓ CSC parity: cells={result_memory.cell_mask.sum()}, genes={result_memory.gene_mask.sum()}")


def test_quality_control_summary_dispatch(tmp_output_dir):
    """Test that quality_control_summary correctly dispatches based on data size."""
    from crispyx.qc import quality_control_summary
    
    dataset = TEST_DATASETS["csr_small"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    # Test with force_streaming=False (should use in-memory for small data)
    result1 = quality_control_summary(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        output_dir=tmp_output_dir,
        data_name="test1",
        force_streaming=False,
        **QC_PARAMS,
    )
    
    # Test with force_streaming=True (should use streaming)
    result2 = quality_control_summary(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        output_dir=tmp_output_dir,
        data_name="test2",
        force_streaming=True,
        **QC_PARAMS,
    )
    
    # Results should be identical
    assert np.array_equal(result1.cell_mask, result2.cell_mask), (
        f"Cell mask mismatch between dispatch modes: "
        f"{result1.cell_mask.sum()} vs {result2.cell_mask.sum()}"
    )
    assert np.array_equal(result1.gene_mask, result2.gene_mask), (
        f"Gene mask mismatch between dispatch modes: "
        f"{result1.gene_mask.sum()} vs {result2.gene_mask.sum()}"
    )
    
    print(f"✓ Dispatch parity verified")


def test_qc_against_scanpy(tmp_output_dir):
    """Compare crispyx QC results against Scanpy QC as ground truth."""
    import anndata as ad
    import scanpy as sc
    import scipy.sparse as sp
    
    dataset = TEST_DATASETS["csr_small"]
    if not dataset["path"].exists():
        pytest.skip(f"Dataset {dataset['path']} not found")
    
    from crispyx.qc import quality_control_summary
    from crispyx.data import resolve_control_label, read_backed
    
    # Get control label
    backed = read_backed(dataset["path"])
    labels = backed.obs[dataset["perturbation_column"]].astype(str).to_numpy()
    control_label = resolve_control_label(labels, None, verbose=False)
    backed.file.close()
    
    # Run crispyx QC
    crispyx_result = quality_control_summary(
        dataset["path"],
        perturbation_column=dataset["perturbation_column"],
        output_dir=tmp_output_dir,
        data_name="crispyx",
        **QC_PARAMS,
    )
    
    # Run Scanpy QC
    adata = ad.read_h5ad(dataset["path"])
    if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()
    
    # Filter cells
    sc.pp.filter_cells(adata, min_genes=QC_PARAMS["min_genes"])
    
    # Filter perturbations
    labels = adata.obs[dataset["perturbation_column"]].astype(str)
    counts = labels.value_counts()
    keep = labels.eq(control_label) | counts.loc[labels].ge(QC_PARAMS["min_cells_per_perturbation"]).to_numpy()
    adata = adata[keep].copy()
    
    # Filter genes
    sc.pp.filter_genes(adata, min_cells=QC_PARAMS["min_cells_per_gene"])
    
    # Compare results
    assert crispyx_result.cell_mask.sum() == adata.n_obs, (
        f"Cell count mismatch: crispyx={crispyx_result.cell_mask.sum()}, scanpy={adata.n_obs}"
    )
    assert crispyx_result.gene_mask.sum() == adata.n_vars, (
        f"Gene count mismatch: crispyx={crispyx_result.gene_mask.sum()}, scanpy={adata.n_vars}"
    )
    
    print(f"✓ Scanpy parity: cells={adata.n_obs}, genes={adata.n_vars}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
