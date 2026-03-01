import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crispyx.data import (
    AnnData,
    calculate_nb_glm_chunk_size,
    ensure_gene_symbol_column,
    read_h5ad_ondisk,
    resolve_control_label,
)
from crispyx.pseudobulk import compute_average_log_expression


def _create_dataset(tmp_path: Path) -> Path:
    x = np.array(
        [
            [0, 0, 0],
            [1, 2, 0],
            [0, 0, 1],
            [3, 0, 4],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {"perturbation": ["ctrl", "ctrl", "KO1", "KO2"]},
        index=[f"cell_{idx}" for idx in range(x.shape[0])],
    )
    var = pd.DataFrame({"gene_symbol": [f"gene{idx}" for idx in range(x.shape[1])]})
    var.index = var["gene_symbol"]
    adata = ad.AnnData(x, obs=obs, var=var)
    path = tmp_path / "test.h5ad"
    adata.write(path)
    return path


def test_ensure_gene_symbol_column_uses_var_names(caplog):
    caplog.set_level(logging.INFO, logger="crispyx.data")
    adata = ad.AnnData(np.ones((2, 2)))
    adata.var_names = pd.Index(["g1", "g2"])

    names = ensure_gene_symbol_column(adata, None)

    assert list(names) == ["g1", "g2"]
    assert "using adata.var_names" in caplog.text


def test_resolve_control_label_infers_ctrl(caplog):
    caplog.set_level(logging.INFO, logger="crispyx.data")

    inferred = resolve_control_label(["KO", "CTRL_cells"], None)

    assert inferred == "CTRL_cells"
    assert "Inferred control label" in caplog.text


def test_read_h5ad_ondisk_returns_backed_object(tmp_path, capsys):
    path = _create_dataset(tmp_path)

    adata_ro = read_h5ad_ondisk(path, n_obs=1, n_vars=1)
    captured = capsys.readouterr()

    assert "AnnData object" in captured.out
    assert "First obs rows:" in captured.out
    assert isinstance(adata_ro, AnnData)
    assert adata_ro.backed.isbacked
    adata_ro.close()


def test_compute_average_log_expression_infers_control(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="crispyx.data")
    path = _create_dataset(tmp_path)

    result = compute_average_log_expression(
        path,
        perturbation_column="perturbation",
        control_label=None,
        gene_name_column="gene_symbol",
    )

    assert isinstance(result, AnnData)
    assert set(result.obs.index) == {"KO1", "KO2"}
    loaded_var = result.var.load()
    assert list(loaded_var.index) == ["gene0", "gene1", "gene2"]
    assert "Inferred control label" in caplog.text
    result.close()


# ============================================================================
# Tests for calculate_nb_glm_chunk_size
# ============================================================================


def test_calculate_nb_glm_chunk_size_returns_max_for_small_dataset():
    """Small datasets should use max chunk size (256)."""
    chunk_size = calculate_nb_glm_chunk_size(
        n_obs=10000,
        n_vars=5000,
        n_groups=50,
        available_memory_gb=128,
    )
    assert chunk_size == 256  # max_chunk default


def test_calculate_nb_glm_chunk_size_reduces_for_large_dataset():
    """Large datasets should get reduced chunk size to fit memory."""
    chunk_size = calculate_nb_glm_chunk_size(
        n_obs=1200000,  # 1.2M cells (like Feng-ts)
        n_vars=36000,
        n_groups=500,
        available_memory_gb=128,
    )
    # Should be less than max_chunk due to memory constraints
    assert chunk_size < 256
    assert chunk_size >= 32  # min_chunk default


def test_calculate_nb_glm_chunk_size_respects_memory_limit():
    """memory_limit_gb should cap the available memory."""
    # With high available memory, should use max
    chunk_high = calculate_nb_glm_chunk_size(
        n_obs=500000,
        n_vars=20000,
        n_groups=200,
        available_memory_gb=256,
    )
    
    # With memory_limit_gb, should be constrained
    chunk_limited = calculate_nb_glm_chunk_size(
        n_obs=500000,
        n_vars=20000,
        n_groups=200,
        available_memory_gb=256,
        memory_limit_gb=32,  # Lower limit
    )
    
    assert chunk_limited <= chunk_high


def test_calculate_nb_glm_chunk_size_respects_min_max_bounds():
    """Chunk size should be clamped to [min_chunk, max_chunk]."""
    # Even with huge memory, don't exceed max_chunk
    chunk_max = calculate_nb_glm_chunk_size(
        n_obs=1000,
        n_vars=100,
        n_groups=10,
        available_memory_gb=1000,  # Huge memory
        max_chunk=128,
    )
    assert chunk_max == 128
    
    # Even with tiny memory, don't go below min_chunk
    chunk_min = calculate_nb_glm_chunk_size(
        n_obs=10000000,  # Very large
        n_vars=50000,
        n_groups=1000,
        available_memory_gb=1,  # Tiny memory
        min_chunk=64,
    )
    assert chunk_min == 64


def test_calculate_nb_glm_chunk_size_handles_none_n_groups():
    """Function should work without n_groups specified."""
    chunk_size = calculate_nb_glm_chunk_size(
        n_obs=100000,
        n_vars=20000,
        n_groups=None,  # Unknown groups
        available_memory_gb=64,
    )
    assert 32 <= chunk_size <= 256
