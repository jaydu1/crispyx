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

from streamlined_crispr.data import (
    AnnData,
    ensure_gene_symbol_column,
    read_h5ad_ondisk,
    resolve_control_label,
)
from streamlined_crispr.pseudobulk import compute_average_log_expression


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
    caplog.set_level(logging.INFO, logger="streamlined_crispr.data")
    adata = ad.AnnData(np.ones((2, 2)))
    adata.var_names = pd.Index(["g1", "g2"])

    names = ensure_gene_symbol_column(adata, None)

    assert list(names) == ["g1", "g2"]
    assert "using adata.var_names" in caplog.text


def test_resolve_control_label_infers_ctrl(caplog):
    caplog.set_level(logging.INFO, logger="streamlined_crispr.data")

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
    caplog.set_level(logging.INFO, logger="streamlined_crispr.data")
    path = _create_dataset(tmp_path)

    result = compute_average_log_expression(
        path,
        perturbation_column="perturbation",
        control_label=None,
        gene_name_column="gene_symbol",
    )

    assert set(result.index) == {"KO1", "KO2"}
    assert "Inferred control label" in caplog.text
