"""Tests covering benchmark helpers that summarise experiment outputs."""

from __future__ import annotations

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

from benchmarking.run_benchmarks import _summarise_dataframe
from streamlined_crispr import compute_average_log_expression


def _create_demo_dataset(tmp_path: Path) -> Path:
    matrix = np.array(
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
        index=[f"cell_{idx}" for idx in range(matrix.shape[0])],
    )
    var = pd.DataFrame({"gene_symbol": [f"gene{idx}" for idx in range(matrix.shape[1])]})
    var.index = var["gene_symbol"]
    adata = ad.AnnData(matrix, obs=obs, var=var)
    dataset_path = tmp_path / "benchmark_demo.h5ad"
    adata.write(dataset_path)
    return dataset_path


def test_summarise_dataframe_includes_result_path(tmp_path: Path) -> None:
    dataset_path = _create_demo_dataset(tmp_path)
    result = compute_average_log_expression(
        dataset_path,
        perturbation_column="perturbation",
        control_label="ctrl",
        gene_name_column="gene_symbol",
        output_dir=tmp_path,
        data_name="summary_test",
    )

    context = {"output_dir": tmp_path}
    summary = _summarise_dataframe(result, context)

    assert summary["rows"] == 2
    assert summary["columns"] == 4
    assert summary["result_path"] == "summary_test_avg_log_effects.h5ad"

    result.close()
