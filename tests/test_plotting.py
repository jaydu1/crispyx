from __future__ import annotations

from pathlib import Path
import sys

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
import anndata as ad

from crispyx import t_test
from crispyx.plotting import (
    materialize_rank_genes_groups,
    plot_ma,
    plot_qc_perturbation_counts,
    plot_qc_summary,
    plot_top_genes_bar,
    plot_volcano,
    rank_genes_groups_df,
)
from crispyx.qc import quality_control_summary


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("plotting")
    rng = np.random.default_rng(0)
    n_cells = 30
    n_genes = 25
    X = rng.poisson(1.0, size=(n_cells, n_genes)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "perturbation": np.repeat(["control", "pert1", "pert2"], n_cells // 3),
        }
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = ad.AnnData(sp.csr_matrix(X), obs=obs, var=var)
    data_path = tmp_path / "toy.h5ad"
    adata.write(data_path)
    return data_path


def test_materialize_rank_genes_groups(small_dataset, tmp_path):
    result = t_test(
        small_dataset,
        perturbation_column="perturbation",
        control_label="control",
        cell_chunk_size=10,
        n_jobs=1,
        output_dir=tmp_path,
        data_name="toy",
    )
    adata_plot = materialize_rank_genes_groups(result.result_path, n_genes=10)
    rgg = adata_plot.uns["rank_genes_groups"]
    assert "names" in rgg
    assert rgg["names"].shape[0] == 10
    assert set(rgg["names"].dtype.names or []) == {"pert1", "pert2"}


def test_de_plotting_functions(small_dataset, tmp_path):
    pytest.importorskip("matplotlib")
    result = t_test(
        small_dataset,
        perturbation_column="perturbation",
        control_label="control",
        cell_chunk_size=10,
        n_jobs=1,
        output_dir=tmp_path,
        data_name="toy",
    )
    df = rank_genes_groups_df(result.result_path, group="pert1", n_genes=50)
    ax = plot_volcano(de_df=df, group="pert1", show=False)
    assert ax is not None
    ax = plot_top_genes_bar(de_df=df, group="pert1", topn=10, show=False)
    assert ax is not None
    ax = plot_ma(
        data=small_dataset,
        de_result=result.result_path,
        group="pert1",
        reference="control",
        perturbation_column="perturbation",
        mean_mode="raw",
        n_genes=50,
        show=False,
    )
    assert ax is not None


def test_qc_plotting_functions(small_dataset, tmp_path):
    pytest.importorskip("matplotlib")
    qc = quality_control_summary(
        small_dataset,
        perturbation_column="perturbation",
        control_label="control",
        min_genes=1,
        min_cells_per_perturbation=2,
        min_cells_per_gene=1,
        output_dir=tmp_path,
        data_name="toy",
    )
    ax = plot_qc_perturbation_counts(
        data=small_dataset,
        perturbation_column="perturbation",
        cell_mask=qc.cell_mask,
        show=False,
    )
    assert ax is not None
    axes = plot_qc_summary(qc, min_genes=1, min_cells_per_gene=1, show=False)
    assert axes is not None
