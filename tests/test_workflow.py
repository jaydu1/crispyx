import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from streamlined_crispr import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
    quality_control_summary,
    wald_test,
    wilcoxon_test,
    write_h5ad,
)


def create_test_dataset(tmp_path):
    matrix = [
        [0, 0, 0, 0],
        [1, 0, 3, 0],
        [0, 1, 0, 0],
        [2, 1, 0, 4],
        [0, 0, 0, 5],
        [1, 0, 0, 6],
    ]
    obs = [
        {"perturbation": "ctrl"},
        {"perturbation": "ctrl"},
        {"perturbation": "KO1"},
        {"perturbation": "KO1"},
        {"perturbation": "KO2"},
        {"perturbation": "KO2"},
    ]
    var = [f"gene{i}" for i in range(len(matrix[0]))]
    path = tmp_path / "test.h5ad"
    write_h5ad(path, matrix, obs, var)
    return path, matrix


def test_quality_control(tmp_path):
    path, _ = create_test_dataset(tmp_path)
    qc = quality_control_summary(
        path,
        min_genes=1,
        min_cells_per_perturbation=2,
        min_cells_per_gene=1,
        perturbation_column="perturbation",
        control_label="ctrl",
    )
    assert sum(qc.cell_mask) == 5
    assert sum(qc.gene_mask) == 4


def test_pseudobulk_estimators(tmp_path):
    path, matrix = create_test_dataset(tmp_path)
    avg = compute_average_log_expression(path, control_label="ctrl")
    pseudo = compute_pseudobulk_expression(path, control_label="ctrl")
    assert set(avg.keys()) == {"KO1", "KO2"}
    control_cells = matrix[:2]
    ko1_cells = matrix[2:4]
    control_mean = sum(math.log1p(row[0]) for row in control_cells) / 2
    ko1_mean = sum(math.log1p(row[0]) for row in ko1_cells) / 2
    assert abs(avg["KO1"][0] - (ko1_mean - control_mean)) < 1e-9
    control_bulk = math.log1p(sum(row[0] for row in control_cells) / 2)
    ko1_bulk = math.log1p(sum(row[0] for row in ko1_cells) / 2)
    assert abs(pseudo["KO1"][0] - (ko1_bulk - control_bulk)) < 1e-9


def test_differential_expression(tmp_path):
    path, _ = create_test_dataset(tmp_path)
    wald_results = wald_test(path, control_label="ctrl", min_cells_expressed=2)
    ko1 = wald_results["KO1"]
    assert ko1.effect_size[0] > 0
    assert ko1.pvalue[2] == 1.0

    wilcoxon_results = wilcoxon_test(path, control_label="ctrl", min_cells_expressed=2)
    ko1_w = wilcoxon_results["KO1"]
    assert ko1_w.effect_size[0] > -0.5
    assert ko1_w.pvalue[2] == 1.0
