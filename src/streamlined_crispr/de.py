"""Differential expression tests implemented with pure Python primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

from .data import read_h5ad


@dataclass
class DifferentialExpressionResult:
    genes: List[str]
    effect_size: List[float]
    statistic: List[float]
    pvalue: List[float]
    method: str
    perturbation: str


def _candidate_labels(labels: List[str], control_label: str, perturbation: Optional[str]) -> List[str]:
    if perturbation is not None:
        if perturbation == control_label:
            raise ValueError("Perturbation must not match control label")
        return [perturbation]
    seen = []
    for label in labels:
        if label == control_label or label in seen:
            continue
        seen.append(label)
    return seen


def wald_test(
    path: str,
    *,
    column: str = "perturbation",
    control_label: str = "ctrl",
    perturbation: Optional[str] = None,
    min_cells_expressed: int = 0,
) -> Dict[str, DifferentialExpressionResult]:
    adata = read_h5ad(path)
    labels = [cell[column] for cell in adata.obs]
    candidates = _candidate_labels(labels, control_label, perturbation)
    results: Dict[str, DifferentialExpressionResult] = {}

    for target in candidates:
        control = {"sum": [0.0] * adata.n_vars, "sumsq": [0.0] * adata.n_vars}
        pert = {"sum": [0.0] * adata.n_vars, "sumsq": [0.0] * adata.n_vars}
        expr_control = [0] * adata.n_vars
        expr_pert = [0] * adata.n_vars
        n_control = 0
        n_pert = 0

        for idx, row in enumerate(adata.iter_rows()):
            label = labels[idx]
            log_row = [math.log1p(value) for value in row]
            if label == control_label:
                control["sum"] = [a + b for a, b in zip(control["sum"], log_row)]
                control["sumsq"] = [a + b * b for a, b in zip(control["sumsq"], log_row)]
                expr_control = [e + (1 if value > 0 else 0) for e, value in zip(expr_control, row)]
                n_control += 1
            elif label == target:
                pert["sum"] = [a + b for a, b in zip(pert["sum"], log_row)]
                pert["sumsq"] = [a + b * b for a, b in zip(pert["sumsq"], log_row)]
                expr_pert = [e + (1 if value > 0 else 0) for e, value in zip(expr_pert, row)]
                n_pert += 1

        if n_control == 0 or n_pert == 0:
            raise ValueError(f"Insufficient cells for perturbation '{target}'")

        effect = []
        statistic = []
        pvalue = []
        for gene_idx in range(adata.n_vars):
            total_expr = expr_control[gene_idx] + expr_pert[gene_idx]
            if total_expr < min_cells_expressed:
                effect.append(0.0)
                statistic.append(0.0)
                pvalue.append(1.0)
                continue
            mean_control = control["sum"][gene_idx] / n_control
            mean_pert = pert["sum"][gene_idx] / n_pert
            diff = mean_pert - mean_control

            var_control = 0.0
            if n_control > 1:
                var_control = (
                    control["sumsq"][gene_idx]
                    - (control["sum"][gene_idx] ** 2) / n_control
                ) / (n_control - 1)
            var_pert = 0.0
            if n_pert > 1:
                var_pert = (
                    pert["sumsq"][gene_idx]
                    - (pert["sum"][gene_idx] ** 2) / n_pert
                ) / (n_pert - 1)
            pooled_denom = n_control + n_pert - 2
            pooled = 0.0 if pooled_denom <= 0 else (
                (n_control - 1) * var_control + (n_pert - 1) * var_pert
            ) / pooled_denom
            se = math.sqrt(pooled * (1.0 / n_control + 1.0 / n_pert)) if pooled > 0 else 0.0
            if se == 0.0:
                effect.append(0.0)
                statistic.append(0.0)
                pvalue.append(1.0)
                continue
            z = diff / se
            effect.append(diff)
            statistic.append(z)
            pvalue.append(math.erfc(abs(z) / math.sqrt(2.0)))

        results[target] = DifferentialExpressionResult(
            genes=list(adata.var),
            effect_size=effect,
            statistic=statistic,
            pvalue=pvalue,
            method="wald",
            perturbation=target,
        )
    return results


def wilcoxon_test(
    path: str,
    *,
    column: str = "perturbation",
    control_label: str = "ctrl",
    perturbation: Optional[str] = None,
    min_cells_expressed: int = 0,
) -> Dict[str, DifferentialExpressionResult]:
    from collections import defaultdict

    adata = read_h5ad(path)
    labels = [cell[column] for cell in adata.obs]
    candidates = _candidate_labels(labels, control_label, perturbation)
    results: Dict[str, DifferentialExpressionResult] = {}

    for target in candidates:
        control_hist = [defaultdict(int) for _ in range(adata.n_vars)]
        pert_hist = [defaultdict(int) for _ in range(adata.n_vars)]
        expr_counts = [0] * adata.n_vars
        n_control = 0
        n_pert = 0

        for idx, row in enumerate(adata.iter_rows()):
            label = labels[idx]
            if label == control_label:
                target_hist = control_hist
                n_control += 1
            elif label == target:
                target_hist = pert_hist
                n_pert += 1
            else:
                continue
            for gene_idx, value in enumerate(row):
                if value > 0:
                    expr_counts[gene_idx] += 1
                target_hist[gene_idx][value] += 1

        if n_control == 0 or n_pert == 0:
            raise ValueError(f"Insufficient cells for perturbation '{target}'")

        effect = [0.0] * adata.n_vars
        statistic = [0.0] * adata.n_vars
        pvalue = [1.0] * adata.n_vars
        total_cells = n_control + n_pert

        for gene_idx in range(adata.n_vars):
            if expr_counts[gene_idx] < min_cells_expressed:
                continue
            control_counts = control_hist[gene_idx]
            pert_counts = pert_hist[gene_idx]
            values = sorted(set(control_counts) | set(pert_counts))
            less_control = 0.0
            u_stat = 0.0
            tie_sum = 0.0
            for value in values:
                c0 = control_counts.get(value, 0)
                c1 = pert_counts.get(value, 0)
                total = c0 + c1
                u_stat += c1 * less_control + 0.5 * c1 * c0
                tie_sum += total * (total * total - 1)
                less_control += c0
            mean_u = n_control * n_pert / 2
            var_u = n_control * n_pert * (total_cells + 1) / 12
            if total_cells > 1:
                var_u -= (n_control * n_pert * tie_sum) / (12 * total_cells * (total_cells - 1))
            if var_u <= 0:
                continue
            z = (u_stat - mean_u) / math.sqrt(var_u)
            statistic[gene_idx] = z
            pvalue[gene_idx] = math.erfc(abs(z) / math.sqrt(2.0))
            effect[gene_idx] = u_stat / (n_control * n_pert) - 0.5

        results[target] = DifferentialExpressionResult(
            genes=list(adata.var),
            effect_size=effect,
            statistic=statistic,
            pvalue=pvalue,
            method="wilcoxon",
            perturbation=target,
        )
    return results
