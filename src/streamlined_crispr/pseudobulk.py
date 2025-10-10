"""Pseudo-bulk effect size estimators for the lightweight ``.h5ad`` format."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from .data import read_h5ad


def _candidate_labels(labels: List[str], control_label: str, perturbations: Optional[Iterable[str]]) -> List[str]:
    if perturbations is not None:
        return [label for label in perturbations if label != control_label]
    seen = []
    for label in labels:
        if label == control_label or label in seen:
            continue
        seen.append(label)
    return seen


def compute_average_log_expression(
    path: str,
    *,
    column: str = "perturbation",
    control_label: str = "ctrl",
    perturbations: Optional[Iterable[str]] = None,
) -> Dict[str, List[float]]:
    adata = read_h5ad(path)
    labels = [cell[column] for cell in adata.obs]
    candidates = _candidate_labels(labels, control_label, perturbations)
    group_labels = [control_label] + candidates

    sums = {label: [0.0] * adata.n_vars for label in group_labels}
    counts = {label: 0 for label in group_labels}

    for idx, row in enumerate(adata.iter_rows()):
        label = labels[idx]
        if label not in sums:
            continue
        log_row = [math.log1p(value) for value in row]
        sums[label] = [a + b for a, b in zip(sums[label], log_row)]
        counts[label] += 1

    if counts[control_label] == 0:
        raise ValueError("Control group contains no cells")
    control_mean = [value / counts[control_label] for value in sums[control_label]]

    result: Dict[str, List[float]] = {}
    for label in candidates:
        if counts[label] == 0:
            raise ValueError(f"Perturbation '{label}' contains no cells")
        pert_mean = [value / counts[label] for value in sums[label]]
        result[label] = [p - c for p, c in zip(pert_mean, control_mean)]
    return result


def compute_pseudobulk_expression(
    path: str,
    *,
    column: str = "perturbation",
    control_label: str = "ctrl",
    perturbations: Optional[Iterable[str]] = None,
    baseline_count: float = 1.0,
) -> Dict[str, List[float]]:
    if baseline_count <= 0:
        raise ValueError("baseline_count must be positive")

    adata = read_h5ad(path)
    labels = [cell[column] for cell in adata.obs]
    candidates = _candidate_labels(labels, control_label, perturbations)
    group_labels = [control_label] + candidates

    sums = {label: [0.0] * adata.n_vars for label in group_labels}
    counts = {label: 0 for label in group_labels}

    for idx, row in enumerate(adata.iter_rows()):
        label = labels[idx]
        if label not in sums:
            continue
        sums[label] = [a + b for a, b in zip(sums[label], row)]
        counts[label] += 1

    if counts[control_label] == 0:
        raise ValueError("Control group contains no cells")
    control_bulk = [math.log1p(baseline_count * value / counts[control_label]) for value in sums[control_label]]

    result: Dict[str, List[float]] = {}
    for label in candidates:
        if counts[label] == 0:
            raise ValueError(f"Perturbation '{label}' contains no cells")
        pert_bulk = [math.log1p(baseline_count * value / counts[label]) for value in sums[label]]
        result[label] = [p - c for p, c in zip(pert_bulk, control_bulk)]
    return result
