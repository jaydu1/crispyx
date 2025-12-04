"""Utilities for creating the synthetic benchmarking dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def create_demo_dataset(
    *,
    n_cells: int = 400,
    n_genes: int = 100,
    perturbations: int = 5,
    seed: int | None = 0,
) -> ad.AnnData:
    """Return a small AnnData object suitable for benchmarking examples.

    The generated dataset mimics a CRISPR screen with one control group and a
    configurable number of perturbations. Each perturbation targets a small
    subset of genes with modest effect sizes so that the downstream
    differential-expression utilities have meaningful signals to recover.
    """

    if perturbations < 1:
        raise ValueError("perturbations must be at least 1")
    if n_cells < perturbations * 10:
        raise ValueError("n_cells must provide enough observations per perturbation")
    if n_genes < perturbations * 5:
        raise ValueError("n_genes must provide enough targets per perturbation")

    rng = np.random.default_rng(seed)

    guide_labels = ["ctrl"] + [f"perturb_{i+1}" for i in range(perturbations)]
    cell_types = ["T cell", "B cell", "NK cell"]

    perturbation_choices = rng.choice(
        guide_labels,
        size=n_cells,
        p=[0.3] + [0.7 / perturbations] * perturbations,
    )
    celltype_choices = rng.choice(cell_types, size=n_cells)

    base_counts = rng.negative_binomial(20, 0.95, size=(n_cells, n_genes)).astype(np.float32)

    # Inject perturbation-specific effects by boosting a handful of genes per guide.
    genes = np.array([f"G{i:03d}" for i in range(n_genes)])
    for idx, label in enumerate(guide_labels[1:], start=1):
        mask = perturbation_choices == label
        if not np.any(mask):
            continue
        start = (idx * 7) % n_genes
        affected = np.arange(start, start + 5) % n_genes
        boost = rng.poisson(10, size=(mask.sum(), affected.size)).astype(np.float32)
        base_counts[np.ix_(mask, affected)] += boost

    expression = sp.csr_matrix(base_counts)

    obs = pd.DataFrame(
        {
            "perturbation": perturbation_choices,
            "celltype": celltype_choices,
        },
        index=pd.Index([f"cell_{i:03d}" for i in range(n_cells)], name="cell"),
    )

    var = pd.DataFrame(
        {
            "gene_symbols": genes,
        },
        index=pd.Index([f"gene_{i:03d}" for i in range(n_genes)], name="gene"),
    )

    adata = ad.AnnData(X=expression, obs=obs, var=var)
    adata.uns["description"] = {
        "generated_by": "benchmarking.generate_demo_dataset",
        "n_cells": n_cells,
        "n_genes": n_genes,
        "perturbations": perturbations,
        "seed": seed,
    }
    return adata


def write_demo_dataset(path: str | Path, **kwargs) -> Path:
    """Generate the demo dataset and write it to ``path``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata = create_demo_dataset(**kwargs)
    adata.write_h5ad(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the synthetic benchmarking dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/demo_benchmark.h5ad"),
        help="Location to write the generated AnnData file",
    )
    parser.add_argument("--cells", type=int, default=400, help="Number of cells to simulate")
    parser.add_argument("--genes", type=int, default=100, help="Number of genes to simulate")
    parser.add_argument(
        "--perturbations",
        type=int,
        default=5,
        help="Number of perturbations (excluding control) to include",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (set to -1 to sample a random seed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = None if args.seed == -1 else args.seed
    path = write_demo_dataset(
        args.output,
        n_cells=args.cells,
        n_genes=args.genes,
        perturbations=args.perturbations,
        seed=seed,
    )
    print(f"Demo dataset written to {path}")


if __name__ == "__main__":
    main()
