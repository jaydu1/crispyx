#!/usr/bin/env python3
"""Benchmark the actual nb_glm_test implementation to measure the frozen control optimization.

Usage:
    python -m benchmarking.tools.benchmark_nb_glm_frozen --dataset Replogle-GW-k562 --n-perturbations 2
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_dataset_path(dataset_name: str) -> Path:
    """Resolve dataset name to path."""
    base_dir = Path(__file__).parent.parent.parent
    
    candidates = [
        base_dir / "data" / f"{dataset_name}.h5ad",
        base_dir / ".cache" / f"{dataset_name}.h5ad",
        base_dir / "benchmarking" / ".cache" / f"{dataset_name}.h5ad",
        base_dir / "benchmarking" / "results" / dataset_name / ".cache" / f"standardized_{dataset_name}.h5ad",
        base_dir / "benchmarking" / "results" / dataset_name / ".cache" / f"{dataset_name}.h5ad",
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    if Path(dataset_name).exists():
        return Path(dataset_name)
    
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found. Checked:\n" +
        "\n".join(f"  - {p}" for p in candidates)
    )


def benchmark_frozen_control(
    dataset_path: Path,
    n_perturbations: int = 2,
    perturbation_column: str = "perturbation",
    control_label: str | None = None,
):
    """Benchmark the frozen control optimization."""
    import tempfile
    import anndata as ad
    from crispyx.de import nb_glm_test
    
    logger.info(f"Loading dataset: {dataset_path}")
    adata = ad.read_h5ad(dataset_path)
    
    n_cells, n_genes = adata.shape
    labels = adata.obs[perturbation_column].values
    unique_labels = np.unique(labels)
    
    # Infer control label
    if control_label is None:
        for candidate in ["control", "Control", "CONTROL", "ctrl", "NT", "non-targeting"]:
            if candidate in unique_labels:
                control_label = candidate
                logger.info(f"Inferred control label '{control_label}'")
                break
    
    if control_label is None:
        raise ValueError("Could not infer control label")
    
    # Get perturbation-only labels
    pert_labels = [l for l in unique_labels if l != control_label]
    pert_labels = sorted(pert_labels)[:n_perturbations]
    
    n_control = int((labels == control_label).sum())
    logger.info(f"Dataset: {n_cells:,} cells, {n_genes:,} genes")
    logger.info(f"Control cells: {n_control:,}")
    logger.info(f"Testing {n_perturbations} perturbations: {pert_labels}")
    
    # Subset to control + selected perturbations
    mask = (labels == control_label) | np.isin(labels, pert_labels)
    adata_subset = adata[mask].copy()
    
    logger.info(f"Subset: {adata_subset.n_obs:,} cells")
    
    # Save subset to temporary file (nb_glm_test needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as f:
        temp_path = Path(f.name)
    
    logger.info(f"Saving subset to temporary file: {temp_path}")
    adata_subset.write_h5ad(temp_path)
    
    try:
        # Run nb_glm_test
        logger.info("\nRunning nb_glm_test...")
        t0 = time.perf_counter()
        
        result = nb_glm_test(
            temp_path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            n_jobs=1,  # Single job to measure per-perturbation time
        )
        
        total_time = time.perf_counter() - t0
        time_per_pert = total_time / n_perturbations
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Time per perturbation: {time_per_pert:.2f}s")
        logger.info(f"Perturbations tested: {n_perturbations}")
        logger.info(f"Control cells: {n_control:,}")
        logger.info(f"{'='*60}")
        
        return result, total_time, time_per_pert
    
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Benchmark frozen control optimization")
    parser.add_argument("--dataset", required=True, help="Dataset name or path")
    parser.add_argument("--n-perturbations", type=int, default=2, help="Number of perturbations to test")
    parser.add_argument("--perturbation-column", default="perturbation", help="Column name for perturbation labels")
    parser.add_argument("--control-label", default=None, help="Control label (auto-detected if not specified)")
    args = parser.parse_args()
    
    dataset_path = get_dataset_path(args.dataset)
    logger.info(f"Using dataset: {dataset_path}")
    
    benchmark_frozen_control(
        dataset_path,
        n_perturbations=args.n_perturbations,
        perturbation_column=args.perturbation_column,
        control_label=args.control_label,
    )


if __name__ == "__main__":
    main()
