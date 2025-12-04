# Benchmarking tools package for CRISPYx
"""Python modules for benchmarking CRISPYx against other tools."""

from .comparison import DE_METRIC_KEYS, compute_de_comparison_metrics
from .run_benchmarks import evaluate_benchmarks, main as run_benchmarks_main

__all__ = [
    "DE_METRIC_KEYS",
    "compute_de_comparison_metrics",
    "evaluate_benchmarks",
    "run_benchmarks_main",
]
