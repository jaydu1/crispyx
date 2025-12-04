# Benchmarking tools package for CRISPYx
"""Python modules for benchmarking CRISPYx against other tools."""

# Use lazy imports to avoid circular import issues when running as __main__
def __getattr__(name: str):
    """Lazy import module attributes to avoid import order issues."""
    if name in ("DE_METRIC_KEYS", "TOP_K_VALUES", "compute_de_comparison_metrics", 
                "compute_pairwise_overlap_matrix"):
        from .comparison import (
            DE_METRIC_KEYS,
            TOP_K_VALUES,
            compute_de_comparison_metrics,
            compute_pairwise_overlap_matrix,
        )
        return locals()[name]
    
    if name in ("evaluate_benchmarks", "run_benchmarks_main"):
        from .run_benchmarks import evaluate_benchmarks, main as run_benchmarks_main
        if name == "run_benchmarks_main":
            return run_benchmarks_main
        return evaluate_benchmarks
    
    if name in ("plot_overlap_heatmap", "generate_overlap_heatmaps"):
        from .visualization import plot_overlap_heatmap, generate_overlap_heatmaps
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DE_METRIC_KEYS",
    "TOP_K_VALUES",
    "compute_de_comparison_metrics",
    "compute_pairwise_overlap_matrix",
    "evaluate_benchmarks",
    "run_benchmarks_main",
    "plot_overlap_heatmap",
    "generate_overlap_heatmaps",
]
