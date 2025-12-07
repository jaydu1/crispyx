# Benchmarking tools package for crispyx
"""Python modules for benchmarking crispyx against other tools."""

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
    
    if name in ("evaluate_benchmarks", "SHRINKAGE_METADATA", "LFCSHRINK_METHODS", 
                "NB_GLM_METHODS"):
        from .generate_results import (
            evaluate_benchmarks, 
            SHRINKAGE_METADATA, 
            LFCSHRINK_METHODS,
            NB_GLM_METHODS,
        )
        return locals()[name]
    
    if name == "run_benchmarks_main":
        from .run_benchmarks import main as run_benchmarks_main
        return run_benchmarks_main
    
    if name in ("plot_overlap_heatmap", "generate_overlap_heatmaps", 
                "HEATMAP_METHOD_ORDER"):
        from .visualization import (
            plot_overlap_heatmap, 
            generate_overlap_heatmaps,
            HEATMAP_METHOD_ORDER,
        )
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DE_METRIC_KEYS",
    "TOP_K_VALUES",
    "compute_de_comparison_metrics",
    "compute_pairwise_overlap_matrix",
    "evaluate_benchmarks",
    "SHRINKAGE_METADATA",
    "LFCSHRINK_METHODS",
    "NB_GLM_METHODS",
    "run_benchmarks_main",
    "plot_overlap_heatmap",
    "generate_overlap_heatmaps",
    "HEATMAP_METHOD_ORDER",
]
