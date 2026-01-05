# Benchmarking tools package for crispyx
"""Python modules for benchmarking crispyx against other tools."""

# Use lazy imports to avoid circular import issues when running as __main__
def __getattr__(name: str):
    """Lazy import module attributes to avoid import order issues."""
    # Constants module exports
    if name in ("CACHE_VERSION", "DE_METRIC_KEYS", "TOP_K_VALUES", "SHRINKAGE_METADATA",
                "LFCSHRINK_METHODS", "NB_GLM_METHODS", "ALL_DE_METHODS_FOR_HEATMAP",
                "HEATMAP_METHOD_ORDER", "METHOD_DISPLAY_NAMES", "STANDARD_DE_COLUMNS",
                "STATUS_ORDER"):
        from .constants import (
            CACHE_VERSION,
            DE_METRIC_KEYS,
            TOP_K_VALUES,
            SHRINKAGE_METADATA,
            LFCSHRINK_METHODS,
            NB_GLM_METHODS,
            ALL_DE_METHODS_FOR_HEATMAP,
            HEATMAP_METHOD_ORDER,
            METHOD_DISPLAY_NAMES,
            STANDARD_DE_COLUMNS,
            STATUS_ORDER,
        )
        return locals()[name]
    
    # Cache module exports
    if name in ("save_method_result", "load_method_result", "load_cached_results",
                "save_cache_config", "load_cache_config", "invalidate_cache",
                "check_output_exists", "get_expected_output_path", "resolve_result_path",
                "is_scalar_na", "make_json_serializable", "has_valid_result"):
        from .cache import (
            save_method_result,
            load_method_result,
            load_cached_results,
            save_cache_config,
            load_cache_config,
            invalidate_cache,
            check_output_exists,
            get_expected_output_path,
            resolve_result_path,
            is_scalar_na,
            make_json_serializable,
            has_valid_result,
        )
        return locals()[name]
    
    # Formatting module exports
    if name in ("format_method_name", "get_method_package", "format_full_method_name",
                "is_crispyx_method", "get_shrinkage_type", "format_heatmap_method_name",
                "get_performance_emoji", "get_accuracy_emoji", "format_mean_std",
                "frame_to_markdown_table", "standardise_de_dataframe", "is_scalar_notna",
                "get_method_category", "get_method_sort_key", "get_category_sort_key",
                "format_pct", "format_diff"):
        from .formatting import (
            format_method_name,
            get_method_package,
            format_full_method_name,
            is_crispyx_method,
            get_shrinkage_type,
            format_heatmap_method_name,
            get_performance_emoji,
            get_accuracy_emoji,
            format_mean_std,
            frame_to_markdown_table,
            standardise_de_dataframe,
            is_scalar_notna,
            get_method_category,
            get_method_sort_key,
            get_category_sort_key,
            format_pct,
            format_diff,
        )
        return locals()[name]
    
    # Comparison module exports
    if name in ("compute_de_comparison_metrics", "compute_pairwise_overlap_matrix",
                "compute_pairwise_overlap_matrices_batch"):
        from .comparison import (
            compute_de_comparison_metrics,
            compute_pairwise_overlap_matrix,
            compute_pairwise_overlap_matrices_batch,
        )
        return locals()[name]
    
    # Generate results exports
    if name == "evaluate_benchmarks":
        from .generate_results import evaluate_benchmarks
        return evaluate_benchmarks
    
    # Run benchmarks exports
    if name == "run_benchmarks_main":
        from .run_benchmarks import main as run_benchmarks_main
        return run_benchmarks_main
    
    # Visualization exports
    if name in ("plot_overlap_heatmap", "generate_overlap_heatmaps"):
        from .visualization import (
            plot_overlap_heatmap, 
            generate_overlap_heatmaps,
        )
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Constants
    "CACHE_VERSION",
    "DE_METRIC_KEYS",
    "TOP_K_VALUES",
    "SHRINKAGE_METADATA",
    "LFCSHRINK_METHODS",
    "NB_GLM_METHODS",
    "ALL_DE_METHODS_FOR_HEATMAP",
    "HEATMAP_METHOD_ORDER",
    "METHOD_DISPLAY_NAMES",
    "STANDARD_DE_COLUMNS",
    "STATUS_ORDER",
    # Cache functions
    "save_method_result",
    "load_method_result",
    "load_cached_results",
    "save_cache_config",
    "load_cache_config",
    "invalidate_cache",
    "check_output_exists",
    "get_expected_output_path",
    "resolve_result_path",
    "is_scalar_na",
    "make_json_serializable",
    "has_valid_result",
    # Formatting functions
    "format_method_name",
    "get_method_package",
    "format_full_method_name",
    "is_crispyx_method",
    "get_shrinkage_type",
    "format_heatmap_method_name",
    "get_performance_emoji",
    "get_accuracy_emoji",
    "format_mean_std",
    "frame_to_markdown_table",
    "standardise_de_dataframe",
    "is_scalar_notna",
    "get_method_category",
    "get_method_sort_key",
    "get_category_sort_key",
    "format_pct",
    "format_diff",
    # Comparison functions
    "compute_de_comparison_metrics",
    "compute_pairwise_overlap_matrix",
    "compute_pairwise_overlap_matrices_batch",
    # Evaluation
    "evaluate_benchmarks",
    # Main entry point
    "run_benchmarks_main",
    # Visualization
    "plot_overlap_heatmap",
    "generate_overlap_heatmaps",
]
