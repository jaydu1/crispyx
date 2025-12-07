"""Visualization utilities for benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .generate_results import SHRINKAGE_METADATA, LFCSHRINK_METHODS

# Order for methods in overlap heatmaps
# Ordered by: t-test, Wilcoxon, NB-GLM (base), NB-GLM (shrunk)
# Note: Shrunk results are loaded from shrunk_result_path and stored with _shrunk suffix
HEATMAP_METHOD_ORDER = [
    # t-test
    "crispyx_de_t_test",
    "scanpy_de_t_test",
    # Wilcoxon
    "crispyx_de_wilcoxon",
    "scanpy_de_wilcoxon",
    # NB-GLM (no shrinkage) - edgeR first as reference
    "edger_de_glm",
    "crispyx_de_nb_glm",
    "pertpy_de_pydeseq2",
    # NB-GLM (joint, no shrinkage)
    "crispyx_de_nb_glm_joint",
    # NB-GLM (with shrinkage)
    "crispyx_de_nb_glm_shrunk",
    "crispyx_de_nb_glm_joint_shrunk",
    "pertpy_de_pydeseq2_shrunk",
]


def _format_heatmap_method_name(name: str) -> str:
    """Format method name for heatmap display with shrinkage indicator.
    
    Parameters
    ----------
    name : str
        Internal method name (e.g., 'crispyx_de_nb_glm_joint' or 'crispyx_de_nb_glm_shrunk')
        
    Returns
    -------
    str
        Display name with package prefix and (lfcShrink) suffix if uses shrinkage
    """
    # Handle shrunk method names (these have _shrunk suffix for heatmap purposes)
    if name == "pertpy_de_pydeseq2_shrunk":
        return "PyDESeq2 (lfcShrink)"
    elif name == "crispyx_de_nb_glm_joint_shrunk":
        return "crispyx NB-GLM (joint, lfcShrink)"
    elif name == "crispyx_de_nb_glm_shrunk":
        return "crispyx NB-GLM (lfcShrink)"
    # Handle base method names
    elif name == "pertpy_de_pydeseq2":
        return "PyDESeq2"
    elif name == "crispyx_de_nb_glm_joint":
        return "crispyx NB-GLM (joint)"
    elif name == "crispyx_de_nb_glm":
        return "crispyx NB-GLM"
    elif name == "edger_de_glm":
        return "edgeR NB-GLM"
    elif name == "crispyx_de_t_test":
        return "crispyx t-test"
    elif name == "scanpy_de_t_test":
        return "scanpy t-test"
    elif name == "crispyx_de_wilcoxon":
        return "crispyx Wilcoxon"
    elif name == "scanpy_de_wilcoxon":
        return "scanpy Wilcoxon"
    
    # Generic formatting for other methods
    display_name = name.replace("crispyx_", "crispyx ").replace("scanpy_", "scanpy ")
    display_name = display_name.replace("pertpy_", "pertpy ").replace("edger_", "edgeR ")
    display_name = display_name.replace("de_", "").replace("_", " ")
    if uses_shrinkage:
        display_name = f"{display_name}{shrink_suffix}"
    
    return display_name


def _order_heatmap_methods(methods: list[str]) -> list[str]:
    """Order methods for heatmap display.
    
    Parameters
    ----------
    methods : list[str]
        List of method names
        
    Returns
    -------
    list[str]
        Ordered list of method names
    """
    ordered = []
    # First add methods in the predefined order
    for m in HEATMAP_METHOD_ORDER:
        if m in methods:
            ordered.append(m)
    # Then add any remaining methods not in the predefined order
    for m in methods:
        if m not in ordered:
            ordered.append(m)
    return ordered


def plot_overlap_heatmap(
    overlap_matrix: pd.DataFrame,
    *,
    title: str = "Top-k DE Gene Overlap",
    output_path: Optional[Path] = None,
    effective_k: Optional[int] = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "RdYlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
    annot: bool = True,
    fmt: str = ".2f",
) -> Optional[Path]:
    """Create a heatmap visualization of pairwise DE gene overlap.
    
    This function creates a symmetric heatmap showing the overlap ratio
    between top-k DE genes from different methods. Missing comparisons
    (NaN values) are shown in light gray.
    
    Parameters
    ----------
    overlap_matrix : pd.DataFrame
        Square DataFrame with method names as index and columns,
        containing overlap ratios (0.0 to 1.0). NaN for missing pairs.
    title : str
        Title for the heatmap (default: "Top-k DE Gene Overlap")
    output_path : Path, optional
        If provided, save the figure to this path. Otherwise, returns None.
    effective_k : int, optional
        The actual k value used (may be less than requested k if fewer genes).
        If provided, will be included in the title note.
    figsize : tuple[float, float]
        Figure size in inches (default: (10, 8))
    cmap : str
        Colormap name for valid values (default: "RdYlGn")
    vmin : float
        Minimum value for colormap (default: 0.0)
    vmax : float
        Maximum value for colormap (default: 1.0)
    annot : bool
        Whether to show values in cells (default: True)
    fmt : str
        Format string for annotations (default: ".2f")
        
    Returns
    -------
    Path | None
        Path to saved figure if output_path was provided, otherwise None
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for saving
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        print(f"Warning: Could not import visualization libraries: {e}")
        return None
    
    if overlap_matrix.empty:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a mask for NaN values
    mask = overlap_matrix.isna()
    
    # Use the new formatting function
    formatted_labels = [_format_heatmap_method_name(n) for n in overlap_matrix.columns]
    
    # Create heatmap with custom settings
    heatmap = sns.heatmap(
        overlap_matrix,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Overlap Ratio', 'shrink': 0.8},
        ax=ax,
        xticklabels=formatted_labels,
        yticklabels=formatted_labels,
    )
    
    # Fill NaN cells with light gray background
    for i in range(len(overlap_matrix)):
        for j in range(len(overlap_matrix.columns)):
            if mask.iloc[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=True,
                    facecolor='lightgray',
                    edgecolor='white',
                    linewidth=0.5,
                ))
                # Add "N/A" text for missing values
                ax.text(
                    j + 0.5, i + 0.5, "N/A",
                    ha='center', va='center',
                    fontsize=8, color='gray',
                )
    
    # Set title with effective k note if provided
    full_title = title
    if effective_k is not None:
        full_title += f"\n(k capped at {effective_k} genes)"
    ax.set_title(full_title, fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return output_path
    
    plt.close(fig)
    return None


def generate_overlap_heatmaps(
    de_results: dict[str, pd.DataFrame],
    output_dir: Path,
    k_values: tuple[int, ...] = (50, 100, 500),
) -> dict[str, Path]:
    """Generate heatmaps for multiple k values and both effect/pvalue metrics.
    
    Creates 2 × len(k_values) heatmaps:
    - benchmark_effect_top_{k}_overlap.png for each k
    - benchmark_pvalue_top_{k}_overlap.png for each k
    
    Parameters
    ----------
    de_results : Dict[str, pd.DataFrame]
        Dictionary mapping method names to their DE result DataFrames.
        Each DataFrame should have columns: perturbation, gene, effect_size, pvalue
    output_dir : Path
        Directory to save the heatmap images
    k_values : tuple[int, ...]
        Top-k values to generate heatmaps for (default: (50, 100, 500))
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping heatmap names to their file paths
    """
    from .comparison import compute_pairwise_overlap_matrix
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Order methods for consistent heatmap display
    ordered_methods = _order_heatmap_methods(list(de_results.keys()))
    ordered_de_results = {m: de_results[m] for m in ordered_methods if m in de_results}
    
    generated_files = {}
    
    for k in k_values:
        for metric in ("effect", "pvalue"):
            matrix, effective_k = compute_pairwise_overlap_matrix(
                ordered_de_results,
                top_k=k,
                metric=metric,
            )
            
            if matrix.empty:
                continue
            
            # Reorder matrix rows/columns to match method order
            ordered_names = [m for m in ordered_methods if m in matrix.columns]
            matrix = matrix.loc[ordered_names, ordered_names]
            
            metric_label = "Effect Size" if metric == "effect" else "P-value"
            title = f"Top-{k} {metric_label} Gene Overlap"
            
            # Use benchmark_ prefix for output files
            filename = f"benchmark_{metric}_top_{k}_overlap.png"
            output_path = output_dir / filename
            
            result_path = plot_overlap_heatmap(
                matrix,
                title=title,
                output_path=output_path,
                effective_k=effective_k if effective_k < k else None,
            )
            
            if result_path:
                generated_files[filename] = result_path
    
    return generated_files
