"""Generate benchmark results reports from cached benchmark data.

This module handles the evaluation and reporting of benchmark results,
separated from the benchmark execution logic in run_benchmarks.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


# ============================================================================
# Helper functions for safe NA checking
# ============================================================================

def _is_scalar_na(value: Any) -> bool:
    """Check if a value is NA/NaN, handling arrays properly.
    
    Returns True only if value is None or a scalar NA.
    Returns False for arrays (they're not NA even if they contain NAs).
    """
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return False  # Arrays are not scalar NA
    if isinstance(value, (list, dict)):
        return False  # Collections are not scalar NA
    try:
        return pd.isna(value)
    except (ValueError, TypeError):
        return False


def _is_scalar_notna(value: Any) -> bool:
    """Check if a value is not NA/NaN, handling arrays properly.
    
    Returns True if value is not None and is not a scalar NA.
    Also returns True for arrays (they're considered "not NA" even if they contain NAs).
    """
    return not _is_scalar_na(value)


# ============================================================================
# Shrinkage Metadata Constants
# ============================================================================

# Mapping of method names to their LFC shrinkage type
# Note: With integrated shrinkage, base methods now produce both base and shrunk outputs.
# The shrunk output paths are stored in 'shrunk_result_path' column.
# This dict is kept for compatibility but shrinkage is now integrated into base methods.
SHRINKAGE_METADATA: Dict[str, str] = {
    # Base methods that produce shrunk outputs via integrated shrinkage
    "crispyx_de_nb_glm": "apeglm",          # uses lfc_shrinkage_type="apeglm" for shrunk output
    "crispyx_de_nb_glm_joint": "apeglm",    # uses lfc_shrinkage_type="apeglm" for shrunk output
    "pertpy_de_pydeseq2": "apeglm",         # uses lfcShrink(type="apeglm") for shrunk output
}

# Set of methods that produce LFC shrinkage outputs (derived from SHRINKAGE_METADATA)
LFCSHRINK_METHODS = set(SHRINKAGE_METADATA.keys())

# All NB-GLM methods for explicit heatmap collection (base methods)
NB_GLM_METHODS = [
    "crispyx_de_nb_glm",
    "crispyx_de_nb_glm_joint",
    "edger_de_glm",
    "pertpy_de_pydeseq2",
]

# All DE methods for heatmap collection (includes t-test, Wilcoxon, and NB-GLM)
ALL_DE_METHODS_FOR_HEATMAP = [
    # t-test
    "crispyx_de_t_test",
    "scanpy_de_t_test",
    # Wilcoxon
    "crispyx_de_wilcoxon",
    "scanpy_de_wilcoxon",
    # NB-GLM
    "crispyx_de_nb_glm",
    "crispyx_de_nb_glm_joint",
    "edger_de_glm",
    "pertpy_de_pydeseq2",
]

# Methods that produce shrunk outputs (for heatmap collection)
# These will be loaded from shrunk_result_path and added with _shrunk suffix
METHODS_WITH_SHRUNK_OUTPUT = [
    "crispyx_de_nb_glm",
    "crispyx_de_nb_glm_joint",
    "pertpy_de_pydeseq2",
]

# Standard DE result columns
_STANDARD_DE_COLUMNS = ["perturbation", "gene", "effect_size", "statistic", "pvalue"]

# Order for DE method display in tables: t-test, Wilcoxon, NB-GLM, NB-GLM (joint)
DE_METHOD_DISPLAY_ORDER = [
    "t-test",
    "Wilcoxon", 
    "NB-GLM",
    "NB-GLM (joint)",
]

# Order for category subsections in Performance and Accuracy
CATEGORY_DISPLAY_ORDER = [
    "Preprocessing / QC",
    "DE: t-test",
    "DE: Wilcoxon",
    "DE: NB GLM",
    "Other",
]


def _get_method_sort_key(method_name: str) -> int:
    """Get sort key for a method name based on DE_METHOD_DISPLAY_ORDER."""
    for i, order_name in enumerate(DE_METHOD_DISPLAY_ORDER):
        if order_name in method_name:
            return i
    return len(DE_METHOD_DISPLAY_ORDER)  # Unknown methods go last


def _get_category_sort_key(category: str) -> int:
    """Get sort key for a category based on CATEGORY_DISPLAY_ORDER."""
    try:
        return CATEGORY_DISPLAY_ORDER.index(category)
    except ValueError:
        return len(CATEGORY_DISPLAY_ORDER)  # Unknown categories go last


# ============================================================================
# Path Resolution Utilities
# ============================================================================

def _get_expected_output_path(method_name: str, output_dir: Path) -> Optional[Path]:
    """Get the expected output path for a benchmark method.
    
    This is a fallback when result_path is not cached.
    
    Parameters
    ----------
    method_name : str
        Name of the benchmark method
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    Optional[Path]
        Expected output file path, or None if cannot be determined
    """
    # Phase-based directories
    preprocessing_dir = output_dir / "preprocessing"
    de_dir = output_dir / "de"
    
    # crispyx methods
    if method_name == "crispyx_qc_filtered":
        return preprocessing_dir / "crispyx_qc_filtered.h5ad"
    elif method_name == "crispyx_pb_avg_log":
        return preprocessing_dir / "crispyx_pb_avg_log.h5ad"
    elif method_name == "crispyx_pb_pseudobulk":
        return preprocessing_dir / "crispyx_pb_pseudobulk.h5ad"
    elif method_name == "crispyx_de_t_test":
        return de_dir / "crispyx_de_t_test.h5ad"
    elif method_name == "crispyx_de_wilcoxon":
        return de_dir / "crispyx_de_wilcoxon.h5ad"
    elif method_name == "crispyx_de_nb_glm":
        return de_dir / "crispyx_de_nb_glm.h5ad"
    elif method_name == "crispyx_de_nb_glm_joint":
        return de_dir / "crispyx_de_nb_glm_joint_nb_glm.h5ad"
    
    # Scanpy methods
    elif method_name == "scanpy_qc_filtered":
        return preprocessing_dir / "scanpy_qc_filtered.h5ad"
    elif method_name == "scanpy_de_t_test":
        return de_dir / "scanpy_de_t_test.h5ad"
    elif method_name == "scanpy_de_wilcoxon":
        return de_dir / "scanpy_de_wilcoxon.h5ad"
    
    # Reference tool CSV outputs
    elif method_name == "edger_de_glm":
        return de_dir / "edger_de_glm.csv"
    elif method_name == "pertpy_de_pydeseq2":
        return de_dir / "pertpy_de_pydeseq2.csv"
    
    return None


def _resolve_result_path(
    method_name: str, 
    result_path_val: Optional[str], 
    output_dir: Path
) -> Optional[Path]:
    """Resolve the result path with fallback to expected path.
    
    Parameters
    ----------
    method_name : str
        Name of the benchmark method
    result_path_val : Optional[str]
        Cached result_path value (may be None or NaN)
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    Optional[Path]
        Resolved path to result file, or None if not found
    """
    # First try using the cached result_path
    if result_path_val is not None and not _is_scalar_na(result_path_val):
        # First try as an absolute path or path relative to workspace root
        result_path = Path(result_path_val)
        if result_path.exists():
            return result_path
        
        # Try as relative path from output_dir 
        result_path = output_dir / str(result_path_val)
        if result_path.exists():
            return result_path
        
        # Try extracting just the filename and looking in expected locations
        filename = Path(result_path_val).name
        for subdir in ["de", "qc", "pb", "preprocessing"]:
            potential_path = output_dir / subdir / filename
            if potential_path.exists():
                return potential_path
    
    # Fallback to expected output path
    expected_path = _get_expected_output_path(method_name, output_dir)
    if expected_path is not None and expected_path.exists():
        return expected_path
    
    return None


# ============================================================================
# Cache Loading Utilities
# ============================================================================

def _load_method_result(method_name: str, output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load individual method benchmark result from cache.
    
    Parameters
    ----------
    method_name : str
        Name of the benchmark method
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Cached result dictionary, or None if cache doesn't exist or is corrupted
    """
    cache_file = output_dir / ".benchmark_cache" / f"{method_name}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _load_cached_results(output_dir: Path) -> List[Dict[str, Any]]:
    """Load all cached benchmark results.
    
    Parameters
    ----------
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    List[Dict[str, Any]]
        List of cached result dictionaries
    """
    cache_dir = output_dir / ".benchmark_cache"
    if not cache_dir.exists():
        return []
    
    cached_results = []
    for cache_file in cache_dir.glob("*.json"):
        if cache_file.name == "config.json":
            continue
        try:
            with open(cache_file) as fh:
                data = json.load(fh)
                if "summary" in data:
                    summary = data.pop("summary")
                    data.update(summary)
                cached_results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    
    return cached_results


def _load_cache_config(output_dir: Path) -> Optional[Dict[str, Any]]:
    """Load cache configuration.
    
    Parameters
    ----------
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Cached config, or None if doesn't exist or is corrupted
    """
    config_file = output_dir / ".benchmark_cache" / "config.json"
    
    if not config_file.exists():
        return None
    
    try:
        with open(config_file) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


# ============================================================================
# DE Result Loading and Conversion Utilities
# ============================================================================

def _anndata_to_de_dict(adata) -> Dict[str, Any]:
    """Convert AnnData with DE results to dictionary format.
    
    Properly extracts 1D dense arrays from sparse matrices in adata.X and adata.layers.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing DE results in layers (crispyx format)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping perturbation names to SimpleNamespace objects with
        genes, effect_size, statistic, and pvalue attributes
    """
    from types import SimpleNamespace
    
    def _extract_row(matrix, idx: int) -> np.ndarray | None:
        """Extract row from matrix (sparse or dense) as 1D dense array."""
        if matrix is None:
            return None
        if idx >= matrix.shape[0]:
            return None
        try:
            row = matrix[idx, :]
            return np.asarray(row).flatten()
        except Exception:
            return None
    
    stream_result_dict = {}
    
    if "pvalue" in adata.layers and "perturbation" in adata.obs.columns:
        perturbations = adata.obs["perturbation"].tolist()
        for idx, group in enumerate(perturbations):
            if "z_score" in adata.layers:
                statistic_values = _extract_row(adata.layers["z_score"], idx)
            elif "statistic" in adata.layers:
                statistic_values = _extract_row(adata.layers["statistic"], idx)
            else:
                statistic_values = None
            
            if "logfoldchange" in adata.layers:
                effect_size_values = _extract_row(adata.layers["logfoldchange"], idx)
            elif adata.X is not None:
                effect_size_values = _extract_row(adata.X, idx)
            else:
                effect_size_values = None
            
            stream_result_dict[group] = SimpleNamespace(
                genes=adata.var_names.to_numpy(),
                effect_size=effect_size_values,
                statistic=statistic_values,
                pvalue=_extract_row(adata.layers["pvalue"], idx),
            )
    elif "rank_genes_groups" in adata.uns:
        rgg = adata.uns["rank_genes_groups"]
        if hasattr(rgg["names"], "dtype") and hasattr(rgg["names"].dtype, "names"):
            groups = list(rgg["names"].dtype.names)
        else:
            groups = []
        
        for group in groups:
            if "names" in rgg:
                genes = rgg["names"][group]
            else:
                genes = adata.uns.get("genes", adata.var_names.to_numpy())

            stream_result_dict[group] = SimpleNamespace(
                genes=genes,
                effect_size=rgg["logfoldchanges"][group] if "logfoldchanges" in rgg else None,
                statistic=rgg["scores"][group] if "scores" in rgg else None,
                pvalue=rgg["pvals"][group] if "pvals" in rgg else None,
            )
    
    return stream_result_dict


def _anndata_to_de_dict_raw(adata) -> Dict[str, Any]:
    """Convert AnnData with DE results to dictionary format, using RAW (unshrunken) LFCs.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing DE results in layers (crispyx format)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary mapping perturbation names to SimpleNamespace objects with
        genes, effect_size (raw), statistic, and pvalue attributes
    """
    from types import SimpleNamespace
    
    def _extract_row(matrix, idx: int) -> np.ndarray | None:
        if matrix is None:
            return None
        if idx >= matrix.shape[0]:
            return None
        try:
            row = matrix[idx, :]
            return np.asarray(row).flatten()
        except Exception:
            return None
    
    stream_result_dict = {}
    
    if "pvalue" in adata.layers and "perturbation" in adata.obs.columns:
        perturbations = adata.obs["perturbation"].tolist()
        for idx, group in enumerate(perturbations):
            if "z_score" in adata.layers:
                statistic_values = _extract_row(adata.layers["z_score"], idx)
            elif "statistic" in adata.layers:
                statistic_values = _extract_row(adata.layers["statistic"], idx)
            else:
                statistic_values = None
            
            if "logfoldchange_raw" in adata.layers:
                effect_size_values = _extract_row(adata.layers["logfoldchange_raw"], idx)
            elif "logfoldchange" in adata.layers:
                effect_size_values = _extract_row(adata.layers["logfoldchange"], idx)
            elif adata.X is not None:
                effect_size_values = _extract_row(adata.X, idx)
            else:
                effect_size_values = None
            
            stream_result_dict[group] = SimpleNamespace(
                genes=adata.var_names.to_numpy(),
                effect_size=effect_size_values,
                statistic=statistic_values,
                pvalue=_extract_row(adata.layers["pvalue"], idx),
            )
    
    return stream_result_dict


def _streaming_de_to_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Convert a streaming differential expression mapping to a tidy DataFrame."""
    frames = []
    for perturbation, entry in result.items():
        genes = getattr(entry, "genes", None)
        if genes is None:
            continue
        gene_index = pd.Index(genes).astype(str)
        n_rows = len(gene_index)
        frame = pd.DataFrame(
            {
                "perturbation": [str(perturbation)] * n_rows,
                "gene": gene_index,
                "effect_size": getattr(entry, "effect_size", pd.NA),
                "statistic": getattr(entry, "statistic", pd.NA),
                "pvalue": getattr(entry, "pvalue", pd.NA),
            }
        )
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=_STANDARD_DE_COLUMNS)


def _standardise_de_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return ``df`` with standard differential expression column names."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_STANDARD_DE_COLUMNS)

    result = df.copy()
    lower_to_original = {col.lower(): col for col in result.columns}

    def _resolve_column(candidates: Iterable[str]) -> Optional[str]:
        for candidate in candidates:
            original = lower_to_original.get(candidate.lower())
            if original is not None:
                return original
        return None

    rename: Dict[str, str] = {}
    perturbation_col = _resolve_column(["perturbation", "group", "cluster", "label", "contrast"])
    gene_col = _resolve_column(["gene", "genes", "name", "names", "feature", "variable"])
    effect_col = _resolve_column(["effect_size", "logfoldchange", "logfoldchanges", "logfc", "lfc", "log_fc", "coefficient"])
    stat_col = _resolve_column(["statistic", "statistics", "stat", "score", "scores", "wald_statistic", "zscore", "t_stat", "t_value", "t_statistic", "f", "f_value", "u_stat"])
    pvalue_col = _resolve_column(["pvalue", "p_value", "pval", "pvals", "pvalue_raw", "pvalue_adj", "pvals_adj"])

    if perturbation_col is not None:
        rename[perturbation_col] = "perturbation"
    if gene_col is not None:
        rename[gene_col] = "gene"
    if effect_col is not None:
        rename[effect_col] = "effect_size"
    if stat_col is not None:
        rename[stat_col] = "statistic"
    if pvalue_col is not None:
        rename[pvalue_col] = "pvalue"

    result = result.rename(columns=rename)

    for column in _STANDARD_DE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result = result[_STANDARD_DE_COLUMNS]
    result["perturbation"] = result["perturbation"].astype(str).str.strip()
    result["gene"] = result["gene"].astype(str).str.strip()
    result["effect_size"] = pd.to_numeric(result["effect_size"], errors="coerce")
    result["statistic"] = pd.to_numeric(result["statistic"], errors="coerce")
    result["pvalue"] = pd.to_numeric(result["pvalue"], errors="coerce")
    return result


# ============================================================================
# Formatting Helpers
# ============================================================================

def _get_performance_emoji(pct: Optional[float], is_lower_better: bool = True) -> str:
    """Return emoji indicator for performance comparison."""
    if pct is None or _is_scalar_na(pct):
        return ""
    
    if is_lower_better:
        if pct < 90:
            return "✅"
        elif pct > 110:
            return "❌"
        else:
            return "⚠️"
    else:
        if pct > 110:
            return "✅"
        elif pct < 90:
            return "❌"
        else:
            return "⚠️"


def _get_accuracy_emoji(corr: Optional[float]) -> str:
    """Return emoji indicator for accuracy/correlation."""
    if corr is None or _is_scalar_na(corr):
        return ""
    
    if corr >= 0.95:
        return "✅"
    elif corr >= 0.8:
        return "⚠️"
    else:
        return "❌"


def _format_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    """Format mean ± std as a two-line string for markdown tables."""
    if mean is None or _is_scalar_na(mean):
        return "-"
    
    if std is None or _is_scalar_na(std) or std == 0:
        return f"{mean:.3f}"
    
    return f"{mean:.3f}<br><small>±{std:.3f}</small>"


def _get_method_category(method_name: str) -> tuple[str, str, int]:
    """Get category and test type for a method."""
    if not isinstance(method_name, str):
        return ("other", "unknown", 99)
    
    if "_qc_" in method_name or method_name.endswith("_qc_filtered"):
        return ("Preprocessing / QC", "qc", 0)
    elif "_pb_avg" in method_name:
        return ("Preprocessing / QC", "pseudobulk_avg", 1)
    elif "_pb_pseudobulk" in method_name:
        return ("Preprocessing / QC", "pseudobulk", 2)
    elif "_de_t_test" in method_name:
        return ("DE: t-test", "t_test", 10)
    elif "_de_wilcoxon" in method_name:
        return ("DE: Wilcoxon", "wilcoxon", 20)
    elif "_de_nb_glm_joint" in method_name:
        return ("DE: NB GLM", "nb_glm_joint", 31)
    elif "_de_nb_glm" in method_name:
        return ("DE: NB GLM", "nb_glm", 30)
    elif "_de_glm" in method_name:
        return ("DE: NB GLM", "edger", 32)
    elif "_de_pydeseq2_shrunk" in method_name:
        return ("DE: NB GLM", "pydeseq2_shrunk", 34)
    elif "_de_pydeseq2" in method_name:
        return ("DE: NB GLM", "pydeseq2", 33)
    else:
        return ("Other", "unknown", 99)


def _format_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format percentage value."""
    if value is None or _is_scalar_na(value):
        return "-"
    return f"{value:.{decimals}f}%"


def _format_diff(value: Optional[float], unit: str = "s", decimals: int = 1) -> str:
    """Format difference value with sign."""
    if value is None or _is_scalar_na(value):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}{unit}"


def _format_method_name(method: str) -> str:
    """Format method name for display with proper capitalization.
    
    Note: shrinkage status is indicated via the lfcShrink column, not in method name.
    """
    if "edger_" in method.lower():
        return "NB-GLM"
    if "pertpy_" in method.lower() and "pydeseq2" in method.lower():
        return "NB-GLM"
    
    name = method.replace("crispyx_", "").replace("scanpy_", "").replace("pertpy_", "").replace("edger_", "")
    name = name.replace("_", " ")
    
    # Order matters: match longer patterns first to avoid partial matches
    replacements = {
        "de t test": "t-test",
        "de wilcoxon": "Wilcoxon",
        "de nb glm joint shrunk": "NB-GLM (joint)",  # shrunk suffix stripped
        "de nb glm joint": "NB-GLM (joint)",
        "de nb glm shrunk": "NB-GLM",               # shrunk suffix stripped
        "de nb glm": "NB-GLM",
        "de glm": "NB-GLM",
        "de pydeseq2 shrunk": "NB-GLM",              # shrunk suffix stripped
        "de pydeseq2": "NB-GLM",
        "qc filtered": "QC filter",
        "pb avg log": "pseudobulk (avg log)",
        "pb avg": "pseudobulk (avg)",
        "pb pseudobulk": "pseudobulk",
    }
    
    for old, new in replacements.items():
        if old in name:
            name = name.replace(old, new)
            break
    
    return name


def _get_method_package(method: str) -> str:
    """Get the package name for a method."""
    if method.startswith("crispyx_"):
        return "crispyx"
    elif method.startswith("scanpy_"):
        return "scanpy"
    elif method.startswith("edger_"):
        return "edgeR"
    elif method.startswith("pertpy_"):
        return "pertpy"
    return ""


def _format_full_method_name(method: str) -> str:
    """Format full method name including package."""
    package = _get_method_package(method)
    name = _format_method_name(method)
    if package:
        return f"{package} {name}"
    return name


def _is_crispyx_method(method: str) -> bool:
    """Check if method is a crispyx method."""
    return method.startswith("crispyx_")


def _get_shrinkage_type(method: str) -> str:
    """Get the shrinkage type for a method, or empty string if none."""
    # Strip any suffix like " (lfcShrink)" that may be added in comparison strings
    clean_method = method.split(" (")[0] if " (" in method else method
    return SHRINKAGE_METADATA.get(clean_method, "")


def _frame_to_markdown_table(table: pd.DataFrame) -> str:
    """Render ``table`` as a Markdown table suitable for GitHub."""
    if table.empty:
        return "| |\n|---|\n"

    formatted = table.copy()
    numeric_cols = formatted.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        formatted[numeric_cols] = formatted[numeric_cols].round(3)

    headers = list(formatted.columns)
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = []
    for _, row in formatted.iterrows():
        values = []
        for value in row:
            if _is_scalar_na(value):
                values.append("")
            else:
                values.append(str(value))
        data_rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header_row, separator_row, *data_rows]) + "\n"


# ============================================================================
# Markdown Generation
# ============================================================================

def _generate_improved_markdown(
    perf_df: pd.DataFrame,
    perf_comp_results: List[Dict[str, Any]],
    accuracy_results: List[Dict[str, Any]],
    overlap_heatmaps: Optional[Dict[str, Path]] = None,
) -> str:
    """Generate improved markdown with categorized tables and emoji indicators."""
    
    md = "# Benchmark Results\n\n"
    
    # =========================================================================
    # Section 1: Performance by Category
    # =========================================================================
    md += "## 1. Performance\n\n"
    
    if not perf_df.empty:
        perf_df = perf_df.copy()
        perf_df["_category"] = perf_df["method"].apply(lambda x: _get_method_category(x)[0])
        perf_df["_sort_order"] = perf_df["method"].apply(lambda x: _get_method_category(x)[2])
        perf_df = perf_df.sort_values("_sort_order")
        
        perf_df["Package"] = perf_df["method"].apply(_get_method_package)
        perf_df["Method"] = perf_df["method"].apply(_format_method_name)
        
        categories = perf_df["_category"].unique()
        
        for category in categories:
            cat_df = perf_df[perf_df["_category"] == category].copy()
            
            md += f"### {category}\n\n"
            
            # Add lfcShrink column for NB GLM category
            is_nb_glm = "NB GLM" in category
            
            if "Preprocessing" in category or "QC" in category:
                cols = ["Package", "Method", "status", "elapsed_seconds", "peak_memory_mb", "cells_kept", "genes_kept"]
            else:
                if is_nb_glm:
                    cols = ["Package", "Method", "status", "elapsed_seconds", "peak_memory_mb", "groups", "genes"]
                else:
                    cols = ["Package", "Method", "status", "elapsed_seconds", "peak_memory_mb", "groups", "genes"]
            
            cols = [c for c in cols if c in cat_df.columns]
            display_df = cat_df[cols].copy()
            
            # Add lfcShrink column for NB GLM (use tick mark)
            if is_nb_glm:
                display_df.insert(2, "lfcShrink", cat_df["method"].apply(lambda m: "✓" if _get_shrinkage_type(m) else ""))
            
            rename_map = {
                "status": "Status",
                "elapsed_seconds": "Time (s)",
                "peak_memory_mb": "Memory (MB)",
                "cells_kept": "Cells",
                "genes_kept": "Genes",
                "groups": "Groups",
                "genes": "Genes",
            }
            display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
            
            def _safe_int(x):
                """Safely convert to int, handling arrays, lists, and NA values."""
                if _is_scalar_na(x):
                    return x
                if isinstance(x, (int, float, np.integer, np.floating)):
                    return int(x)
                if isinstance(x, str):
                    try:
                        return int(float(x))
                    except (ValueError, TypeError):
                        return x
                # For lists, arrays, etc., return length or the original
                if isinstance(x, (list, np.ndarray)):
                    return len(x)
                return x
            
            for col in display_df.columns:
                if col in ["Cells", "Genes", "Groups"]:
                    display_df[col] = display_df[col].apply(_safe_int)
                elif col in display_df.select_dtypes(include=["number"]).columns:
                    display_df[col] = display_df[col].round(2)
            
            md += _frame_to_markdown_table(display_df)
            md += "\n\n"
    
    # =========================================================================
    # Section 2: Performance Comparison
    # =========================================================================
    md += "## 2. Performance Comparison\n\n"
    
    if perf_comp_results:
        crispyx_comps = []
        other_comps = []
        
        for comp in perf_comp_results:
            comparison = comp["comparison"]
            method_a = comparison.split(" vs ")[0]
            if _is_crispyx_method(method_a):
                crispyx_comps.append(comp)
            else:
                other_comps.append(comp)
        
        if crispyx_comps:
            md += "### crispyx vs Reference Tools\n\n"
            md += "_crispyx as baseline. Negative values = crispyx is faster/uses less memory._\n\n"
            
            comp_by_category: Dict[str, List[Dict[str, Any]]] = {}
            for comp in crispyx_comps:
                comparison = comp["comparison"]
                method_a = comparison.split(" vs ")[0]
                category = _get_method_category(method_a)[0]
                
                if category not in comp_by_category:
                    comp_by_category[category] = []
                comp_by_category[category].append(comp)
            
            # Sort categories by predefined order
            sorted_categories = sorted(comp_by_category.keys(), key=_get_category_sort_key)
            
            for category in sorted_categories:
                comps = comp_by_category[category]
                md += f"#### {category}\n\n"
                
                is_nb_glm = "NB GLM" in category
                rows = []
                for comp in comps:
                    comparison = comp["comparison"]
                    parts = comparison.split(" vs ")
                    method_a_raw = parts[0]
                    method_b_raw = parts[1]
                    method_a = _format_method_name(method_a_raw)
                    method_b = _format_full_method_name(method_b_raw)
                    
                    time_pct = comp.get("time_pct")
                    mem_pct = comp.get("mem_pct")
                    time_diff = comp.get("time_diff_s")
                    mem_diff = comp.get("mem_diff_mb")
                    
                    time_emoji = _get_performance_emoji(time_pct, is_lower_better=True)
                    mem_emoji = _get_performance_emoji(mem_pct, is_lower_better=True)
                    
                    row = {
                        "crispyx method": method_a,
                    }
                    if is_nb_glm:
                        # Use tick mark for lfcShrink columns
                        row["lfcShrink"] = "✓" if _get_shrinkage_type(method_a_raw) else ""
                        row["compared to"] = method_b
                        row["lfcShrink (ref)"] = "✓" if _get_shrinkage_type(method_b_raw) else ""
                    else:
                        row["compared to"] = method_b
                    row.update({
                        "Time Δ": _format_diff(time_diff, "s"),
                        "Time %": _format_pct(time_pct),
                        "": time_emoji,
                        "Mem Δ": _format_diff(mem_diff, " MB"),
                        "Mem %": _format_pct(mem_pct),
                        " ": mem_emoji,
                    })
                    rows.append(row)
                
                # Sort rows by method order: t-test, Wilcoxon, NB-GLM, NB-GLM (joint)
                rows.sort(key=lambda r: (_get_method_sort_key(r["crispyx method"]), r["crispyx method"], r.get("lfcShrink", ""), r["compared to"]))
                md += _frame_to_markdown_table(pd.DataFrame(rows))
                md += "\n\n"
        
        if other_comps:
            md += "### Tool Comparisons\n\n"
            md += "_Comparisons between external tools._\n\n"
            
            rows = []
            for comp in other_comps:
                comparison = comp["comparison"]
                parts = comparison.split(" vs ")
                method_a = _format_method_name(parts[0])
                method_b = _format_method_name(parts[1])
                
                time_pct = comp.get("time_pct")
                mem_pct = comp.get("mem_pct")
                time_diff = comp.get("time_diff_s")
                mem_diff = comp.get("mem_diff_mb")
                
                time_emoji = _get_performance_emoji(time_pct, is_lower_better=True)
                mem_emoji = _get_performance_emoji(mem_pct, is_lower_better=True)
                
                rows.append({
                    "package A": _get_method_package(parts[0]),
                    "method A": method_a,
                    "package B": _get_method_package(parts[1]),
                    "method B": method_b,
                    "Time Δ (A-B)": _format_diff(time_diff, "s"),
                    "Time % (A/B)": _format_pct(time_pct),
                    "": time_emoji,
                    "Mem Δ (A-B)": _format_diff(mem_diff, " MB"),
                    "Mem % (A/B)": _format_pct(mem_pct),
                    " ": mem_emoji,
                })
            
            md += _frame_to_markdown_table(pd.DataFrame(rows))
            md += "\n\n"
    
    # =========================================================================
    # Section 3: Accuracy Comparison
    # =========================================================================
    md += "## 3. Accuracy\n\n"
    md += "_Correlation metrics between crispyx and reference methods. ✅ >0.95, ⚠️ 0.8-0.95, ❌ <0.8_\n\n"
    
    if accuracy_results:
        crispyx_accs = []
        other_accs = []
        
        for acc in accuracy_results:
            comparison = acc["comparison"]
            method_a = comparison.split(" vs ")[0]
            if _is_crispyx_method(method_a):
                crispyx_accs.append(acc)
            else:
                other_accs.append(acc)
        
        if crispyx_accs:
            acc_by_category: Dict[str, List[Dict[str, Any]]] = {}
            
            for acc in crispyx_accs:
                comparison = acc["comparison"]
                method_a = comparison.split(" vs ")[0]
                category = _get_method_category(method_a)[0]
                
                if category not in acc_by_category:
                    acc_by_category[category] = []
                acc_by_category[category].append(acc)
            
            # Sort categories by predefined order
            sorted_categories = sorted(acc_by_category.keys(), key=_get_category_sort_key)
            
            for category in sorted_categories:
                accs = acc_by_category[category]
                md += f"### {category}\n\n"
                
                is_qc = "QC" in category or "Preprocessing" in category
                is_nb_glm = "NB GLM" in category
                
                if is_qc:
                    rows = []
                    for acc in accs:
                        comparison = acc["comparison"]
                        parts = comparison.split(" vs ")
                        method_a = _format_method_name(parts[0])
                        method_b = _format_full_method_name(parts[1])
                        
                        cells_diff = acc.get("cells_diff", 0)
                        genes_diff = acc.get("genes_diff", 0)
                        
                        cells_emoji = "✅" if cells_diff == 0 else ("⚠️" if abs(cells_diff) < 10 else "❌")
                        genes_emoji = "✅" if genes_diff == 0 else ("⚠️" if abs(genes_diff) < 10 else "❌")
                        
                        rows.append({
                            "crispyx method": method_a,
                            "compared to": method_b,
                            "Cells Δ": f"{int(cells_diff):+d}" if _is_scalar_notna(cells_diff) else "-",
                            "": cells_emoji,
                            "Genes Δ": f"{int(genes_diff):+d}" if _is_scalar_notna(genes_diff) else "-",
                            " ": genes_emoji,
                        })
                    
                    md += _frame_to_markdown_table(pd.DataFrame(rows))
                else:
                    rows = []
                    for acc in accs:
                        comparison = acc["comparison"]
                        parts = comparison.split(" vs ")
                        method_a_raw = parts[0]
                        method_b_raw = parts[1]
                        method_a = _format_method_name(method_a_raw)
                        method_b = _format_full_method_name(method_b_raw)
                        
                        effect_p_mean = acc.get("effect_pearson_corr_mean")
                        effect_p_std = acc.get("effect_pearson_corr_std")
                        effect_s_mean = acc.get("effect_spearman_corr_mean")
                        effect_s_std = acc.get("effect_spearman_corr_std")
                        stat_p_mean = acc.get("statistic_pearson_corr_mean")
                        stat_p_std = acc.get("statistic_pearson_corr_std")
                        stat_s_mean = acc.get("statistic_spearman_corr_mean")
                        stat_s_std = acc.get("statistic_spearman_corr_std")
                        pval_p_mean = acc.get("pvalue_log_pearson_corr_mean")
                        pval_p_std = acc.get("pvalue_log_pearson_corr_std")
                        pval_s_mean = acc.get("pvalue_log_spearman_corr_mean")
                        pval_s_std = acc.get("pvalue_log_spearman_corr_std")
                        
                        row = {
                            "crispyx method": method_a,
                        }
                        if is_nb_glm:
                            # Use tick mark for lfcShrink columns
                            row["lfcShrink"] = "✓" if _get_shrinkage_type(method_a_raw) else ""
                            row["compared to"] = method_b
                            row["lfcShrink (ref)"] = "✓" if _get_shrinkage_type(method_b_raw) else ""
                        else:
                            row["compared to"] = method_b
                        row.update({
                            "Eff ρ": _format_mean_std(effect_p_mean, effect_p_std),
                            "": _get_accuracy_emoji(effect_p_mean),
                            "Eff ρₛ": _format_mean_std(effect_s_mean, effect_s_std),
                            " ": _get_accuracy_emoji(effect_s_mean),
                            "Stat ρ": _format_mean_std(stat_p_mean, stat_p_std),
                            "  ": _get_accuracy_emoji(stat_p_mean),
                            "Stat ρₛ": _format_mean_std(stat_s_mean, stat_s_std),
                            "   ": _get_accuracy_emoji(stat_s_mean),
                            "log-Pval ρ": _format_mean_std(pval_p_mean, pval_p_std),
                            "    ": _get_accuracy_emoji(pval_p_mean),
                            "log-Pval ρₛ": _format_mean_std(pval_s_mean, pval_s_std),
                            "     ": _get_accuracy_emoji(pval_s_mean),
                        })
                        rows.append(row)
                    
                    # Sort rows by method order: t-test, Wilcoxon, NB-GLM, NB-GLM (joint)
                    rows.sort(key=lambda r: (_get_method_sort_key(r["crispyx method"]), r["crispyx method"], r.get("lfcShrink", ""), r["compared to"]))
                    md += _frame_to_markdown_table(pd.DataFrame(rows))
                
                md += "\n\n"
        
        if other_accs:
            md += "### Tool Comparisons\n\n"
            
            rows = []
            for acc in other_accs:
                comparison = acc["comparison"]
                parts = comparison.split(" vs ")
                method_a = _format_method_name(parts[0])
                method_b = _format_method_name(parts[1])
                
                effect_p_mean = acc.get("effect_pearson_corr_mean")
                effect_p_std = acc.get("effect_pearson_corr_std")
                effect_s_mean = acc.get("effect_spearman_corr_mean")
                effect_s_std = acc.get("effect_spearman_corr_std")
                stat_p_mean = acc.get("statistic_pearson_corr_mean")
                stat_p_std = acc.get("statistic_pearson_corr_std")
                stat_s_mean = acc.get("statistic_spearman_corr_mean")
                stat_s_std = acc.get("statistic_spearman_corr_std")
                pval_p_mean = acc.get("pvalue_log_pearson_corr_mean")
                pval_p_std = acc.get("pvalue_log_pearson_corr_std")
                pval_s_mean = acc.get("pvalue_log_spearman_corr_mean")
                pval_s_std = acc.get("pvalue_log_spearman_corr_std")
                
                rows.append({
                    "package A": _get_method_package(parts[0]),
                    "method A": method_a,
                    "package B": _get_method_package(parts[1]),
                    "method B": method_b,
                    "Eff ρ": _format_mean_std(effect_p_mean, effect_p_std),
                    "": _get_accuracy_emoji(effect_p_mean),
                    "Eff ρₛ": _format_mean_std(effect_s_mean, effect_s_std),
                    " ": _get_accuracy_emoji(effect_s_mean),
                    "Stat ρ": _format_mean_std(stat_p_mean, stat_p_std),
                    "  ": _get_accuracy_emoji(stat_p_mean),
                    "Stat ρₛ": _format_mean_std(stat_s_mean, stat_s_std),
                    "   ": _get_accuracy_emoji(stat_s_mean),
                    "log-Pval ρ": _format_mean_std(pval_p_mean, pval_p_std),
                    "    ": _get_accuracy_emoji(pval_p_mean),
                    "log-Pval ρₛ": _format_mean_std(pval_s_mean, pval_s_std),
                    "     ": _get_accuracy_emoji(pval_s_mean),
                })
            
            md += _frame_to_markdown_table(pd.DataFrame(rows))
            md += "\n\n"
    
    # =========================================================================
    # Section 4: Gene Set Overlap
    # =========================================================================
    md += "## 4. Gene Set Overlap\n\n"
    md += "_Overlap ratio of top-k DE genes between methods. ✅ >0.7, ⚠️ 0.5-0.7, ❌ <0.5_\n\n"
    
    if accuracy_results:
        for metric_type, metric_label in [("effect", "Effect Size"), ("pvalue", "P-value")]:
            md += f"### {metric_label} Overlap\n\n"
            
            rows = []
            for acc in accuracy_results:
                comparison = acc["comparison"]
                parts = comparison.split(" vs ")
                method_a_raw = parts[0]
                method_b_raw = parts[1]
                method_a = _format_method_name(method_a_raw)
                method_b = _format_full_method_name(method_b_raw)
                
                # Remove "(lfcShrink)" suffix from method_b display name since we have a column for it
                method_b = method_b.replace(" (lfcShrink)", "")
                
                k50 = acc.get(f"{metric_type}_top_k_overlap")
                k100_mean = acc.get(f"{metric_type}_top_100_overlap_mean")
                k100_std = acc.get(f"{metric_type}_top_100_overlap_std")
                k500_mean = acc.get(f"{metric_type}_top_500_overlap_mean")
                k500_std = acc.get(f"{metric_type}_top_500_overlap_std")
                
                if k50 is None and k100_mean is None and k500_mean is None:
                    continue
                
                def _get_overlap_emoji(val: Optional[float]) -> str:
                    if val is None or _is_scalar_na(val):
                        return ""
                    if val >= 0.7:
                        return "✅"
                    elif val >= 0.5:
                        return "⚠️"
                    else:
                        return "❌"
                
                # Use tick for lfcShrink columns
                shrink_a = "✓" if _get_shrinkage_type(method_a_raw) else ""
                shrink_b = "✓" if _get_shrinkage_type(method_b_raw) else ""
                
                rows.append({
                    "crispyx method": method_a,
                    "lfcShrink": shrink_a,
                    "compared to": method_b,
                    "lfcShrink (ref)": shrink_b,
                    "Top-50": f"{k50:.3f}" if k50 is not None and _is_scalar_notna(k50) else "-",
                    "": _get_overlap_emoji(k50),
                    "Top-100": _format_mean_std(k100_mean, k100_std),
                    " ": _get_overlap_emoji(k100_mean),
                    "Top-500": _format_mean_std(k500_mean, k500_std),
                    "  ": _get_overlap_emoji(k500_mean),
                })
            
            if rows:
                # Sort rows by method order: t-test, Wilcoxon, NB-GLM, NB-GLM (joint)
                rows.sort(key=lambda r: (_get_method_sort_key(r["crispyx method"]), r["crispyx method"], r.get("lfcShrink", ""), r["compared to"]))
                md += _frame_to_markdown_table(pd.DataFrame(rows))
            else:
                md += "_No overlap data available._\n"
            md += "\n\n"
    else:
        md += "_No overlap data available._\n\n"
    
    # Embed top-100 overlap heatmaps
    if overlap_heatmaps:
        md += "### Overlap Heatmaps (Top-100)\n\n"
        
        for metric, label in [("effect", "Effect Size"), ("pvalue", "P-value")]:
            heatmap_key = f"benchmark_{metric}_top_100_overlap.png"
            if heatmap_key in overlap_heatmaps:
                md += f"#### {label}\n\n"
                md += f"![{label} Top-100 Overlap]({heatmap_key})\n\n"
    
    # =========================================================================
    # Legend
    # =========================================================================
    md += "---\n\n"
    md += "**Legend:**\n"
    md += "- Performance: ✅ >10% better | ⚠️ within ±10% | ❌ >10% worse\n"
    md += "- Accuracy: ✅ ρ≥0.95 | ⚠️ 0.8≤ρ<0.95 | ❌ ρ<0.8\n"
    md += "- Overlap: ✅ ≥0.7 | ⚠️ 0.5-0.7 | ❌ <0.5\n"
    md += "- ρ = Pearson correlation, ρₛ = Spearman correlation\n"
    md += "- Correlation and overlap values shown as mean±std across perturbations\n"
    md += "- log-Pval: correlations computed on -log₁₀(p) transformed values\n"
    md += "- Top-k overlap: fraction of top-k genes shared between methods\n"
    md += "- lfcShrink column: shrinkage type used (apeglm, ashr, normal) or blank if none\n"
    
    return md


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_benchmarks(output_dir: Path) -> None:
    """Evaluate benchmark results and generate report.
    
    This function loads cached benchmark results, computes comparison metrics,
    generates overlap heatmaps, and writes summary files.
    
    Output files (all with benchmark_ prefix):
    - benchmark_performance.csv: Performance metrics table
    - benchmark_accuracy.csv: Accuracy comparison metrics
    - benchmark_report.md: Full markdown report
    - benchmark_effect_top_{k}_overlap.png: Effect size overlap heatmaps
    - benchmark_pvalue_top_{k}_overlap.png: P-value overlap heatmaps
    
    Parameters
    ----------
    output_dir : Path
        Directory containing .benchmark_cache with cached results
    """
    from .comparison import compute_de_comparison_metrics
    
    results_dir = output_dir / ".benchmark_cache"
    if not results_dir.exists():
        return
        
    # Load all results
    results = _load_cached_results(output_dir)
    if not results:
        return
        
    df = pd.DataFrame(results)
    
    # Rename max_memory_mb to peak_memory_mb if needed
    if "max_memory_mb" in df.columns and "peak_memory_mb" not in df.columns:
        df = df.rename(columns={"max_memory_mb": "peak_memory_mb"})
    
    # Generate Performance Table
    perf_cols = [
        "method", "status", "elapsed_seconds", "spawn_overhead_seconds",
        "import_seconds", "load_seconds", "process_seconds", 
        "de_seconds", "convert_seconds", "save_seconds",
        "peak_memory_mb", "avg_memory_mb",
        "cells_kept", "genes_kept", "groups"
    ]
    perf_df = df[[c for c in perf_cols if c in df.columns]].copy()
    if "method" in perf_df.columns:
        perf_df = perf_df.sort_values("method")
    
    accuracy_results = []
    perf_comp_results = []
    de_results_for_heatmaps: Dict[str, pd.DataFrame] = {}
    
    # Define comparisons
    # Note: We only compare like-with-like:
    # - base crispyx vs base tools (edgeR, PyDESeq2) - uses result_path
    # - shrunk crispyx vs shrunk tools (PyDESeq2) - uses shrunk_result_path for de_lfcshrink
    # - shrunk crispyx also vs edgeR (as a reference baseline)
    # For de_lfcshrink comparisons, the code automatically uses shrunk_result_path
    comparisons = [
        ("crispyx_qc_filtered", "scanpy_qc_filtered", "qc"),
        # DE GLM - base independent vs external tools (base LFCs)
        ("crispyx_de_nb_glm", "edger_de_glm", "de"),
        ("crispyx_de_nb_glm", "pertpy_de_pydeseq2", "de"),
        # DE GLM - shrunk independent vs shrunk PyDESeq2 (uses shrunk_result_path)
        ("crispyx_de_nb_glm", "pertpy_de_pydeseq2", "de_lfcshrink"),
        # DE GLM - base joint vs base tools
        ("crispyx_de_nb_glm_joint", "edger_de_glm", "de"),
        ("crispyx_de_nb_glm_joint", "pertpy_de_pydeseq2", "de"),
        # DE GLM - shrunk joint vs shrunk PyDESeq2 (uses shrunk_result_path)
        ("crispyx_de_nb_glm_joint", "pertpy_de_pydeseq2", "de_lfcshrink"),
        # DE GLM - external tool comparison (base)
        ("edger_de_glm", "pertpy_de_pydeseq2", "de"),
        # DE Tests
        ("crispyx_de_t_test", "scanpy_de_t_test", "de"),
        ("crispyx_de_wilcoxon", "scanpy_de_wilcoxon", "de"),
    ]
    
    def _load_de_result(method_name: str, result_path_val: Optional[str], use_raw_lfc: bool = False) -> Optional[pd.DataFrame]:
        """Load and standardize a DE result from file.
        
        Uses _resolve_result_path to fallback to expected path if result_path is missing.
        """
        # Resolve the path with fallback to expected output path
        result_path = _resolve_result_path(method_name, result_path_val, output_dir)
        if result_path is None:
            return None
            
        try:
            if str(result_path).endswith('.h5ad'):
                import anndata as ad
                adata = ad.read_h5ad(str(result_path))
                if use_raw_lfc:
                    result_dict = _anndata_to_de_dict_raw(adata)
                else:
                    result_dict = _anndata_to_de_dict(adata)
                return _streaming_de_to_frame(result_dict)
            else:
                result_df = pd.read_csv(result_path)
                return _standardise_de_dataframe(result_df)
        except Exception as e:
            print(f"Warning: Could not load DE result for {method_name}: {e}")
            return None
    
    # Explicitly collect all DE methods for heatmaps (base results)
    for method_name in ALL_DE_METHODS_FOR_HEATMAP:
        method_res = df[df["method"] == method_name]
        if method_res.empty or method_res.iloc[0]["status"] != "success":
            continue
        result_path_val = method_res.iloc[0].get("result_path")
        # Now _load_de_result handles fallback, so we can always try to load
        if method_name not in de_results_for_heatmaps:
            method_df = _load_de_result(method_name, result_path_val)
            if method_df is not None:
                de_results_for_heatmaps[method_name] = method_df
    
    # Also collect shrunk results for heatmaps (stored with _shrunk suffix)
    for method_name in METHODS_WITH_SHRUNK_OUTPUT:
        method_res = df[df["method"] == method_name]
        if method_res.empty or method_res.iloc[0]["status"] != "success":
            continue
        shrunk_path_val = method_res.iloc[0].get("shrunk_result_path")
        if pd.isna(shrunk_path_val) or shrunk_path_val is None:
            continue
        shrunk_method_name = f"{method_name}_shrunk"
        if shrunk_method_name not in de_results_for_heatmaps:
            # Load shrunk result directly from path
            # Handle both absolute paths and paths relative to project root
            shrunk_path_str = str(shrunk_path_val)
            if shrunk_path_str.startswith("/"):
                # Absolute path
                shrunk_path = Path(shrunk_path_str)
            elif shrunk_path_str.startswith("benchmarking/") or shrunk_path_str.startswith("de/"):
                # Path relative to project root or output_dir
                if shrunk_path_str.startswith("de/"):
                    shrunk_path = output_dir / shrunk_path_str
                else:
                    # Path from project root - extract the part after output_dir
                    shrunk_path = Path(shrunk_path_str)
            else:
                shrunk_path = output_dir / shrunk_path_str
            if shrunk_path.exists():
                try:
                    if str(shrunk_path).endswith('.h5ad'):
                        import anndata as ad
                        adata = ad.read_h5ad(str(shrunk_path))
                        result_dict = _anndata_to_de_dict(adata)
                        method_df = _streaming_de_to_frame(result_dict)
                    else:
                        result_df = pd.read_csv(shrunk_path)
                        method_df = _standardise_de_dataframe(result_df)
                    if method_df is not None:
                        de_results_for_heatmaps[shrunk_method_name] = method_df
                except Exception as e:
                    print(f"Warning: Could not load shrunk DE result for {shrunk_method_name}: {e}")
    
    for method_a_name, method_b_name, comp_type in comparisons:
        method_a_res = df[df["method"] == method_a_name]
        method_b_res = df[df["method"] == method_b_name]
        
        if method_a_res.empty or method_b_res.empty:
            continue
            
        if method_a_res.iloc[0]["status"] != "success" or method_b_res.iloc[0]["status"] != "success":
            continue
            
        a_row = method_a_res.iloc[0]
        b_row = method_b_res.iloc[0]
        
        a_time = a_row.get("elapsed_seconds", np.nan)
        b_time = b_row.get("elapsed_seconds", np.nan)
        a_mem = a_row.get("peak_memory_mb", np.nan)
        b_mem = b_row.get("peak_memory_mb", np.nan)
        
        comp = {
            "comparison": f"{method_a_name} vs {method_b_name}",
            "method_a_time_s": a_time,
            "method_b_time_s": b_time,
            "time_diff_s": a_time - b_time if _is_scalar_notna(a_time) and _is_scalar_notna(b_time) else None,
            "time_pct": (a_time / b_time * 100) if _is_scalar_notna(a_time) and _is_scalar_notna(b_time) and b_time > 0 else None,
            "method_a_mem_mb": a_mem,
            "method_b_mem_mb": b_mem,
            "mem_diff_mb": a_mem - b_mem if _is_scalar_notna(a_mem) and _is_scalar_notna(b_mem) else None,
            "mem_pct": (a_mem / b_mem * 100) if _is_scalar_notna(a_mem) and _is_scalar_notna(b_mem) and b_mem > 0 else None,
        }
        perf_comp_results.append(comp)
        
        try:
            # For de_lfcshrink comparisons, use shrunk_result_path if available
            if comp_type == "de_lfcshrink":
                method_a_path_val = method_a_res.iloc[0].get("shrunk_result_path", method_a_res.iloc[0].get("result_path"))
                method_b_path_val = method_b_res.iloc[0].get("shrunk_result_path", method_b_res.iloc[0].get("result_path"))
                # Handle case where shrunk_result_path is NaN - fall back to result_path
                if pd.isna(method_a_path_val):
                    method_a_path_val = method_a_res.iloc[0].get("result_path")
                if pd.isna(method_b_path_val):
                    method_b_path_val = method_b_res.iloc[0].get("result_path")
            else:
                # Get result_path values from cache (may be None)
                method_a_path_val = method_a_res.iloc[0].get("result_path")
                method_b_path_val = method_b_res.iloc[0].get("result_path")
            
            # Use _resolve_result_path for fallback - don't skip based on cached value alone
            method_a_resolved = _resolve_result_path(method_a_name, method_a_path_val, output_dir)
            method_b_resolved = _resolve_result_path(method_b_name, method_b_path_val, output_dir)
            
            if method_a_resolved is None or method_b_resolved is None:
                print(f"Skipping comparison {method_a_name} vs {method_b_name}: missing result file")
                continue
            
            if comp_type == "qc":
                acc = {
                    "comparison": f"{method_a_name} vs {method_b_name}",
                    "cells_diff": float(method_a_res.iloc[0]["cells_kept"] - method_b_res.iloc[0]["cells_kept"]),
                    "genes_diff": float(method_a_res.iloc[0]["genes_kept"] - method_b_res.iloc[0]["genes_kept"]),
                }
                accuracy_results.append(acc)
                
            elif comp_type == "de":
                method_a_df = _load_de_result(method_a_name, method_a_path_val)
                method_b_df = _load_de_result(method_b_name, method_b_path_val)
                
                if method_a_df is not None and method_a_name not in de_results_for_heatmaps:
                    de_results_for_heatmaps[method_a_name] = method_a_df
                if method_b_df is not None and method_b_name not in de_results_for_heatmaps:
                    de_results_for_heatmaps[method_b_name] = method_b_df
                
                if method_a_df is None or method_b_df is None:
                    print(f"Skipping comparison {method_a_name} vs {method_b_name}: could not load results")
                    continue
                
                metrics = compute_de_comparison_metrics(method_a_df, method_b_df)
                acc = {"comparison": f"{method_a_name} vs {method_b_name}"}
                acc.update(metrics)
                accuracy_results.append(acc)
            
            elif comp_type == "de_raw":
                method_a_df = _load_de_result(method_a_name, method_a_path_val, use_raw_lfc=True)
                method_b_df = _load_de_result(method_b_name, method_b_path_val, use_raw_lfc=True)
                
                if method_a_df is None or method_b_df is None:
                    print(f"Skipping comparison {method_a_name} vs {method_b_name}: could not load results")
                    continue
                
                metrics = compute_de_comparison_metrics(method_a_df, method_b_df)
                acc = {"comparison": f"{method_a_name} vs {method_b_name}"}
                acc.update(metrics)
                accuracy_results.append(acc)
            
            elif comp_type == "de_lfcshrink":
                method_a_df = _load_de_result(method_a_name, method_a_path_val, use_raw_lfc=False)
                method_b_df = _load_de_result(method_b_name, method_b_path_val, use_raw_lfc=False)
                
                if method_a_df is None or method_b_df is None:
                    print(f"Skipping comparison {method_a_name} vs {method_b_name} (lfcShrink): could not load results")
                    continue
                
                metrics = compute_de_comparison_metrics(method_a_df, method_b_df)
                acc = {"comparison": f"{method_a_name} vs {method_b_name} (lfcShrink)"}
                acc.update(metrics)
                accuracy_results.append(acc)
                
        except Exception as e:
            print(f"Error comparing {method_a_name} vs {method_b_name}: {e}")
    
    # Generate overlap heatmaps
    overlap_heatmaps: Dict[str, Path] = {}
    if de_results_for_heatmaps:
        try:
            from .visualization import generate_overlap_heatmaps
            overlap_heatmaps = generate_overlap_heatmaps(
                de_results_for_heatmaps,
                output_dir,
                k_values=(50, 100, 500),
            )
        except Exception as e:
            print(f"Warning: Could not generate overlap heatmaps: {e}")
            
    # Save tables with benchmark_ prefix
    perf_df.to_csv(output_dir / "benchmark_performance.csv", index=False)
    if accuracy_results:
        acc_df = pd.DataFrame(accuracy_results)
        acc_df.to_csv(output_dir / "benchmark_accuracy.csv", index=False)
        
    # Generate improved Markdown
    md = _generate_improved_markdown(
        perf_df, 
        perf_comp_results, 
        accuracy_results,
        overlap_heatmaps,
    )
        
    with open(output_dir / "benchmark_report.md", "w") as f:
        f.write(md)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    """CLI entry point for regenerating benchmark reports."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark reports from cached results."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory containing .benchmark_cache with cached results",
    )
    
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        print(f"Error: Directory {args.output_dir} does not exist")
        return
    
    cache_dir = args.output_dir / ".benchmark_cache"
    if not cache_dir.exists():
        print(f"Error: No .benchmark_cache directory found in {args.output_dir}")
        return
    
    print(f"Generating benchmark report for {args.output_dir}")
    evaluate_benchmarks(args.output_dir)
    print(f"Report generated: {args.output_dir / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
