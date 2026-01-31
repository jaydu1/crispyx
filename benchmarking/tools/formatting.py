"""Formatting utilities for benchmark reports.

This module provides formatting functions for method names, performance
metrics, and markdown table generation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .constants import (
    METHOD_DISPLAY_NAMES,
    DE_METHOD_DISPLAY_ORDER,
    CATEGORY_DISPLAY_ORDER,
    SHRINKAGE_METADATA,
    STANDARD_DE_COLUMNS,
)


# ============================================================================
# NA Checking Helpers
# ============================================================================

def is_scalar_na(value: Any) -> bool:
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


def is_scalar_notna(value: Any) -> bool:
    """Check if a value is not NA/NaN, handling arrays properly.
    
    Returns True if value is not None and is not a scalar NA.
    Also returns True for arrays (they're considered "not NA" even if they contain NAs).
    """
    return not is_scalar_na(value)


# ============================================================================
# Method Name Formatting
# ============================================================================

def format_method_name(method: str) -> str:
    """Format method name for display with proper capitalization.
    
    Uses centralized METHOD_DISPLAY_NAMES mapping for consistency.
    Note: shrinkage status is indicated via the lfcShrink column, not in method name.
    """
    # Strip _shrunk suffix if present (shrinkage shown separately)
    base_method = method.replace("_shrunk", "")
    
    # Use centralized mapping if available
    if base_method in METHOD_DISPLAY_NAMES:
        return METHOD_DISPLAY_NAMES[base_method]
    
    # Fallback for unmapped methods
    name = method.replace("crispyx_", "").replace("scanpy_", "").replace("pertpy_", "").replace("edger_", "")
    name = name.replace("_", " ")
    
    # Order matters: match longer patterns first to avoid partial matches
    replacements = {
        "de t test": "t-test",
        "de wilcoxon": "Wilcoxon",
        "de nb glm sf per": "NB-GLM sf=per",
        "de nb glm joint shrunk": "NB-GLM (joint)",
        "de nb glm joint": "NB-GLM (joint)",
        "de nb glm shrunk": "NB-GLM",
        "de nb glm": "NB-GLM",
        "de glm": "NB-GLM",
        "de pydeseq2 shrunk": "NB-GLM",
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


def get_method_package(method: str) -> str:
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


def format_full_method_name(method: str) -> str:
    """Format full method name including package."""
    package = get_method_package(method)
    name = format_method_name(method)
    if package:
        return f"{package} {name}"
    return name


def is_crispyx_method(method: str) -> bool:
    """Check if method is a crispyx method."""
    return method.startswith("crispyx_")


def get_shrinkage_type(method: str) -> str:
    """Get the shrinkage type for a method, or empty string if none."""
    # Strip any suffix like " (lfcShrink)" that may be added in comparison strings
    clean_method = method.split(" (")[0] if " (" in method else method
    return SHRINKAGE_METADATA.get(clean_method, "")


def format_heatmap_method_name(name: str) -> str:
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
    elif name == "crispyx_de_nb_glm_shrunk":
        return "crispyx NB-GLM (lfcShrink)"
    elif name == "crispyx_de_nb_glm_pydeseq2_shrunk":
        return "crispyx NB-GLM pydeseq2 (lfcShrink)"
    # Handle base method names
    elif name == "pertpy_de_pydeseq2":
        return "PyDESeq2"
    elif name == "crispyx_de_nb_glm":
        return "crispyx NB-GLM"
    elif name == "crispyx_de_nb_glm_pydeseq2":
        return "crispyx NB-GLM pydeseq2"
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
    # Handle new standalone lfcshrink methods
    elif name == "crispyx_de_lfcshrink":
        return "crispyx lfcShrink"
    elif name == "crispyx_de_lfcshrink_pydeseq2":
        return "crispyx lfcShrink pydeseq2"
    elif name == "pertpy_de_lfcshrink":
        return "PyDESeq2 lfcShrink"
    
    # Generic formatting for other methods
    display_name = name.replace("crispyx_", "crispyx ").replace("scanpy_", "scanpy ")
    display_name = display_name.replace("pertpy_", "pertpy ").replace("edger_", "edgeR ")
    display_name = display_name.replace("de_", "").replace("_", " ")
    
    return display_name


# ============================================================================
# Sort Key Functions
# ============================================================================

def get_method_sort_key(method_name: str) -> int:
    """Get sort key for a method name based on DE_METHOD_DISPLAY_ORDER."""
    for i, order_name in enumerate(DE_METHOD_DISPLAY_ORDER):
        if order_name in method_name:
            return i
    return len(DE_METHOD_DISPLAY_ORDER)  # Unknown methods go last


def get_category_sort_key(category: str) -> int:
    """Get sort key for a category based on CATEGORY_DISPLAY_ORDER."""
    try:
        return CATEGORY_DISPLAY_ORDER.index(category)
    except ValueError:
        return len(CATEGORY_DISPLAY_ORDER)  # Unknown categories go last


def get_method_category(method_name: str) -> tuple[str, str, int]:
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


# ============================================================================
# Performance / Accuracy Formatting
# ============================================================================

def get_performance_emoji(pct: Optional[float], is_lower_better: bool = True) -> str:
    """Return emoji indicator for performance comparison."""
    if pct is None or is_scalar_na(pct):
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


def get_accuracy_emoji(corr: Optional[float]) -> str:
    """Return emoji indicator for accuracy/correlation."""
    if corr is None or is_scalar_na(corr):
        return ""
    
    if corr >= 0.95:
        return "✅"
    elif corr >= 0.8:
        return "⚠️"
    else:
        return "❌"


def format_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    """Format mean ± std as a two-line string for markdown tables."""
    if mean is None or is_scalar_na(mean):
        return "-"
    
    if std is None or is_scalar_na(std) or std == 0:
        return f"{mean:.3f}"
    
    return f"{mean:.3f}<br><small>±{std:.3f}</small>"


def format_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format percentage value."""
    if value is None or is_scalar_na(value):
        return "-"
    return f"{value:.{decimals}f}%"


def format_diff(value: Optional[float], unit: str = "s", decimals: int = 1) -> str:
    """Format difference value with sign."""
    if value is None or is_scalar_na(value):
        return "-"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}{unit}"


# ============================================================================
# Markdown Table Generation
# ============================================================================

def frame_to_markdown_table(table: pd.DataFrame) -> str:
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
            if is_scalar_na(value):
                values.append("")
            else:
                values.append(str(value))
        data_rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header_row, separator_row, *data_rows]) + "\n"


# ============================================================================
# DataFrame Standardization
# ============================================================================

def standardise_de_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return ``df`` with standard differential expression column names."""
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_DE_COLUMNS)

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

    for column in STANDARD_DE_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result = result[STANDARD_DE_COLUMNS]
    result["perturbation"] = result["perturbation"].astype(str).str.strip()
    result["gene"] = result["gene"].astype(str).str.strip()
    result["effect_size"] = pd.to_numeric(result["effect_size"], errors="coerce")
    result["statistic"] = pd.to_numeric(result["statistic"], errors="coerce")
    result["pvalue"] = pd.to_numeric(result["pvalue"], errors="coerce")
    return result
