"""Utility functions for comparing differential expression results."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DE_METRIC_KEYS: tuple[str, ...] = (
    "effect_max_abs_diff",
    "effect_pearson_corr_mean",
    "effect_pearson_corr_std",
    "effect_spearman_corr_mean",
    "effect_spearman_corr_std",
    "effect_top_k_overlap",
    "effect_top_100_overlap_mean",
    "effect_top_100_overlap_std",
    "effect_top_500_overlap_mean",
    "effect_top_500_overlap_std",
    "statistic_max_abs_diff",
    "statistic_pearson_corr_mean",
    "statistic_pearson_corr_std",
    "statistic_spearman_corr_mean",
    "statistic_spearman_corr_std",
    "statistic_top_k_overlap",
    "pvalue_max_abs_diff",
    "pvalue_log_pearson_corr_mean",
    "pvalue_log_pearson_corr_std",
    "pvalue_log_spearman_corr_mean",
    "pvalue_log_spearman_corr_std",
    "pvalue_top_k_overlap",
    "pvalue_top_100_overlap_mean",
    "pvalue_top_100_overlap_std",
    "pvalue_top_500_overlap_mean",
    "pvalue_top_500_overlap_std",
    "pvalue_method_a_auroc",
    "pvalue_method_b_auroc",
    "statistic_type_mismatch",
    "n_perturbations",
)

# Top-k values for overlap metrics
TOP_K_VALUES: tuple[int, ...] = (50, 100, 500)

# Threshold for detecting F-statistics (which are always positive)
# If all reference statistics are positive and the ratio of max to min
# exceeds this threshold, we consider it likely to be F-statistics
_F_STATISTIC_DETECTION_RATIO = 100.0

_DEFAULT_TOP_K = 50
_LABEL_COLUMN_CANDIDATES: tuple[str, ...] = (
    "is_hit",
    "hit",
    "hits",
    "label",
    "labels",
    "truth",
    "is_target",
    "target",
    "targets",
    "positive",
    "is_positive",
    "ground_truth",
)
_POSITIVE_LABEL_VALUES: set[str] = {
    "true",
    "t",
    "yes",
    "y",
    "1",
    "positive",
    "pos",
    "hit",
    "hits",
    "target",
    "targets",
    "perturbed",
    "responsive",
}


def _is_likely_f_statistic(series: pd.Series) -> bool:
    """Detect if a statistic series is likely F-statistics (always non-negative).
    
    F-statistics from edgeR's quasi-likelihood F-test are always non-negative,
    unlike Wald statistics (coef/SE) which can be positive or negative.
    Direct comparison of these is meaningless for raw correlation.
    
    Returns True if the series appears to contain F-statistics.
    """
    valid = series.dropna()
    if len(valid) < 10:
        return False
    
    # F-statistics are always non-negative (can be 0)
    if (valid < 0).any():
        return False
    
    # Wald statistics are symmetric around 0, F-statistics are right-skewed
    # Check if all values are non-negative AND there's right skew
    mean_val = valid.mean()
    median_val = valid.median()
    
    # F-statistics: mean > median (right skew), all non-negative
    # Wald statistics: mean ≈ median (symmetric around 0)
    
    # Key check: Wald statistics will have negative values
    # F-statistics will not. Since we already checked for negatives above,
    # we check if the distribution looks like it could have come from 
    # absolute values of a symmetric distribution
    
    # Check the ratio of values above vs below the mean
    # For symmetric Wald: roughly 50% above mean
    # For F-stat: more concentrated toward lower values
    above_mean_ratio = (valid > mean_val).mean()
    
    # F-statistics are chi-squared-like: many small values, few large ones
    # So above_mean_ratio should be < 0.5 (typically 0.3-0.4)
    if above_mean_ratio < 0.45:
        return True
    
    return False


def _safe_corr(series_a: pd.Series, series_b: pd.Series, method: str) -> Optional[float]:
    valid = series_a.notna() & series_b.notna()
    if valid.sum() < 2:
        return None
    try:
        corr = series_a[valid].corr(series_b[valid], method=method)
    except Exception:  # pragma: no cover - defensive guard
        return None
    return float(corr) if pd.notna(corr) else None


def _compute_per_perturbation_corr(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    method: str,
    transform_log: bool = False,
) -> tuple[Optional[float], Optional[float], int]:
    """Compute correlation per perturbation and return mean, std, and count.
    
    Parameters
    ----------
    merged : pd.DataFrame
        Merged DataFrame with perturbation column and value columns
    col_a : str
        Column name for method A values
    col_b : str
        Column name for method B values
    method : str
        Correlation method ('pearson' or 'spearman')
    transform_log : bool
        If True, apply -log10 transformation before computing correlation
        
    Returns
    -------
    tuple[Optional[float], Optional[float], int]
        (mean_correlation, std_correlation, n_perturbations)
        Returns (None, None, 0) if no valid correlations can be computed
    """
    if col_a not in merged.columns or col_b not in merged.columns:
        return None, None, 0
    
    correlations = []
    
    for pert, group in merged.groupby("perturbation"):
        series_a = pd.to_numeric(group[col_a], errors="coerce")
        series_b = pd.to_numeric(group[col_b], errors="coerce")
        
        if transform_log:
            series_a = -np.log10(np.clip(series_a.to_numpy(), 1e-300, 1))
            series_b = -np.log10(np.clip(series_b.to_numpy(), 1e-300, 1))
            series_a = pd.Series(series_a, index=group.index)
            series_b = pd.Series(series_b, index=group.index)
        
        corr = _safe_corr(series_a, series_b, method)
        if corr is not None:
            correlations.append(corr)
    
    if not correlations:
        return None, None, 0
    
    mean_corr = float(np.mean(correlations))
    std_corr = float(np.std(correlations, ddof=1)) if len(correlations) > 1 else 0.0
    
    return mean_corr, std_corr, len(correlations)


def _top_k_overlap(
    method_a: pd.Series,
    method_b: pd.Series,
    *,
    top_k: int,
    use_absolute: bool,
    ascending: bool,
) -> Optional[float]:
    frame = pd.DataFrame({"method_a": method_a, "method_b": method_b}).dropna()
    if frame.empty:
        return None
    k = min(int(top_k), len(frame))
    if k <= 0:
        return None

    if use_absolute:
        method_a_rank = frame["method_a"].abs().sort_values(ascending=False).index[:k]
        method_b_rank = frame["method_b"].abs().sort_values(ascending=False).index[:k]
    else:
        method_a_rank = frame["method_a"].sort_values(ascending=ascending).index[:k]
        method_b_rank = frame["method_b"].sort_values(ascending=ascending).index[:k]

    if not method_a_rank.size or not method_b_rank.size:
        return None

    overlap = len(set(method_a_rank) & set(method_b_rank))
    return float(overlap) / float(k)


def _compute_per_perturbation_overlap(
    merged: pd.DataFrame,
    col_a: str,
    col_b: str,
    top_k: int,
    use_absolute: bool,
    ascending: bool,
) -> tuple[Optional[float], Optional[float], int]:
    """Compute top-k overlap per perturbation and return mean, std, and count.
    
    This function computes the overlap between top-k genes from two methods
    for each perturbation separately, then aggregates into mean ± std.
    
    Parameters
    ----------
    merged : pd.DataFrame
        Merged DataFrame with perturbation column and value columns
    col_a : str
        Column name for method A values
    col_b : str
        Column name for method B values
    top_k : int
        Number of top genes to consider. Will be capped at min(top_k, n_genes)
        for each perturbation.
    use_absolute : bool
        If True, rank by absolute values (for effect sizes)
    ascending : bool
        If True, sort ascending (for p-values); if False, sort descending
        
    Returns
    -------
    tuple[Optional[float], Optional[float], int]
        (mean_overlap, std_overlap, n_perturbations)
        Returns (None, None, 0) if no valid overlaps can be computed
    """
    if col_a not in merged.columns or col_b not in merged.columns:
        return None, None, 0
    
    overlaps = []
    
    for pert, group in merged.groupby("perturbation"):
        series_a = pd.to_numeric(group[col_a], errors="coerce")
        series_b = pd.to_numeric(group[col_b], errors="coerce")
        
        overlap = _top_k_overlap(
            series_a,
            series_b,
            top_k=top_k,
            use_absolute=use_absolute,
            ascending=ascending,
        )
        if overlap is not None:
            overlaps.append(overlap)
    
    if not overlaps:
        return None, None, 0
    
    mean_overlap = float(np.mean(overlaps))
    std_overlap = float(np.std(overlaps, ddof=1)) if len(overlaps) > 1 else 0.0
    
    return mean_overlap, std_overlap, len(overlaps)


def compute_pairwise_overlap_matrix(
    de_results: Dict[str, pd.DataFrame],
    top_k: int = 100,
    metric: str = "effect",
) -> tuple[pd.DataFrame, int]:
    """Compute pairwise top-k overlap matrix between all DE methods.
    
    For each pair of methods, computes the mean top-k overlap across all
    shared perturbations. The matrix is symmetric with 1.0 on the diagonal.
    
    Parameters
    ----------
    de_results : Dict[str, pd.DataFrame]
        Dictionary mapping method names to their DE result DataFrames.
        Each DataFrame should have columns: perturbation, gene, effect_size, pvalue
    top_k : int
        Number of top genes to consider for overlap (default: 100)
    metric : str
        Which metric to use for ranking: "effect" (by |effect_size|) or
        "pvalue" (by ascending p-value). Default: "effect"
        
    Returns
    -------
    tuple[pd.DataFrame, int]
        (overlap_matrix, effective_k) where:
        - overlap_matrix: DataFrame with method names as index and columns,
          containing mean overlap values. NaN for missing/incompatible pairs.
        - effective_k: The actual k used (may be less than top_k if fewer genes)
    """
    method_names = list(de_results.keys())
    n_methods = len(method_names)
    
    # Initialize matrix with NaN
    matrix = pd.DataFrame(
        np.full((n_methods, n_methods), np.nan),
        index=method_names,
        columns=method_names,
    )
    
    # Set diagonal to 1.0 (perfect self-overlap)
    for method in method_names:
        matrix.loc[method, method] = 1.0
    
    # Track effective k (minimum across all comparisons)
    effective_k = top_k
    
    # Configure ranking based on metric type
    if metric == "effect":
        col_name = "effect_size"
        use_absolute = True
        ascending = False
    else:  # pvalue
        col_name = "pvalue"
        use_absolute = False
        ascending = True
    
    # Compute pairwise overlaps
    for i, method_a in enumerate(method_names):
        for j, method_b in enumerate(method_names):
            if i >= j:  # Skip diagonal and lower triangle (will mirror)
                continue
            
            df_a = de_results[method_a]
            df_b = de_results[method_b]
            
            if df_a is None or df_b is None or df_a.empty or df_b.empty:
                continue
            
            # Standardize column names if needed
            if col_name not in df_a.columns or col_name not in df_b.columns:
                continue
            
            # Merge on perturbation + gene (compute on gene intersection)
            merged = df_a.merge(
                df_b,
                on=["perturbation", "gene"],
                suffixes=("_a", "_b"),
                how="inner",
            )
            
            if merged.empty:
                continue
            
            col_a = f"{col_name}_a"
            col_b = f"{col_name}_b"
            
            mean_overlap, _, n_pert = _compute_per_perturbation_overlap(
                merged, col_a, col_b, top_k, use_absolute, ascending
            )
            
            if mean_overlap is not None:
                matrix.loc[method_a, method_b] = mean_overlap
                matrix.loc[method_b, method_a] = mean_overlap
                
                # Track minimum genes per perturbation for effective k
                for pert, group in merged.groupby("perturbation"):
                    n_genes = len(group)
                    effective_k = min(effective_k, n_genes, top_k)
    
    return matrix, effective_k


def _normalise_labels(series: pd.Series) -> Optional[pd.Series]:
    if series.empty:
        return None

    data = series.dropna()
    if data.empty:
        return None

    if data.dtype == bool:
        mapped = data.astype(int)
    else:
        numeric = pd.to_numeric(data, errors="coerce")
        numeric_unique = {float(value) for value in numeric.dropna().unique().tolist()}
        if numeric.notna().sum() >= 2 and numeric_unique <= {0.0, 1.0} and len(numeric_unique) == 2:
            mapped = numeric.astype(int)
        else:
            lowered = data.astype(str).str.lower()
            positive_mask = lowered.isin(_POSITIVE_LABEL_VALUES)
            if not positive_mask.any() or positive_mask.all():
                return None
            mapped = positive_mask.astype(int)

    if mapped.nunique(dropna=True) < 2:
        return None

    output = pd.Series(np.nan, index=series.index, dtype=float)
    output.loc[mapped.index] = mapped.astype(float)
    return output


def _find_label_series(frame: pd.DataFrame) -> Optional[pd.Series]:
    for column in frame.columns:
        base = column.replace("_stream", "").replace("_reference", "")
        if base in _LABEL_COLUMN_CANDIDATES:
            labels = _normalise_labels(frame[column])
            if labels is not None:
                return labels
    return None


def _prepare_auc_scores(series: pd.Series) -> Optional[pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    valid = numeric.notna()
    if valid.sum() < 2:
        return None
    clipped = np.clip(numeric[valid].to_numpy(dtype=float), 1e-300, None)
    transformed = -np.log10(clipped)
    return pd.Series(transformed, index=numeric.index[valid], dtype=float)


def _compute_binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    sum_ranks_pos = float(ranks[positives].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_de_comparison_metrics(
    method_a: pd.DataFrame,
    method_b: pd.DataFrame,
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> Dict[str, Optional[float]]:
    """Return a dictionary of metrics comparing two differential expression tables.
    
    Computes correlation, overlap, and statistical comparison metrics between two
    differential expression result DataFrames. Both inputs must contain columns:
    'perturbation', 'gene', 'effect_size', 'statistic', and 'pvalue'.
    
    Label-based metrics (pvalue_method_a_auroc, pvalue_method_b_auroc) require
    ground truth labels in a column such as 'is_hit', 'label', or 'ground_truth'.
    These AUROC metrics are typically only available in benchmark/validation
    scenarios where true positive perturbations are known.
    
    Args:
        method_a: DataFrame with first method's DE results
        method_b: DataFrame with second method's DE results
        top_k: Number of top hits to consider for overlap metrics (default: 50)
    
    Returns:
        Dictionary with 14 metric keys defined in DE_METRIC_KEYS. Metrics include:
        - Correlation (Pearson, Spearman) for effect sizes, statistics, and p-values
        - Top-k overlap for ranking concordance
        - Max absolute differences
        - AUROC for p-values (only when ground truth labels are present)
    """

    metrics: Dict[str, Optional[float]] = {key: None for key in DE_METRIC_KEYS}

    if method_a is None or method_b is None or method_a.empty or method_b.empty:
        return metrics

    merged = method_a.merge(
        method_b,
        on=["perturbation", "gene"],
        suffixes=("_method_a", "_method_b"),
        how="inner",
    )
    if merged.empty:
        return metrics

    merged = merged.copy()
    pair_index = merged["perturbation"].astype(str) + "||" + merged["gene"].astype(str)
    merged.index = pair_index

    column_settings = {
        "effect": {
            "method_a": "effect_size_method_a",
            "method_b": "effect_size_method_b",
            "use_absolute": True,
            "ascending": False,
        },
        "statistic": {
            "method_a": "statistic_method_a",
            "method_b": "statistic_method_b",
            "use_absolute": True,
            "ascending": False,
        },
        "pvalue": {
            "method_a": "pvalue_method_a",
            "method_b": "pvalue_method_b",
            "use_absolute": False,
            "ascending": True,
        },
    }

    # Detect if we're comparing Wald statistics (signed) vs F-statistics (unsigned)
    statistic_type_mismatch = False
    if "statistic_method_a" in merged.columns and "statistic_method_b" in merged.columns:
        method_a_stat = pd.to_numeric(merged["statistic_method_a"], errors="coerce")
        method_b_stat = pd.to_numeric(merged["statistic_method_b"], errors="coerce")
        method_a_is_f = _is_likely_f_statistic(method_a_stat)
        method_b_is_f = _is_likely_f_statistic(method_b_stat)
        if method_a_is_f != method_b_is_f:
            statistic_type_mismatch = True
            metrics["statistic_type_mismatch"] = 1.0

    for name, settings in column_settings.items():
        method_a_col = settings["method_a"]
        method_b_col = settings["method_b"]
        if method_a_col not in merged.columns or method_b_col not in merged.columns:
            continue
        method_a_series = pd.to_numeric(merged[method_a_col], errors="coerce")
        method_b_series = pd.to_numeric(merged[method_b_col], errors="coerce")
        valid = method_a_series.notna() & method_b_series.notna()
        if valid.sum() == 0:
            continue
        diff = (method_a_series[valid] - method_b_series[valid]).abs()
        if not diff.empty:
            metrics[f"{name}_max_abs_diff"] = float(diff.max())
        
        # For statistics, if there's a type mismatch (Wald vs F), skip raw correlation
        # but still compute overlap using absolute values
        if name == "statistic" and statistic_type_mismatch:
            # Use absolute values for correlation when comparing Wald vs F
            # Create a copy with absolute values for per-perturbation correlation
            merged_abs = merged.copy()
            merged_abs[method_a_col] = pd.to_numeric(merged[method_a_col], errors="coerce").abs()
            merged_abs[method_b_col] = pd.to_numeric(merged[method_b_col], errors="coerce").abs()
            
            mean_p, std_p, n_pert = _compute_per_perturbation_corr(merged_abs, method_a_col, method_b_col, "pearson")
            metrics[f"{name}_pearson_corr_mean"] = mean_p
            metrics[f"{name}_pearson_corr_std"] = std_p
            
            mean_s, std_s, _ = _compute_per_perturbation_corr(merged_abs, method_a_col, method_b_col, "spearman")
            metrics[f"{name}_spearman_corr_mean"] = mean_s
            metrics[f"{name}_spearman_corr_std"] = std_s
        elif name == "pvalue":
            # For p-values, compute correlations on -log10 transformed values
            mean_p, std_p, n_pert = _compute_per_perturbation_corr(
                merged, method_a_col, method_b_col, "pearson", transform_log=True
            )
            metrics["pvalue_log_pearson_corr_mean"] = mean_p
            metrics["pvalue_log_pearson_corr_std"] = std_p
            
            mean_s, std_s, _ = _compute_per_perturbation_corr(
                merged, method_a_col, method_b_col, "spearman", transform_log=True
            )
            metrics["pvalue_log_spearman_corr_mean"] = mean_s
            metrics["pvalue_log_spearman_corr_std"] = std_s
            metrics["n_perturbations"] = float(n_pert)
        else:
            mean_p, std_p, n_pert = _compute_per_perturbation_corr(merged, method_a_col, method_b_col, "pearson")
            metrics[f"{name}_pearson_corr_mean"] = mean_p
            metrics[f"{name}_pearson_corr_std"] = std_p
            
            mean_s, std_s, _ = _compute_per_perturbation_corr(merged, method_a_col, method_b_col, "spearman")
            metrics[f"{name}_spearman_corr_mean"] = mean_s
            metrics[f"{name}_spearman_corr_std"] = std_s
            
            if name == "effect":
                metrics["n_perturbations"] = float(n_pert)
        
        metrics[f"{name}_top_k_overlap"] = _top_k_overlap(
            method_a_series,
            method_b_series,
            top_k=top_k,
            use_absolute=settings["use_absolute"],
            ascending=settings["ascending"],
        )
        
        # Compute per-perturbation overlap for additional k values (effect and pvalue only)
        if name in ("effect", "pvalue"):
            for k in (100, 500):
                mean_ovl, std_ovl, _ = _compute_per_perturbation_overlap(
                    merged,
                    method_a_col,
                    method_b_col,
                    top_k=k,
                    use_absolute=settings["use_absolute"],
                    ascending=settings["ascending"],
                )
                metrics[f"{name}_top_{k}_overlap_mean"] = mean_ovl
                metrics[f"{name}_top_{k}_overlap_std"] = std_ovl

    label_series = _find_label_series(merged)
    if label_series is not None:
        label_series = label_series.astype(float)
        if "pvalue_method_a" in merged.columns:
            method_a_scores = _prepare_auc_scores(merged["pvalue_method_a"])
            if method_a_scores is not None:
                labels = label_series.dropna()
                valid_idx = labels.index.intersection(method_a_scores.index)
                if len(valid_idx) >= 2:
                    y_true = labels.loc[valid_idx].astype(int).to_numpy()
                    if np.unique(y_true).size == 2:
                        y_score = method_a_scores.loc[valid_idx].to_numpy()
                        metrics["pvalue_method_a_auroc"] = _compute_binary_roc_auc(
                            y_true, y_score
                        )
        if "pvalue_method_b" in merged.columns:
            method_b_scores = _prepare_auc_scores(merged["pvalue_method_b"])
            if method_b_scores is not None:
                labels = label_series.dropna()
                valid_idx = labels.index.intersection(method_b_scores.index)
                if len(valid_idx) >= 2:
                    y_true = labels.loc[valid_idx].astype(int).to_numpy()
                    if np.unique(y_true).size == 2:
                        y_score = method_b_scores.loc[valid_idx].to_numpy()
                        metrics["pvalue_method_b_auroc"] = _compute_binary_roc_auc(
                            y_true, y_score
                        )

    return metrics

