"""Utility functions for comparing differential expression results."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DE_METRIC_KEYS: tuple[str, ...] = (
    "effect_max_abs_diff",
    "effect_pearson_corr",
    "effect_spearman_corr",
    "effect_top_k_overlap",
    "statistic_max_abs_diff",
    "statistic_pearson_corr",
    "statistic_spearman_corr",
    "statistic_top_k_overlap",
    "pvalue_max_abs_diff",
    "pvalue_pearson_corr",
    "pvalue_spearman_corr",
    "pvalue_top_k_overlap",
    "pvalue_method_a_auroc",
    "pvalue_method_b_auroc",
    "statistic_type_mismatch",
)

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
            method_a_abs = method_a_series.abs()
            method_b_abs = method_b_series.abs()
            metrics[f"{name}_pearson_corr"] = _safe_corr(method_a_abs, method_b_abs, "pearson")
            metrics[f"{name}_spearman_corr"] = _safe_corr(method_a_abs, method_b_abs, "spearman")
        else:
            metrics[f"{name}_pearson_corr"] = _safe_corr(method_a_series, method_b_series, "pearson")
            metrics[f"{name}_spearman_corr"] = _safe_corr(method_a_series, method_b_series, "spearman")
        
        metrics[f"{name}_top_k_overlap"] = _top_k_overlap(
            method_a_series,
            method_b_series,
            top_k=top_k,
            use_absolute=settings["use_absolute"],
            ascending=settings["ascending"],
        )

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

