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
    "pvalue_stream_auroc",
    "pvalue_reference_auroc",
)

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
    stream: pd.Series,
    reference: pd.Series,
    *,
    top_k: int,
    use_absolute: bool,
    ascending: bool,
) -> Optional[float]:
    frame = pd.DataFrame({"stream": stream, "reference": reference}).dropna()
    if frame.empty:
        return None
    k = min(int(top_k), len(frame))
    if k <= 0:
        return None

    if use_absolute:
        stream_rank = frame["stream"].abs().sort_values(ascending=False).index[:k]
        ref_rank = frame["reference"].abs().sort_values(ascending=False).index[:k]
    else:
        stream_rank = frame["stream"].sort_values(ascending=ascending).index[:k]
        ref_rank = frame["reference"].sort_values(ascending=ascending).index[:k]

    if not stream_rank.size or not ref_rank.size:
        return None

    overlap = len(set(stream_rank) & set(ref_rank))
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
    streaming: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> Dict[str, Optional[float]]:
    """Return a dictionary of metrics comparing two differential expression tables."""

    metrics: Dict[str, Optional[float]] = {key: None for key in DE_METRIC_KEYS}

    if streaming is None or reference is None or streaming.empty or reference.empty:
        return metrics

    merged = streaming.merge(
        reference,
        on=["perturbation", "gene"],
        suffixes=("_stream", "_reference"),
        how="inner",
    )
    if merged.empty:
        return metrics

    merged = merged.copy()
    pair_index = merged["perturbation"].astype(str) + "||" + merged["gene"].astype(str)
    merged.index = pair_index

    column_settings = {
        "effect": {
            "stream": "effect_size_stream",
            "reference": "effect_size_reference",
            "use_absolute": True,
            "ascending": False,
        },
        "statistic": {
            "stream": "statistic_stream",
            "reference": "statistic_reference",
            "use_absolute": True,
            "ascending": False,
        },
        "pvalue": {
            "stream": "pvalue_stream",
            "reference": "pvalue_reference",
            "use_absolute": False,
            "ascending": True,
        },
    }

    for name, settings in column_settings.items():
        stream_col = settings["stream"]
        ref_col = settings["reference"]
        if stream_col not in merged.columns or ref_col not in merged.columns:
            continue
        stream_series = pd.to_numeric(merged[stream_col], errors="coerce")
        ref_series = pd.to_numeric(merged[ref_col], errors="coerce")
        valid = stream_series.notna() & ref_series.notna()
        if valid.sum() == 0:
            continue
        diff = (stream_series[valid] - ref_series[valid]).abs()
        if not diff.empty:
            metrics[f"{name}_max_abs_diff"] = float(diff.max())
        metrics[f"{name}_pearson_corr"] = _safe_corr(stream_series, ref_series, "pearson")
        metrics[f"{name}_spearman_corr"] = _safe_corr(stream_series, ref_series, "spearman")
        metrics[f"{name}_top_k_overlap"] = _top_k_overlap(
            stream_series,
            ref_series,
            top_k=top_k,
            use_absolute=settings["use_absolute"],
            ascending=settings["ascending"],
        )

    label_series = _find_label_series(merged)
    if label_series is not None:
        label_series = label_series.astype(float)
        if "pvalue_stream" in merged.columns:
            stream_scores = _prepare_auc_scores(merged["pvalue_stream"])
            if stream_scores is not None:
                labels = label_series.dropna()
                valid_idx = labels.index.intersection(stream_scores.index)
                if len(valid_idx) >= 2:
                    y_true = labels.loc[valid_idx].astype(int).to_numpy()
                    if np.unique(y_true).size == 2:
                        y_score = stream_scores.loc[valid_idx].to_numpy()
                        metrics["pvalue_stream_auroc"] = _compute_binary_roc_auc(
                            y_true, y_score
                        )
        if "pvalue_reference" in merged.columns:
            reference_scores = _prepare_auc_scores(merged["pvalue_reference"])
            if reference_scores is not None:
                labels = label_series.dropna()
                valid_idx = labels.index.intersection(reference_scores.index)
                if len(valid_idx) >= 2:
                    y_true = labels.loc[valid_idx].astype(int).to_numpy()
                    if np.unique(y_true).size == 2:
                        y_score = reference_scores.loc[valid_idx].to_numpy()
                        metrics["pvalue_reference_auroc"] = _compute_binary_roc_auc(
                            y_true, y_score
                        )

    return metrics

