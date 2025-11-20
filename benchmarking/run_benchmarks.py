"""Utilities to benchmark streaming CRISPR screen analysis methods."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import resource
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# Ensure the local package is importable when the project has not been installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.generate_demo_dataset import write_demo_dataset
from crispyx.data import (
    read_backed,
    resolve_control_label,
    calculate_adaptive_qc_thresholds,
    standardize_dataset,
)
from crispyx.de import wald_test, wilcoxon_test
from crispyx.metrics import DE_METRIC_KEYS, compute_de_comparison_metrics
from crispyx.pseudobulk import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
)
from crispyx.qc import quality_control_summary
from crispyx.scanpy_validation import ComparisonResult, compare_with_scanpy


@dataclass
class BenchmarkMethod:
    """Description of a method that should be benchmarked."""

    name: str
    description: str
    function: Callable[..., Any]
    kwargs: Dict[str, Any]
    summary: Callable[[Any, Dict[str, Any]], Dict[str, Any]]
    category: str = "core"


@dataclass
class QCParams:
    """Quality control filtering parameters."""

    min_genes: int = 5
    min_cells_per_perturbation: int = 5
    min_cells_per_gene: int = 5
    chunk_size: int = 2048


@dataclass
class ResourceLimits:
    """Resource constraints for benchmark execution."""

    time_limit: int = 300  # seconds, 0 = no limit
    memory_limit: float = 4.0  # GB, 0 = no limit


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    dataset_path: Path
    dataset_name: str
    output_dir: Path
    perturbation_column: str = "perturbation"
    control_label: Optional[str] = None
    gene_name_column: Optional[str] = "gene_symbols"
    qc_params: Optional[QCParams] = field(default_factory=QCParams)  # None = adaptive
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    methods_to_run: Optional[List[str]] = None
    show_progress: bool = True
    quiet: bool = False
    n_cores: Optional[int] = None
    force_restandardize: bool = False
    adaptive_qc_mode: str = "conservative"

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], base_output_dir: Optional[Path] = None
    ) -> "BenchmarkConfig":
        """Create config from dictionary (e.g., loaded from YAML)."""
        dataset_path = Path(data["dataset_path"])
        dataset_name = data.get("dataset_name") or dataset_path.stem

        if base_output_dir:
            output_dir = base_output_dir / dataset_name
        else:
            output_dir = Path(
                data.get("output_dir", f"benchmarking/results/{dataset_name}")
            )

        qc_data = data.get("qc_params", {})
        # Allow qc_params to be null for adaptive calculation
        if qc_data is None:
            qc_params = None
        else:
            qc_params = QCParams(
                min_genes=qc_data.get("min_genes", 5),
                min_cells_per_perturbation=qc_data.get("min_cells_per_perturbation", 5),
                min_cells_per_gene=qc_data.get("min_cells_per_gene", 5),
                chunk_size=qc_data.get("chunk_size", 2048),
            )

        limits_data = data.get("resource_limits", {})
        resource_limits = ResourceLimits(
            time_limit=limits_data.get("time_limit", 300),
            memory_limit=limits_data.get("memory_limit", 4.0),
        )

        parallel_data = data.get("parallel_config", {})
        n_cores = parallel_data.get("n_cores")

        force_restandardize = data.get("force_restandardize", False)
        adaptive_qc_mode = data.get("adaptive_qc_mode", "conservative")

        return cls(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            output_dir=output_dir,
            perturbation_column=data.get("perturbation_column", "perturbation"),
            control_label=data.get("control_label"),
            gene_name_column=data.get("gene_name_column", "gene_symbols"),
            qc_params=qc_params,
            resource_limits=resource_limits,
            methods_to_run=data.get("methods_to_run"),
            show_progress=data.get("show_progress", True),
            quiet=data.get("quiet", False),
            n_cores=n_cores,
            force_restandardize=force_restandardize,
            adaptive_qc_mode=adaptive_qc_mode,
        )

    @classmethod
    def from_yaml(
        cls, yaml_path: Path
    ) -> Union["BenchmarkConfig", List["BenchmarkConfig"]]:
        """Load configuration(s) from YAML file."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Multi-dataset mode
        if "datasets" in data:
            shared = data.get("shared_config", {})
            base_output = Path(shared.get("output_dir", "benchmarking/results"))

            configs = []
            for dataset_data in data["datasets"]:
                # Merge shared config with dataset-specific config
                merged = {**shared, **dataset_data}
                configs.append(cls.from_dict(merged, base_output))
            return configs

        # Single dataset mode
        return cls.from_dict(data)


@dataclass
class DifferentialComparisonSummary:
    """Summary statistics for comparing streaming DE results to a reference tool."""

    test_type: str
    reference_tool: str
    metrics: Dict[str, Optional[float]]
    streaming_result_path: str | None
    reference_result_path: str | None
    error: Optional[str] = None

    @property
    def effect_max_abs_diff(self) -> Optional[float]:
        return self.metrics.get("effect_max_abs_diff")

    @property
    def statistic_max_abs_diff(self) -> Optional[float]:
        return self.metrics.get("statistic_max_abs_diff")

    @property
    def pvalue_max_abs_diff(self) -> Optional[float]:
        return self.metrics.get("pvalue_max_abs_diff")


_STANDARD_DE_COLUMNS = ["perturbation", "gene", "effect_size", "statistic", "pvalue"]

_CATEGORY_ORDER = [
    "Streaming pipeline",
    "Differential expression",
    "Reference: Scanpy",
    "Reference: edgeR",
    "Reference: Pertpy",
]


_STATUS_ORDER = ["success", "memory_limit", "timeout", "error", "unknown"]


# File naming conventions for benchmark outputs
STREAMING_OUTPUT_NAMES = {
    "quality_control": "crispyx/qc_filtered.h5ad",
    "average_log_expression": "crispyx/pb_avg_log_effects.h5ad",
    "pseudobulk_expression": "crispyx/pb_pseudobulk_effects.h5ad",
    "wald_test": "crispyx/de_wald.h5ad",
    "wilcoxon_test": "crispyx/de_wilcoxon.h5ad",
}

COMPARISON_OUTPUT_PATHS = {
    "scanpy_quality_control_comparison": "scanpy/qc_comparison.json",
    "scanpy_wald_comparison": "scanpy/de_wald.h5ad",
    "scanpy_wilcoxon_comparison": "scanpy/de_wilcoxon.h5ad",
    "edger_direct_comparison": "edger/edger_wald.h5ad",
    "pertpy_pydeseq2_comparison": "pertpy/pydeseq2_wald.h5ad",
    "pertpy_statsmodels_comparison": "pertpy/statsmodels_wald.h5ad",
}


def _normalise_path(path: str | Path | None, context: Mapping[str, Any]) -> str | None:
    """Return ``path`` relative to the output directory or repository root."""

    if not path:
        return None

    path_obj = Path(path)
    candidates: Iterable[Path] = []
    output_dir = context.get("output_dir")
    if output_dir:
        candidates = [Path(str(output_dir))]
    repo_candidates = list(candidates) + [REPO_ROOT]

    for base in repo_candidates:
        try:
            return str(path_obj.resolve().relative_to(base.resolve()))
        except Exception:
            continue
    return str(path_obj)


def _percentage(part: float, total: float) -> float | None:
    """Return the percentage contribution of ``part`` to ``total``."""

    if total in (0, None):
        return None
    try:
        return (float(part) / float(total)) * 100.0
    except ZeroDivisionError:  # pragma: no cover - defensive
        return None


def _format_timing_summary(timings: Mapping[str, float]) -> str | None:
    """Return a compact human-readable summary for ``timings``."""

    if not timings:
        return None
    parts = [f"{name}={value:.3f}s" for name, value in sorted(timings.items())]
    return "; ".join(parts)


def _method_sort_key(method: BenchmarkMethod) -> tuple[int, str]:
    """Return a stable sort key that groups methods by category."""

    try:
        category_index = _CATEGORY_ORDER.index(method.category)
    except ValueError:
        category_index = len(_CATEGORY_ORDER)
    return (category_index, method.name)


def _postprocess_results(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with normalised column names, ordering, and sorting."""

    if df.empty:
        return df

    table = df.copy()

    rename_map = {
        "elapsed_seconds": "runtime_seconds",
        "max_memory_mb": "peak_memory_mb",
    }
    table = table.rename(columns={k: v for k, v in rename_map.items() if k in table.columns})

    numeric_columns = [
        "runtime_seconds",
        "peak_memory_mb",
        "cells_total",
        "cells_kept",
        "cells_removed",
        "cells_kept_pct",
        "genes_total",
        "genes_kept",
        "genes_removed",
        "genes_kept_pct",
        "rows",
        "columns",
        "groups",
        "genes",
        "stream_total_seconds",
        "reference_total_seconds",
        "stream_peak_memory_mb",
        "reference_peak_memory_mb",
    ]
    numeric_columns.extend(
        key for key in DE_METRIC_KEYS if key not in numeric_columns
    )
    for column in numeric_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    if "category" in table.columns:
        categories = table["category"].fillna("Uncategorised").astype(str)
        extra_categories = sorted({c for c in categories.unique() if c not in _CATEGORY_ORDER})
        dtype = pd.CategoricalDtype(categories=_CATEGORY_ORDER + extra_categories, ordered=True)
        table["category"] = categories.astype(dtype)
        table = table.sort_values(["category", "method"], kind="stable")
        table["category"] = table["category"].astype(str)
    else:
        table = table.sort_values(["method"], kind="stable")

    preferred_order = [
        "category",
        "method",
        "description",
        "status",
        "runtime_seconds",
        "peak_memory_mb",
        "cells_total",
        "cells_kept",
        "cells_kept_pct",
        "cells_removed",
        "genes_total",
        "genes_kept",
        "genes_kept_pct",
        "genes_removed",
        "rows",
        "columns",
        "groups",
        "genes",
        "comparison_category",
        "test_type",
        "reference_tool",
        "effect_max_abs_diff",
        "statistic_max_abs_diff",
        "pvalue_max_abs_diff",
        "effect_pearson_corr",
        "effect_spearman_corr",
        "effect_top_k_overlap",
        "statistic_pearson_corr",
        "statistic_spearman_corr",
        "statistic_top_k_overlap",
        "pvalue_pearson_corr",
        "pvalue_spearman_corr",
        "pvalue_top_k_overlap",
        "pvalue_stream_auroc",
        "pvalue_reference_auroc",
        "stream_total_seconds",
        "reference_total_seconds",
        "stream_peak_memory_mb",
        "reference_peak_memory_mb",
        "stream_timing_breakdown",
        "reference_timing_breakdown",
        "result_path",
        "streaming_result_path",
        "reference_result_path",
        "error",
    ]
    ordered_columns = [col for col in preferred_order if col in table.columns]
    remaining_columns = [col for col in table.columns if col not in ordered_columns]
    table = table[ordered_columns + remaining_columns]

    table = table.dropna(axis=1, how="all")
    return table.reset_index(drop=True)


def _compute_aggregate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Return aggregate benchmark statistics derived from ``df``."""

    summary: Dict[str, Any] = {
        "total_methods": int(len(df)),
        "status_counts": {},
        "success_count": 0,
        "timeout_count": 0,
        "memory_limit_count": 0,
        "error_count": 0,
        "non_success_count": int(len(df)),
        "success_rate": None,
        "average_runtime_seconds": None,
        "categories": [],
        "dependency_errors": [],
        "other_errors": [],
        "error_details": [],
    }

    if df.empty:
        return summary

    status_series = df.get("status")
    if status_series is not None:
        filled_status = status_series.fillna("unknown").astype(str)
    else:
        filled_status = pd.Series(["unknown"] * len(df))

    status_counts = Counter(filled_status)
    ordered_status_counts: Dict[str, int] = {}
    for status in _STATUS_ORDER:
        count = int(status_counts.get(status, 0))
        if count:
            ordered_status_counts[status] = count
    for status, count in status_counts.items():
        status = str(status)
        if status not in ordered_status_counts:
            ordered_status_counts[status] = int(count)

    summary["status_counts"] = ordered_status_counts
    summary["success_count"] = ordered_status_counts.get("success", 0)
    summary["timeout_count"] = ordered_status_counts.get("timeout", 0)
    summary["memory_limit_count"] = ordered_status_counts.get("memory_limit", 0)
    summary["error_count"] = ordered_status_counts.get("error", 0)
    summary["non_success_count"] = summary["total_methods"] - summary["success_count"]

    if summary["total_methods"]:
        summary["success_rate"] = summary["success_count"] / summary["total_methods"]

    runtime_series = df.get("runtime_seconds")
    if runtime_series is not None:
        runtimes = pd.to_numeric(runtime_series, errors="coerce").dropna()
        if not runtimes.empty:
            summary["average_runtime_seconds"] = float(runtimes.mean())

    category_summaries = []
    if "category" in df.columns:
        for category, group in df.groupby("category", sort=False):
            group_status = group.get("status")
            if group_status is not None:
                group_status_counts = Counter(group_status.fillna("unknown").astype(str))
            else:
                group_status_counts = Counter()

            ordered_group_counts: Dict[str, int] = {}
            for status in _STATUS_ORDER:
                count = int(group_status_counts.get(status, 0))
                if count:
                    ordered_group_counts[status] = count
            for status, count in group_status_counts.items():
                status = str(status)
                if status not in ordered_group_counts:
                    ordered_group_counts[status] = int(count)

            group_runtime = group.get("runtime_seconds")
            average_runtime = None
            if group_runtime is not None:
                group_runtimes = pd.to_numeric(group_runtime, errors="coerce").dropna()
                if not group_runtimes.empty:
                    average_runtime = float(group_runtimes.mean())

            category_summaries.append(
                {
                    "category": str(category),
                    "method_count": int(len(group)),
                    "status_counts": ordered_group_counts,
                    "average_runtime_seconds": average_runtime,
                }
            )

    summary["categories"] = category_summaries

    if "error" in df.columns:
        error_rows = df[df["error"].notna()]
        details = []
        for _, row in error_rows.iterrows():
            details.append(
                {
                    "method": str(row.get("method", "")),
                    "category": str(row.get("category", "")),
                    "error": str(row["error"]),
                }
            )
        summary["error_details"] = details

        dependency_keywords = ("importerror", "modulenotfounderror", "no module named")
        dependency_errors = {
            detail["error"]
            for detail in details
            if any(keyword in detail["error"].lower() for keyword in dependency_keywords)
        }
        other_errors = {
            detail["error"]
            for detail in details
            if detail["error"] not in dependency_errors
        }
        summary["dependency_errors"] = sorted(dependency_errors)
        summary["other_errors"] = sorted(other_errors)

    return summary


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
    effect_col = _resolve_column(["effect_size", "logfoldchange", "logfoldchanges", "lfc", "log_fc", "coefficient"])
    stat_col = _resolve_column(["statistic", "statistics", "score", "scores", "wald_statistic", "zscore", "t_stat", "t_value", "t_statistic", "f", "f_value", "u_stat"])
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


def _compare_de_frames(
    streaming: pd.DataFrame, reference: pd.DataFrame
) -> Dict[str, Optional[float]]:
    """Return comparison metrics between streaming and reference DE results."""

    return compute_de_comparison_metrics(streaming, reference)


def _prepare_reference_anndata(
    dataset_path: Path,
    *,
    min_genes: int,
    min_cells_per_gene: int,
    min_cells_per_perturbation: int,
    perturbation_column: str,
    control_label: str,
):
    """Return a filtered in-memory AnnData object for reference comparisons."""

    import scanpy as sc  # Imported lazily to keep runtime dependencies optional

    adata = sc.read_h5ad(str(dataset_path))
    if min_genes:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if min_cells_per_gene:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    if min_cells_per_perturbation:
        labels = adata.obs[perturbation_column].astype(str)
        counts = labels.value_counts()
        keep = labels.eq(control_label) | counts.loc[labels].ge(min_cells_per_perturbation).to_numpy()
        adata = adata[keep].copy()
    return adata


def _run_scanpy_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    min_genes: int,
    min_cells_per_gene: int,
    min_cells_per_perturbation: int,
    method: str,
    output_dir: Path,
    data_name: str,
) -> tuple[pd.DataFrame | None, Optional[Path], Optional[str]]:
    """Execute Scanpy's differential expression workflow and return a DataFrame."""

    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        return None, None, str(exc)

    adata = _prepare_reference_anndata(
        dataset_path,
        min_genes=min_genes,
        min_cells_per_gene=min_cells_per_gene,
        min_cells_per_perturbation=min_cells_per_perturbation,
        perturbation_column=perturbation_column,
        control_label=control_label,
    )

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(
        adata,
        groupby=perturbation_column,
        method=method,
        reference=control_label,
        n_genes=adata.n_vars,
    )
    df = sc.get.rank_genes_groups_df(adata, None)

    reference_path: Optional[Path] = None
    if not df.empty:
        reference_path = output_dir / f"{data_name}_{method}_scanpy_de.csv"
        df.to_csv(reference_path, index=False)
    return df, reference_path, None


def _resolve_pertpy_runner(module: Any, method: str) -> Optional[Callable[..., Any]]:
    """Best-effort resolution of a Pertpy differential expression runner."""

    candidates = [
        method,
        f"{method}_de",
        f"run_{method}",
        f"run_{method}_de",
        method.lower(),
        method.upper(),
    ]
    for name in candidates:
        runner = getattr(module, name, None)
        if callable(runner):
            return runner
    return None


def _resolve_pertpy_class_runner(module: Any, method: str) -> Optional[Callable[..., Any]]:
    """Return a callable wrapper for class-based Pertpy differential expression APIs."""

    class_aliases = {
        "edger": ["EdgeR"],
        "pydeseq2": ["PyDESeq2"],
        "statsmodels": ["Statsmodels"],
        "ttest": ["TTest"],
        "wilcoxon": ["WilcoxonTest"],
    }

    method_key = method.lower()
    candidate_names = list(class_aliases.get(method_key, []))
    candidate_names.extend(
        name
        for name in {
            method,
            method.capitalize(),
            method.upper(),
            method.title(),
        }
        if isinstance(name, str)
    )
    # Preserve order while removing duplicates
    seen: set[str] = set()
    deduped: list[str] = []
    for name in candidate_names:
        if not isinstance(name, str):
            continue
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    candidate_names = deduped

    for name in candidate_names:
        cls = getattr(module, name, None)
        if cls is None or not isinstance(cls, type):
            continue
        compare_groups = getattr(cls, "compare_groups", None)
        if compare_groups is None or not callable(compare_groups):
            continue

        current_cls = cls

        def runner(
            adata,
            *,
            groupby=None,
            group_key=None,
            control=None,
            reference=None,
            **kwargs,
        ):
            _ = kwargs  # Allow compatibility with legacy keyword arguments
            column = groupby or group_key
            baseline = control if control is not None else reference
            if column is None:
                raise TypeError("Pertpy runner requires a groupby column")
            if baseline is None:
                raise TypeError("Pertpy runner requires a control/reference label")

            obs_column = adata.obs[column]
            groups_to_compare = [value for value in obs_column.unique().tolist() if value != baseline]
            if not groups_to_compare:
                raise ValueError("No perturbation groups available for comparison")

            return current_cls.compare_groups(
                adata,
                column=column,
                baseline=baseline,
                groups_to_compare=groups_to_compare,
            )

        return runner
    return None


def _convert_reference_result_to_dataframe(result: Any) -> Optional[pd.DataFrame]:
    """Normalise a Pertpy reference result to a ``DataFrame`` when possible."""

    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result.copy()
    if isinstance(result, Mapping):
        frames = []
        for perturbation, value in result.items():
            if isinstance(value, pd.DataFrame):
                frame = value.copy()
                if "perturbation" not in frame.columns:
                    frame["perturbation"] = str(perturbation)
                frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True)
    if hasattr(result, "to_dataframe"):
        return result.to_dataframe()
    if hasattr(result, "to_df"):
        return result.to_df()
    return None


def _run_edger_direct(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    min_genes: int,
    min_cells_per_gene: int,
    min_cells_per_perturbation: int,
    output_dir: Path,
    data_name: str,
) -> tuple[pd.DataFrame | None, Optional[Path], Optional[str]]:
    """Execute edgeR directly via rpy2 without Pertpy wrapper."""

    try:
        import scanpy as sc
        from rpy2 import robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
    except ImportError as exc:
        return None, None, str(exc)

    try:
        adata = _prepare_reference_anndata(
            dataset_path,
            min_genes=min_genes,
            min_cells_per_gene=min_cells_per_gene,
            min_cells_per_perturbation=min_cells_per_perturbation,
            perturbation_column=perturbation_column,
            control_label=control_label,
        )
    except ImportError as exc:
        return None, None, str(exc)

    groups = adata.obs[perturbation_column].astype(str).values
    unique_groups = [g for g in np.unique(groups) if g != control_label]

    if not unique_groups:
        return None, None, "No perturbation groups available for comparison"

    try:
        # Load edgeR
        ro.r('library(edgeR)')

        # Convert count matrix to R
        counts = adata.X.T  # genes x cells
        if hasattr(counts, 'toarray'):
            counts = counts.toarray()

        with localconverter(ro.default_converter + numpy2ri.converter):
            ro.globalenv['counts'] = counts
            ro.globalenv['groups'] = ro.StrVector(groups)
            ro.globalenv['gene_names'] = ro.StrVector(adata.var_names)

            # Run edgeR analysis
            ro.r('''
            rownames(counts) <- gene_names
            y <- DGEList(counts=counts, group=groups)
            y <- calcNormFactors(y)
            design <- model.matrix(~0 + groups)
            colnames(design) <- gsub("groups", "", colnames(design))
            y <- estimateDisp(y, design)
            fit <- glmQLFit(y, design)
            ''')

            # Run tests for each non-control group
            all_results = []

            for group in unique_groups:
                ro.globalenv['target_group'] = group
                ro.globalenv['control'] = control_label

                # Make contrast and run test
                ro.r('''
                contrast_vec <- makeContrasts(
                    contrasts = paste0(target_group, "-", control),
                    levels = design
                )
                lrt <- glmQLFTest(fit, contrast=contrast_vec)
                ''')

                # Extract results manually as vectors (avoids pickling issues)
                genes = np.array(ro.r('rownames(lrt$table)'))
                logFC = np.array(ro.r('lrt$table$logFC'))
                logCPM = np.array(ro.r('lrt$table$logCPM'))
                F_stat = np.array(ro.r('lrt$table$F'))
                PValue = np.array(ro.r('lrt$table$PValue'))
                FDR = np.array(ro.r('p.adjust(lrt$table$PValue, method="BH")'))

                # Create DataFrame
                results_df = pd.DataFrame({
                    'gene': genes,
                    'logFC': logFC,
                    'logCPM': logCPM,
                    'F': F_stat,
                    'PValue': PValue,
                    'FDR': FDR,
                    'perturbation': group
                })

                all_results.append(results_df)

        final_results = pd.concat(all_results, ignore_index=True)

        reference_path: Optional[Path] = None
        if not final_results.empty:
            reference_path = output_dir / f"{data_name}_edger_direct_de.csv"
            final_results.to_csv(reference_path, index=False)

        return final_results, reference_path, None

    except Exception as exc:
        return None, None, str(exc)


def _run_pertpy_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    min_genes: int,
    min_cells_per_gene: int,
    min_cells_per_perturbation: int,
    backend: str,
    output_dir: Path,
    data_name: str,
) -> tuple[pd.DataFrame | None, Optional[Path], Optional[str]]:
    """Execute a Pertpy-backed differential expression method."""

    try:
        import pertpy as pt
        import scanpy as sc  # Needed for AnnData IO
    except ImportError as exc:  # pragma: no cover - optional dependency
        return None, None, str(exc)

    _ = sc  # Silence unused import warnings in environments without Scanpy

    module = getattr(pt, "tools", None)
    if module is None:
        return None, None, "pertpy.tools module unavailable"

    candidate_modules: list[Any] = []
    de_module = getattr(module, "differential_expression", None)
    if de_module is not None:
        candidate_modules.append(de_module)
    candidate_modules.append(module)
    try:
        candidate_modules.append(import_module("pertpy.tools._differential_gene_expression"))
    except Exception:  # pragma: no cover - optional dependency handling
        pass

    runner: Optional[Callable[..., Any]] = None
    for candidate in candidate_modules:
        runner = _resolve_pertpy_runner(candidate, backend)
        if runner is not None:
            break
        runner = _resolve_pertpy_class_runner(candidate, backend)
        if runner is not None:
            break

    if runner is None:
        return None, None, f"Pertpy differential expression runner '{backend}' not found"

    try:
        adata = _prepare_reference_anndata(
            dataset_path,
            min_genes=min_genes,
            min_cells_per_gene=min_cells_per_gene,
            min_cells_per_perturbation=min_cells_per_perturbation,
            perturbation_column=perturbation_column,
            control_label=control_label,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        return None, None, str(exc)

    call_attempts = (
        {"groupby": perturbation_column, "control": control_label},
        {"group_key": perturbation_column, "control": control_label},
        {"groupby": perturbation_column, "reference": control_label},
    )
    last_type_error: Optional[Exception] = None
    result = None
    for kwargs in call_attempts:
        try:
            result = runner(adata, **kwargs)
        except TypeError as exc:
            last_type_error = exc
            continue
        except Exception as exc:  # pragma: no cover - defensive
            return None, None, str(exc)
        else:
            break
    else:
        if last_type_error is not None:
            return None, None, str(last_type_error)
        return None, None, "Pertpy differential expression runner failed to execute"

    df = _convert_reference_result_to_dataframe(result)
    reference_path: Optional[Path] = None
    if df is not None and not df.empty:
        reference_path = output_dir / f"{data_name}_{backend}_pertpy_de.csv"
        df.to_csv(reference_path, index=False)
    return df, reference_path, None


def run_scanpy_qc_comparison(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    output_dir: Path,
) -> ComparisonResult:
    """Run the full Scanpy validation workflow for QC comparisons."""

    # Create scanpy subdirectory for comparison outputs
    scanpy_dir = output_dir / "scanpy"
    scanpy_dir.mkdir(parents=True, exist_ok=True)

    return compare_with_scanpy(
        dataset_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_genes=min_genes,
        min_cells_per_perturbation=min_cells_per_perturbation,
        min_cells_per_gene=min_cells_per_gene,
        gene_name_column=gene_name_column,
        output_dir=scanpy_dir,
        data_name="qc",
    )


def run_scanpy_de_comparison(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    output_dir: Path,
    test_type: str,
) -> DifferentialComparisonSummary:
    """Compare streaming differential expression against Scanpy."""

    # Create scanpy subdirectory for comparison outputs
    scanpy_dir = output_dir / "scanpy"
    scanpy_dir.mkdir(parents=True, exist_ok=True)

    if test_type == "wilcoxon":
        stream_result = wilcoxon_test(
            path=dataset_path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            gene_name_column=gene_name_column,
            output_dir=scanpy_dir,
            data_name="de",
        )
        reference_method = "wilcoxon"
    else:
        stream_result = wald_test(
            path=dataset_path,
            perturbation_column=perturbation_column,
            control_label=control_label,
            gene_name_column=gene_name_column,
            output_dir=scanpy_dir,
            data_name="de",
        )
        reference_method = "t-test"

    streaming_frame = _streaming_de_to_frame(stream_result)
    streaming_path = None
    if not streaming_frame.empty:
        any_result = next(iter(stream_result.values()))
        streaming_path = str(getattr(any_result, "result_path", "")) or None

    reference_df, reference_path, error = _run_scanpy_de(
        dataset_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_genes=min_genes,
        min_cells_per_gene=min_cells_per_gene,
        min_cells_per_perturbation=min_cells_per_perturbation,
        method=reference_method,
        output_dir=scanpy_dir,
        data_name="reference",
    )

    metrics = {key: None for key in DE_METRIC_KEYS}
    if reference_df is not None:
        metrics = _compare_de_frames(streaming_frame, _standardise_de_dataframe(reference_df))

    return DifferentialComparisonSummary(
        test_type=test_type,
        reference_tool=f"scanpy_{reference_method}",
        metrics=metrics,
        streaming_result_path=streaming_path,
        reference_result_path=str(reference_path) if reference_path else None,
        error=error,
    )


def run_edger_direct_comparison(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    output_dir: Path,
) -> DifferentialComparisonSummary:
    """Compare streaming GLM-based tests against edgeR (via direct rpy2)."""

    # Create edger subdirectory for comparison outputs
    edger_dir = output_dir / "edger"
    edger_dir.mkdir(parents=True, exist_ok=True)

    stream_result = wald_test(
        path=dataset_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        gene_name_column=gene_name_column,
        output_dir=edger_dir,
        data_name="edger",
    )
    streaming_frame = _streaming_de_to_frame(stream_result)
    streaming_path = None
    if not streaming_frame.empty:
        any_result = next(iter(stream_result.values()))
        streaming_path = str(getattr(any_result, "result_path", "")) or None

    reference_df, reference_path, error = _run_edger_direct(
        dataset_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_genes=min_genes,
        min_cells_per_gene=min_cells_per_gene,
        min_cells_per_perturbation=min_cells_per_perturbation,
        output_dir=edger_dir,
        data_name="reference",
    )

    metrics = {key: None for key in DE_METRIC_KEYS}
    if reference_df is not None:
        metrics = _compare_de_frames(streaming_frame, _standardise_de_dataframe(reference_df))

    return DifferentialComparisonSummary(
        test_type="glm",
        reference_tool="edger_direct",
        metrics=metrics,
        streaming_result_path=streaming_path,
        reference_result_path=str(reference_path) if reference_path else None,
        error=error,
    )


def run_pertpy_de_comparison(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    gene_name_column: str,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    output_dir: Path,
    backend: str,
) -> DifferentialComparisonSummary:
    """Compare streaming GLM-based tests against a Pertpy backend."""

    # Create pertpy subdirectory for comparison outputs
    pertpy_dir = output_dir / "pertpy"
    pertpy_dir.mkdir(parents=True, exist_ok=True)

    stream_result = wald_test(
        path=dataset_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        gene_name_column=gene_name_column,
        output_dir=pertpy_dir,
        data_name=backend,
    )
    streaming_frame = _streaming_de_to_frame(stream_result)
    streaming_path = None
    if not streaming_frame.empty:
        any_result = next(iter(stream_result.values()))
        streaming_path = str(getattr(any_result, "result_path", "")) or None

    reference_df, reference_path, error = _run_pertpy_de(
        dataset_path,
        perturbation_column=perturbation_column,
        control_label=control_label,
        min_genes=min_genes,
        min_cells_per_gene=min_cells_per_gene,
        min_cells_per_perturbation=min_cells_per_perturbation,
        backend=backend,
        output_dir=pertpy_dir,
        data_name="reference",
    )

    metrics = {key: None for key in DE_METRIC_KEYS}
    if reference_df is not None:
        metrics = _compare_de_frames(streaming_frame, _standardise_de_dataframe(reference_df))

    return DifferentialComparisonSummary(
        test_type="glm",
        reference_tool=f"pertpy_{backend}",
        metrics=metrics,
        streaming_result_path=streaming_path,
        reference_result_path=str(reference_path) if reference_path else None,
        error=error,
    )

def _summarise_quality_control(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    total_cells = int(context.get("dataset_cells", 0))
    total_genes = int(context.get("dataset_genes", 0))
    kept_cells = int(getattr(result, "cell_mask").sum())
    kept_genes = int(getattr(result, "gene_mask").sum())
    removed_cells = max(total_cells - kept_cells, 0)
    removed_genes = max(total_genes - kept_genes, 0)
    result_path = _normalise_path(getattr(result, "filtered_path", None), context)
    return {
        "cells_total": total_cells,
        "cells_kept": kept_cells,
        "cells_removed": removed_cells,
        "cells_kept_pct": _percentage(kept_cells, total_cells) if total_cells else None,
        "genes_total": total_genes,
        "genes_kept": kept_genes,
        "genes_removed": removed_genes,
        "genes_kept_pct": _percentage(kept_genes, total_genes) if total_genes else None,
        "result_path": result_path,
    }


def _summarise_dataframe(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise tabular results, including on-disk export paths when available."""

    rows = cols = 0
    if hasattr(result, "shape"):
        try:
            rows, cols = result.shape
        except Exception:  # pragma: no cover - defensive fallback
            rows = cols = 0

    path_candidate: str | Path | None = None
    for attr in ("path", "filename", "result_path"):
        candidate = getattr(result, attr, None)
        if candidate:
            path_candidate = candidate
            break

    summary: Dict[str, Any] = {
        "rows": int(rows),
        "columns": int(cols),
    }

    if path_candidate:
        summary["result_path"] = _normalise_path(path_candidate, context)

    return summary


def _summarise_de_mapping(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    groups: list[str] = []
    if isinstance(result, Mapping):
        groups = list(result.keys())
    elif hasattr(result, "groups"):
        groups = list(getattr(result, "groups"))

    n_groups = len(groups)
    n_genes = 0
    output_path = getattr(result, "result_path", None)
    if groups:
        first_key = groups[0]
        try:
            first = result[first_key]
        except Exception:  # pragma: no cover - defensive
            first = None
        if first is not None:
            genes = getattr(first, "genes", None)
            if genes is not None:
                n_genes = int(len(genes))
            if output_path is None:
                output_path = getattr(first, "result_path", None)
    summary = {
        "groups": n_groups,
        "genes": n_genes,
    }
    if output_path:
        summary["result_path"] = _normalise_path(output_path, context)
    return summary


def _summarise_scanpy_comparison(result: ComparisonResult, context: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise quality control and preprocessing comparisons with Scanpy."""

    stream_total = sum(result.streamlined_timings.values()) if result.streamlined_timings else None
    reference_total = sum(result.reference_timings.values()) if result.reference_timings else None
    summary = {
        "comparison_category": "quality_control_preprocessing",
        "reference_tool": "scanpy",
        "normalization_max_abs_diff": result.normalization_max_abs_diff,
        "log1p_max_abs_diff": result.log1p_max_abs_diff,
        "avg_log_effect_max_abs_diff": result.avg_log_effect_max_abs_diff,
        "pseudobulk_effect_max_abs_diff": result.pseudobulk_effect_max_abs_diff,
        "streamlined_cell_count": result.streamlined_cell_count,
        "reference_cell_count": result.reference_cell_count,
        "streamlined_gene_count": result.streamlined_gene_count,
        "reference_gene_count": result.reference_gene_count,
        "stream_peak_memory_mb": result.streamlined_peak_memory_mb,
        "reference_peak_memory_mb": result.reference_peak_memory_mb,
        "stream_total_seconds": stream_total,
        "reference_total_seconds": reference_total,
        "stream_timing_breakdown": _format_timing_summary(result.streamlined_timings),
        "reference_timing_breakdown": _format_timing_summary(result.reference_timings),
    }
    summary.update({f"wald_{key}": value for key, value in result.wald_metrics.items()})
    summary.update({f"wilcoxon_{key}": value for key, value in result.wilcoxon_metrics.items()})
    return summary


def _summarise_de_comparison(
    result: DifferentialComparisonSummary, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Summarise differential expression comparisons."""

    summary = {
        "comparison_category": "differential_expression",
        "test_type": result.test_type,
        "reference_tool": result.reference_tool,
        "streaming_result_path": _normalise_path(result.streaming_result_path, context),
        "reference_result_path": _normalise_path(result.reference_result_path, context),
    }
    summary.update(result.metrics)
    if result.error:
        summary["error"] = result.error
    return summary


def _worker(
    queue: mp.Queue,
    method: BenchmarkMethod,
    context: Dict[str, Any],
    memory_limit: int | None,
    time_limit: int | None,
) -> None:
    """Execute ``method`` with optional resource limits and report the outcome."""

    # Force single-threaded operation for R/BLAS to avoid multiprocessing conflicts
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['R_THREADS'] = '1'
    
    if memory_limit and memory_limit > 0:
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    if time_limit and time_limit > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (time_limit, time_limit))

    start = time.perf_counter()
    try:
        result = method.function(**method.kwargs)
        elapsed = time.perf_counter() - start
        usage = resource.getrusage(resource.RUSAGE_SELF)
        max_rss_kb = usage.ru_maxrss
        summary = method.summary(result, context)
        queue.put(
            {
                "status": "success",
                "elapsed_seconds": elapsed,
                "max_rss_kb": max_rss_kb,
                "summary": summary,
            }
        )
    except MemoryError as exc:
        elapsed = time.perf_counter() - start
        queue.put(
            {
                "status": "memory_limit",
                "elapsed_seconds": elapsed,
                "max_rss_kb": None,
                "error": f"MemoryError: {exc}",
            }
        )
    except Exception as exc:  # pragma: no cover - defensive reporting
        elapsed = time.perf_counter() - start
        queue.put(
            {
                "status": "error",
                "elapsed_seconds": elapsed,
                "max_rss_kb": None,
                "error": f"{exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _run_with_limits(
    method: BenchmarkMethod,
    context: Dict[str, Any],
    memory_limit: int | None,
    time_limit: int | None,
) -> Dict[str, Any]:
    # Special handling for edgeR: run directly without multiprocessing to avoid R/fork issues
    if 'edger_direct' in method.name.lower():
        start = time.perf_counter()
        try:
            result = method.function(**method.kwargs)
            elapsed = time.perf_counter() - start
            usage = resource.getrusage(resource.RUSAGE_SELF)
            max_rss_kb = usage.ru_maxrss
            summary = method.summary(result, context)
            return {
                "status": "success",
                "elapsed_seconds": elapsed,
                "max_memory_mb": max_rss_kb / 1024,
                "summary": summary,
            }
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return {
                "status": "error",
                "elapsed_seconds": elapsed,
                "max_memory_mb": None,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
    
    # Use spawn context for R/rpy2 compatibility (avoids fork() issues with R threading)
    needs_spawn = 'pertpy' in method.name.lower()
    mp_context = mp.get_context('spawn') if needs_spawn else mp
    
    queue = mp_context.Queue()
    process = mp_context.Process(
        target=_worker,
        args=(queue, method, context, memory_limit, time_limit),
        name=f"benchmark-{method.name}",
    )
    process.start()
    join_timeout = None
    if time_limit and time_limit > 0:
        join_timeout = time_limit + 5
    process.join(timeout=join_timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "status": "timeout",
            "elapsed_seconds": None,
            "max_memory_mb": None,
            "summary": {},
            "error": f"Exceeded time limit of {time_limit} seconds",
        }

    if not queue.empty():
        payload = queue.get()
    else:
        payload = {
            "status": "error",
            "elapsed_seconds": None,
            "max_rss_kb": None,
            "error": f"Process exited with code {process.exitcode}",
        }

    payload.setdefault("summary", {})
    max_rss_kb = payload.pop("max_rss_kb", None)
    if max_rss_kb is not None:
        payload["max_memory_mb"] = max_rss_kb / 1024
    else:
        payload.setdefault("max_memory_mb", None)
    return payload


def _load_dataset_context(path: Path) -> Dict[str, Any]:
    backed = read_backed(path)
    try:
        context = {
            "dataset_cells": backed.n_obs,
            "dataset_genes": backed.n_vars,
        }
    finally:
        backed.file.close()
    return context


def create_benchmark_suite(
    dataset_path: Path,
    output_dir: Path,
    perturbation_column: str = "perturbation",
    control_label: str | None = None,
    gene_name_column: str | None = "gene_symbols",
    qc_params: QCParams | None = None,
    n_cores: int | None = None,
) -> Dict[str, BenchmarkMethod]:
    """Return the available benchmark methods for the provided dataset.
    
    Parameters
    ----------
    dataset_path
        Path to standardized dataset.
    output_dir
        Directory for benchmark outputs.
    perturbation_column
        Name of perturbation column (should be 'perturbation' after standardization).
    control_label
        Control label (should be 'control' after standardization or None for auto-detect).
    gene_name_column
        Gene name column or None to use var.index.
    qc_params
        QC parameters. If None, will be calculated adaptively.
    n_cores
        Number of cores to use for parallel DE methods. If None, auto-detects.
    """
    import anndata as ad
    
    # Use provided QC params or calculate adaptively
    if qc_params is None:
        # Calculate adaptive QC parameters
        adata_temp = ad.read_h5ad(dataset_path, backed='r')
        try:
            adaptive_thresholds = calculate_adaptive_qc_thresholds(
                adata_temp, perturbation_column, mode='conservative'
            )
            min_genes = adaptive_thresholds['min_genes']
            min_cells_per_perturbation = adaptive_thresholds['min_cells_per_perturbation']
            min_cells_per_gene = adaptive_thresholds['min_cells_per_gene']
            chunk_size = adaptive_thresholds['chunk_size']
        finally:
            adata_temp.file.close()
    else:
        min_genes = qc_params.min_genes
        min_cells_per_perturbation = qc_params.min_cells_per_perturbation
        min_cells_per_gene = qc_params.min_cells_per_gene
        chunk_size = qc_params.chunk_size

    backed = read_backed(dataset_path)
    try:
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' was not found in adata.obs. "
                f"Available columns: {list(backed.obs.columns)}"
            )
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        detected_control = resolve_control_label(labels, control_label, verbose=False)
    finally:
        backed.file.close()

    shared_kwargs = {
        "perturbation_column": perturbation_column,
        "control_label": detected_control,
        "gene_name_column": gene_name_column,
    }

    # Create crispyx subdirectory for streaming outputs
    crispyx_dir = output_dir / "crispyx"
    crispyx_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        "quality_control": BenchmarkMethod(
            name="quality_control",
            description="Streaming quality control filters",
            function=quality_control_summary,
            kwargs={
                "path": dataset_path,
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                **shared_kwargs,
                "output_dir": crispyx_dir,
                "data_name": "qc",
            },
            summary=_summarise_quality_control,
            category="Streaming pipeline",
        ),
        "average_log_expression": BenchmarkMethod(
            name="average_log_expression",
            description="Average log-normalised expression per perturbation",
            function=compute_average_log_expression,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": crispyx_dir,
                "data_name": "pb",
            },
            summary=_summarise_dataframe,
            category="Streaming pipeline",
        ),
        "pseudobulk_expression": BenchmarkMethod(
            name="pseudobulk_expression",
            description="Pseudo-bulk log fold-change per perturbation",
            function=compute_pseudobulk_expression,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": crispyx_dir,
                "data_name": "pb",
            },
            summary=_summarise_dataframe,
            category="Streaming pipeline",
        ),
        "wald_test": BenchmarkMethod(
            name="wald_test",
            description="Wald differential expression test",
            function=wald_test,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": crispyx_dir,
                "data_name": "de",
                "n_jobs": n_cores,
            },
            summary=_summarise_de_mapping,
            category="Differential expression",
        ),
        "wilcoxon_test": BenchmarkMethod(
            name="wilcoxon_test",
            description="Wilcoxon rank-sum differential expression",
            function=wilcoxon_test,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": crispyx_dir,
                "data_name": "de",
                "n_jobs": n_cores,
            },
            summary=_summarise_de_mapping,
            category="Differential expression",
        ),
        "scanpy_quality_control_comparison": BenchmarkMethod(
            name="scanpy_quality_control_comparison",
            description="Quality control comparison against Scanpy",
            function=run_scanpy_qc_comparison,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
            },
            summary=_summarise_scanpy_comparison,
            category="Reference: Scanpy",
        ),
        "scanpy_wald_comparison": BenchmarkMethod(
            name="scanpy_wald_comparison",
            description="Wald/t-test comparison against Scanpy",
            function=run_scanpy_de_comparison,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
                "test_type": "wald",
            },
            summary=_summarise_de_comparison,
            category="Reference: Scanpy",
        ),
        "scanpy_wilcoxon_comparison": BenchmarkMethod(
            name="scanpy_wilcoxon_comparison",
            description="Wilcoxon comparison against Scanpy",
            function=run_scanpy_de_comparison,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
                "test_type": "wilcoxon",
            },
            summary=_summarise_de_comparison,
            category="Reference: Scanpy",
        ),
        "edger_direct_comparison": BenchmarkMethod(
            name="edger_direct_comparison",
            description="GLM comparison against edgeR (direct rpy2)",
            function=run_edger_direct_comparison,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
            },
            summary=_summarise_de_comparison,
            category="Reference: edgeR",
        ),
        "pertpy_pydeseq2_comparison": BenchmarkMethod(
            name="pertpy_pydeseq2_comparison",
            description="GLM comparison against PyDESeq2 via Pertpy",
            function=run_pertpy_de_comparison,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
                "backend": "pydeseq2",
            },
            summary=_summarise_de_comparison,
            category="Reference: Pertpy",
        ),
        "pertpy_statsmodels_comparison": BenchmarkMethod(
            name="pertpy_statsmodels_comparison",
            description="GLM comparison against statsmodels via Pertpy",
            function=run_pertpy_de_comparison,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "gene_name_column": shared_kwargs["gene_name_column"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": output_dir,
                "backend": "statsmodels",
            },
            summary=_summarise_de_comparison,
            category="Reference: Pertpy",
        ),
    }
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark streamlined CRISPR analysis methods")
    default_output = Path(__file__).resolve().parent / "results"
    
    # Config file option (new)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file (overrides individual arguments)",
    )
    
    # Single dataset arguments (backward compatible)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "data" / "demo_benchmark.h5ad",
        help="Path to an AnnData .h5ad file to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to store benchmark outputs and summaries",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Subset of methods to run. Defaults to all available methods.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Maximum number of CPU seconds allowed per method (0 disables the limit)",
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=4.0,
        help="Maximum memory per method in gigabytes (0 disables the limit)",
    )
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help=(
            "Generate the synthetic demo dataset at --data-path before running the benchmarks. "
            "This is useful when bootstrapping a fresh checkout."
        ),
    )
    parser.add_argument(
        "--demo-seed",
        type=int,
        default=0,
        help=(
            "Random seed used when generating the demo dataset (set to -1 to sample a random seed)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars and non-essential output",
    )
    return parser.parse_args()


def _format_summary_markdown(summary: Dict[str, Any]) -> str:
    """Return a narrative Markdown summary for ``summary`` statistics."""

    lines: list[str] = ["## Benchmark summary", ""]

    total_methods = summary.get("total_methods", 0)
    success_count = summary.get("success_count", 0)
    timeout_count = summary.get("timeout_count", 0)
    memory_limit_count = summary.get("memory_limit_count", 0)
    error_count = summary.get("error_count", 0)
    non_success = summary.get("non_success_count", 0)
    success_rate = summary.get("success_rate")
    average_runtime = summary.get("average_runtime_seconds")

    totals_line = f"- **Methods executed:** {total_methods}"
    lines.append(totals_line)

    success_line = f"- **Succeeded:** {success_count}"
    if success_rate is not None:
        success_line += f" ({success_rate * 100:.1f}% success rate)"
    lines.append(success_line)

    if non_success:
        lines.append(f"- **Did not succeed:** {non_success}")
    if timeout_count:
        lines.append(f"  - Timeouts: {timeout_count}")
    if memory_limit_count:
        lines.append(f"  - Memory limit exceeded: {memory_limit_count}")
    if error_count:
        lines.append(f"  - Errors: {error_count}")

    if average_runtime is not None:
        lines.append(f"- **Average runtime:** {average_runtime:.3f}s")

    categories = summary.get("categories", [])
    if categories:
        lines.append("- **Average runtime by category:**")
        for category_summary in categories:
            category_name = category_summary.get("category", "Uncategorised")
            method_count = category_summary.get("method_count", 0)
            category_runtime = category_summary.get("average_runtime_seconds")
            status_counts = category_summary.get("status_counts", {})
            runtime_fragment = (
                f"{category_runtime:.3f}s"
                if category_runtime is not None
                else "no runtime recorded"
            )
            status_fragments = [f"{status}={count}" for status, count in status_counts.items()]
            status_clause = f" ({', '.join(status_fragments)})" if status_fragments else ""
            lines.append(
                f"  - {category_name}: {runtime_fragment} across {method_count} method(s){status_clause}"
            )

    dependency_errors = summary.get("dependency_errors", [])
    other_errors = summary.get("other_errors", [])
    if dependency_errors or other_errors:
        lines.append("- **Notable issues:**")
        if dependency_errors:
            lines.append("  - Dependency errors detected:")
            for message in dependency_errors:
                lines.append(f"    - {message}")
        if other_errors:
            lines.append("  - Other errors recorded:")
            for message in other_errors:
                lines.append(f"    - {message}")
    else:
        lines.append("- **Notable issues:** None")

    return "\n".join(lines).strip() + "\n"


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
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        data_rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header_row, separator_row, *data_rows]) + "\n"


def _dataframe_to_markdown(
    df: pd.DataFrame, summary: Optional[Dict[str, Any]] = None
) -> str:
    """Render ``df`` as grouped Markdown tables, one section per category."""

    computed_summary = summary or _compute_aggregate_statistics(df)
    narrative = _format_summary_markdown(computed_summary)

    if df.empty:
        return narrative + "\n| |\n|---|\n"

    if "category" not in df.columns:
        tables = _frame_to_markdown_table(df)
        return narrative + "\n" + tables

    sections: list[str] = []
    for category, group in df.groupby("category", sort=False):
        reduced = group.drop(columns=["category"]).copy()
        drop_cols = [col for col in reduced.columns if reduced[col].isna().all()]
        if drop_cols:
            reduced = reduced.drop(columns=drop_cols)
        sections.append(f"### {category}\n\n" + _frame_to_markdown_table(reduced))
    tables = "\n".join(sections)
    return narrative + "\n" + tables


def _run_single_benchmark(
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, dict]:
    """Run benchmarks for a single dataset and return results DataFrame and metadata.
    
    Returns
    -------
    tuple
        (results_dataframe, metadata_dict)
    """
    
    # Standardize dataset (uses cache if available)
    if not config.quiet:
        print(f"\nStandardizing dataset: {config.dataset_path.name}")
    
    standardized_path = standardize_dataset(
        dataset_path=config.dataset_path,
        perturbation_column=config.perturbation_column,
        control_label=config.control_label,
        gene_name_column=config.gene_name_column,
        output_dir=config.output_dir,
        force=config.force_restandardize,
    )
    
    context = _load_dataset_context(standardized_path)
    context["dataset_path"] = str(standardized_path)
    context["output_dir"] = config.output_dir
    
    # Track whether QC params are adaptive or user-specified
    adaptive_qc = config.qc_params is None
    qc_params_used = None
    
    # Calculate or use provided QC params (this happens inside create_benchmark_suite)
    # We need to extract them for logging
    if adaptive_qc:
        import anndata as ad
        if not config.quiet:
            print(f"\nCalculating adaptive QC parameters (mode: {config.adaptive_qc_mode})...")
        
        adata_temp = ad.read_h5ad(standardized_path, backed='r')
        try:
            qc_params_used = calculate_adaptive_qc_thresholds(
                adata_temp, "perturbation", mode=config.adaptive_qc_mode
            )
            
            if not config.quiet:
                print(f"  ✓ min_genes: {qc_params_used['min_genes']}")
                print(f"  ✓ min_cells_per_perturbation: {qc_params_used['min_cells_per_perturbation']}")
                print(f"  ✓ min_cells_per_gene: {qc_params_used['min_cells_per_gene']}")
                print(f"  ✓ chunk_size: {qc_params_used['chunk_size']}")
        finally:
            adata_temp.file.close()
    else:
        qc_params_used = {
            "min_genes": config.qc_params.min_genes,
            "min_cells_per_perturbation": config.qc_params.min_cells_per_perturbation,
            "min_cells_per_gene": config.qc_params.min_cells_per_gene,
            "chunk_size": config.qc_params.chunk_size,
        }

    available_methods = create_benchmark_suite(
        dataset_path=standardized_path,
        output_dir=config.output_dir,
        perturbation_column="perturbation",  # Always 'perturbation' after standardization
        control_label="control",  # Always 'control' after standardization
        gene_name_column=config.gene_name_column,
        qc_params=config.qc_params,  # Will use adaptive if None
        n_cores=config.n_cores,
    )
    
    methods_to_run = config.methods_to_run
    if methods_to_run:
        selected_names = methods_to_run
    else:
        ordered_methods = sorted(available_methods.values(), key=_method_sort_key)
        selected_names = [method.name for method in ordered_methods]

    rows = []
    
    # Calculate memory limit in bytes
    memory_limit_bytes = None
    if config.resource_limits.memory_limit > 0:
        memory_limit_bytes = int(config.resource_limits.memory_limit * 1024 * 1024 * 1024)

    # Create progress bar for benchmark execution
    show_progress = config.show_progress and not config.quiet
    if show_progress:
        method_iterator = tqdm(
            selected_names,
            desc="Running benchmarks",
            unit="method",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
    else:
        method_iterator = selected_names

    for name in method_iterator:
        if name not in available_methods:
            raise ValueError(f"Unknown method '{name}'. Available methods: {sorted(available_methods)}")
        method = available_methods[name]
        
        # Update progress bar with current method name
        if show_progress:
            method_iterator.set_description(f"Running {method.name}")  # type: ignore
        
        result = _run_with_limits(
            method, context, memory_limit_bytes, config.resource_limits.time_limit
        )
        
        # Update progress bar with result status
        if show_progress:
            status = result.get("status", "unknown")
            mem_mb = result.get("max_memory_mb") or 0
            elapsed = result.get("elapsed_seconds") or 0
            method_iterator.set_postfix(  # type: ignore
                status=status, 
                memory=f"{mem_mb:.0f}MB",
                time=f"{elapsed:.1f}s",
                refresh=False
            )
        
        row = {
            "category": method.category,
            "method": method.name,
            "description": method.description,
            "status": result.get("status"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "max_memory_mb": result.get("max_memory_mb"),
        }
        summary = result.get("summary", {})
        if summary:
            row.update(summary)
        if result.get("error"):
            row["error"] = result["error"]
        rows.append(row)
    
    # Create metadata dict
    metadata = {
        "qc_params_used": qc_params_used,
        "adaptive_qc": adaptive_qc,
        "adaptive_qc_mode": config.adaptive_qc_mode if adaptive_qc else None,
        "standardized_dataset_path": str(standardized_path),
        "original_dataset_path": str(config.dataset_path),
    }

    return _postprocess_results(pd.DataFrame(rows)), metadata


def main() -> None:
    args = parse_args()
    
    # Load configuration from YAML or command-line arguments
    if args.config:
        config_result = BenchmarkConfig.from_yaml(args.config)
        configs = config_result if isinstance(config_result, list) else [config_result]
    else:
        # Traditional CLI mode - create config from arguments
        dataset_path = args.data_path
        output_dir: Path = args.output_dir
        
        if args.generate_demo:
            seed = None if args.demo_seed == -1 else args.demo_seed
            generated = write_demo_dataset(dataset_path, seed=seed)
            print(f"Generated demo dataset at {generated}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_path}' was not found. "
                "Generate it with --generate-demo or 'python benchmarking/generate_demo_dataset.py', "
                "or supply --data-path to an existing .h5ad file."
            )
        
        memory_limit_bytes = None
        if args.memory_limit and args.memory_limit > 0:
            memory_limit = args.memory_limit
        else:
            memory_limit = 0
            
        # Create single config from CLI args
        configs = [BenchmarkConfig(
            dataset_path=dataset_path,
            dataset_name=dataset_path.stem,
            output_dir=output_dir,
            qc_params=QCParams(),
            resource_limits=ResourceLimits(
                time_limit=args.time_limit,
                memory_limit=memory_limit,
            ),
            methods_to_run=args.methods,
            quiet=args.quiet,
        )]
    
    # Run benchmarks for each configuration
    for i, config in enumerate(configs):
        if len(configs) > 1:
            print(f"\n{'='*60}")
            print(f"Dataset {i+1}/{len(configs)}: {config.dataset_name}")
            print(f"{'='*60}")
        
        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run benchmarks with standardization and adaptive QC
        df, metadata = _run_single_benchmark(config)
        
        # Save results
        summary = _compute_aggregate_statistics(df)
        
        # Add metadata to summary
        summary.update(metadata)
        
        csv_path = config.output_dir / "results.csv"
        md_path = config.output_dir / "results.md"
        summary_json_path = config.output_dir / "summary.json"

        df.to_csv(csv_path, index=False)
        md_path.write_text(_dataframe_to_markdown(df, summary))
        summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

        if not config.quiet:
            print(f"\nBenchmark complete for {config.dataset_name}")
            print(f"  Results: {csv_path}")
            print(f"  Summary: {md_path}")
            print(f"  JSON: {summary_json_path}")
            if metadata.get("adaptive_qc"):
                print(f"\n  Adaptive QC parameters saved to summary.json")


if __name__ == "__main__":
    main()
