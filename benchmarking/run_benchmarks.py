"""Utilities to benchmark streaming CRISPR screen analysis methods."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import resource
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd

# Ensure the local package is importable when the project has not been installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from streamlined_crispr.data import read_backed
from streamlined_crispr.de import wald_test, wilcoxon_test
from streamlined_crispr.pseudobulk import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
)
from streamlined_crispr.qc import quality_control_summary


@dataclass
class BenchmarkMethod:
    """Description of a method that should be benchmarked."""

    name: str
    description: str
    function: Callable[..., Any]
    kwargs: Dict[str, Any]
    summary: Callable[[Any, Dict[str, Any]], Dict[str, Any]]


def _summarise_quality_control(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    total_cells = context.get("dataset_cells", 0)
    total_genes = context.get("dataset_genes", 0)
    kept_cells = int(getattr(result, "cell_mask").sum())
    kept_genes = int(getattr(result, "gene_mask").sum())
    return {
        "total_cells": total_cells,
        "kept_cells": kept_cells,
        "cells_removed": max(total_cells - kept_cells, 0),
        "total_genes": total_genes,
        "kept_genes": kept_genes,
        "genes_removed": max(total_genes - kept_genes, 0),
        "output_path": str(getattr(result, "filtered_path")),
    }


def _summarise_dataframe(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(result, "shape"):
        rows, cols = result.shape
    else:
        rows = cols = 0
    return {"rows": int(rows), "columns": int(cols)}


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
        summary["output_path"] = str(output_path)
    return summary


def _worker(
    queue: mp.Queue,
    method: BenchmarkMethod,
    context: Dict[str, Any],
    memory_limit: int | None,
    time_limit: int | None,
) -> None:
    """Execute ``method`` with optional resource limits and report the outcome."""

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
    queue: mp.Queue = mp.Queue()
    process = mp.Process(
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


def build_methods(dataset_path: Path, output_dir: Path) -> Dict[str, BenchmarkMethod]:
    """Return the available benchmark methods for the provided dataset."""

    shared_kwargs = {
        "perturbation_column": "perturbation",
        "control_label": "ctrl",
        "gene_name_column": "gene_symbols",
    }

    methods = {
        "quality_control": BenchmarkMethod(
            name="quality_control",
            description="Streaming quality control filters",
            function=quality_control_summary,
            kwargs={
                "path": dataset_path,
                "min_genes": 5,
                "min_cells_per_perturbation": 5,
                "min_cells_per_gene": 5,
                **shared_kwargs,
                "output_dir": output_dir,
                "data_name": "benchmark",
            },
            summary=_summarise_quality_control,
        ),
        "average_log_expression": BenchmarkMethod(
            name="average_log_expression",
            description="Average log-normalised expression per perturbation",
            function=compute_average_log_expression,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": output_dir,
                "data_name": "benchmark_avg_log",
            },
            summary=_summarise_dataframe,
        ),
        "pseudobulk_expression": BenchmarkMethod(
            name="pseudobulk_expression",
            description="Pseudo-bulk log fold-change per perturbation",
            function=compute_pseudobulk_expression,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": output_dir,
                "data_name": "benchmark_pseudobulk",
            },
            summary=_summarise_dataframe,
        ),
        "wald_test": BenchmarkMethod(
            name="wald_test",
            description="Wald differential expression test",
            function=wald_test,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": output_dir,
                "data_name": "benchmark_wald",
            },
            summary=_summarise_de_mapping,
        ),
        "wilcoxon_test": BenchmarkMethod(
            name="wilcoxon_test",
            description="Wilcoxon rank-sum differential expression",
            function=wilcoxon_test,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": output_dir,
                "data_name": "benchmark_wilcoxon",
            },
            summary=_summarise_de_mapping,
        ),
    }
    return methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark streamlined CRISPR analysis methods")
    default_output = Path(__file__).resolve().parent / "results"
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
    return parser.parse_args()


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render ``df`` as a GitHub-friendly Markdown table without extra dependencies."""

    if df.empty:
        return "| |\n|---|\n"

    table = df.copy()
    numeric_cols = table.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        table[numeric_cols] = table[numeric_cols].round(3)

    headers = list(table.columns)
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    data_rows = []
    for _, row in table.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        data_rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header_row, separator_row, *data_rows]) + "\n"


def main() -> None:
    args = parse_args()
    dataset_path = args.data_path
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_path}' was not found. "
            "Generate it with 'python benchmarking/generate_demo_dataset.py' "
            "or supply --data-path to an existing .h5ad file."
        )

    context = _load_dataset_context(dataset_path)
    context["dataset_path"] = str(dataset_path)

    available_methods = build_methods(dataset_path, output_dir)
    selected_names = args.methods or sorted(available_methods)

    rows = []
    memory_limit_bytes = None
    if args.memory_limit and args.memory_limit > 0:
        memory_limit_bytes = int(args.memory_limit * (1024**3))

    for name in selected_names:
        if name not in available_methods:
            raise ValueError(f"Unknown method '{name}'. Available methods: {sorted(available_methods)}")
        method = available_methods[name]
        result = _run_with_limits(method, context, memory_limit_bytes, args.time_limit)
        row = {
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

    df = pd.DataFrame(rows)
    csv_path = output_dir / "benchmark_results.csv"
    md_path = output_dir / "benchmark_results.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_dataframe_to_markdown(df))

    print(f"Benchmark complete. Saved results to {csv_path}")
    print(f"Markdown summary written to {md_path}")


if __name__ == "__main__":
    main()
