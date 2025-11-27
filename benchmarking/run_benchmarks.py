"""Utilities to benchmark streaming CRISPR screen analysis methods."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import resource
import sys
import threading
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
import anndata as ad
from tqdm import tqdm

# Ensure the local package is importable when the project has not been installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.env_config import (
    EnvironmentConfig,
    configure_r_environment,
    set_thread_env_vars,
)
from benchmarking.generate_demo_dataset import write_demo_dataset
from crispyx.data import (
    read_backed,
    resolve_control_label,
    calculate_adaptive_qc_thresholds,
    standardize_dataset,
)
from crispyx.de import t_test, wilcoxon_test, nb_glm_test
from crispyx.comparison import DE_METRIC_KEYS, compute_de_comparison_metrics
from crispyx.pseudobulk import (
    compute_average_log_expression,
    compute_pseudobulk_expression,
)
from crispyx.qc import quality_control_summary


# ============================================================================
# Reference Method Runners
# ============================================================================

import importlib
from scipy.stats import norm, rankdata

from crispyx.data import (
    ensure_gene_symbol_column,
    iter_matrix_chunks,
    normalize_total_block,
)
from crispyx.de import _tie_correction


def _get_peak_memory_bytes() -> float:
    """Return the current process peak RSS in bytes."""

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage)
    return float(usage) * 1024.0


def _normalize_total(matrix: np.ndarray, target_sum: float = 1e4) -> tuple[np.ndarray, np.ndarray]:
    import scipy.sparse as sp
    
    # Handle both sparse and dense matrices
    if sp.issparse(matrix):
        library_size = np.asarray(matrix.sum(axis=1)).flatten()
    else:
        library_size = matrix.sum(axis=1)
    
    scale = np.divide(
        target_sum,
        library_size,
        out=np.zeros_like(library_size, dtype=np.float64),
        where=library_size > 0,
    )
    
    if sp.issparse(matrix):
        # For sparse matrices, use multiply to keep sparse format
        # multiply() returns COO, so convert back to CSR for compatibility
        normalised = matrix.multiply(scale[:, None]).tocsr()
    else:
        normalised = matrix * scale[:, None]
    
    return normalised, library_size


def _log1p(matrix: np.ndarray) -> np.ndarray:
    import scipy.sparse as sp
    
    if sp.issparse(matrix):
        # For sparse matrices, apply log1p and keep sparse
        return matrix.log1p()
    else:
        return np.log1p(matrix)


def _filter_cells(matrix: np.ndarray, min_genes: int) -> np.ndarray:
    import scipy.sparse as sp
    
    if sp.issparse(matrix):
        expressed = np.asarray((matrix > 0).sum(axis=1)).flatten()
    else:
        expressed = (matrix > 0).sum(axis=1)
    return expressed >= min_genes


def _filter_genes(matrix: np.ndarray, min_cells: int) -> np.ndarray:
    import scipy.sparse as sp
    
    if sp.issparse(matrix):
        expressed = np.asarray((matrix > 0).sum(axis=0)).flatten()
    else:
        expressed = (matrix > 0).sum(axis=0)
    return expressed >= min_cells


def _get_peak_memory_mb() -> float:
    """Return the current process peak RSS in megabytes."""

    return _get_peak_memory_bytes() / (1024.0 * 1024.0)


def _peak_memory_delta_mb(baseline_bytes: float) -> float:
    return max(0.0, (_get_peak_memory_bytes() - baseline_bytes) / (1024.0 * 1024.0))


# Global flag to track if current RSS warning has been logged
_CURRENT_RSS_WARNING_LOGGED = False


def _get_current_rss_bytes() -> Optional[float]:
    """Return the current process RSS in bytes (not peak).
    
    Platform-specific implementation:
    - Linux: Reads /proc/self/statm (VmRSS field)
    - macOS: Falls back to peak RSS (ru_maxrss) - not true current
    - Windows: Attempts psutil if available
    
    Returns
    -------
    float | None
        Current RSS in bytes, or None if unavailable
        
    Notes
    -----
    On macOS, this function returns peak RSS (same as _get_peak_memory_bytes())
    because macOS lacks a direct API for current RSS without psutil.
    A warning is logged once when this limitation is detected.
    Users should install psutil for accurate average memory tracking on macOS.
    """
    global _CURRENT_RSS_WARNING_LOGGED
    
    if sys.platform.startswith('linux'):
        # Linux: Read /proc/self/statm
        # Format: size resident shared text lib data dt
        # We want resident (field 2), which is in pages
        try:
            with open('/proc/self/statm', 'r') as f:
                fields = f.readline().split()
                if len(fields) >= 2:
                    resident_pages = int(fields[1])
                    page_size = resource.getpagesize()
                    return float(resident_pages * page_size)
        except (IOError, OSError, ValueError) as exc:
            if not _CURRENT_RSS_WARNING_LOGGED:
                print(f"Warning: Failed to read /proc/self/statm for current RSS: {exc}")
                _CURRENT_RSS_WARNING_LOGGED = True
            return None
    
    elif sys.platform == 'darwin':
        # macOS: No direct current RSS API without psutil
        # Fall back to peak RSS (ru_maxrss)
        if not _CURRENT_RSS_WARNING_LOGGED:
            print("Warning: macOS does not support current RSS without psutil. "
                  "Average memory will equal peak memory. Install psutil for accurate tracking.")
            _CURRENT_RSS_WARNING_LOGGED = True
        return _get_peak_memory_bytes()
    
    elif sys.platform == 'win32':
        # Windows: Try psutil if available
        try:
            import psutil
            process = psutil.Process()
            return float(process.memory_info().rss)
        except ImportError:
            if not _CURRENT_RSS_WARNING_LOGGED:
                print("Warning: psutil not available on Windows for current RSS. "
                      "Average memory tracking unavailable.")
                _CURRENT_RSS_WARNING_LOGGED = True
            return None
        except Exception as exc:
            if not _CURRENT_RSS_WARNING_LOGGED:
                print(f"Warning: Failed to get current RSS via psutil: {exc}")
                _CURRENT_RSS_WARNING_LOGGED = True
            return None
    
    # Unknown platform
    if not _CURRENT_RSS_WARNING_LOGGED:
        print(f"Warning: Platform {sys.platform} not supported for current RSS tracking.")
        _CURRENT_RSS_WARNING_LOGGED = True
    return None


def _sample_memory_continuously(
    stop_event: threading.Event,
    memory_samples: list,
    sample_interval: float = 0.1
) -> None:
    """Background thread function to sample memory usage continuously.
    
    Samples current RSS (not peak) for accurate average memory calculation.
    On platforms where current RSS is unavailable, samples will be None.
    
    Parameters
    ----------
    stop_event : threading.Event
        Event to signal when to stop sampling
    memory_samples : list
        List to store memory samples in bytes (float or None)
    sample_interval : float
        Time interval between samples in seconds (default: 0.1)
    """
    while not stop_event.is_set():
        current_rss_bytes = _get_current_rss_bytes()
        memory_samples.append(current_rss_bytes)
        time.sleep(sample_interval)


def _track_memory_stats(baseline_bytes: float) -> tuple[float, float]:
    """Start background memory tracking and return peak and average memory.
    
    Returns peak and average memory delta from baseline in MB.
    This is a context manager approach - use with start/stop.
    
    Parameters
    ----------
    baseline_bytes : float
        Baseline memory in bytes before execution
        
    Returns
    -------
    tuple[float, float]
        (peak_memory_mb, avg_memory_mb) - both as delta from baseline
    """
    # This is a helper that will be used by reference methods
    # Actual implementation is in the reference method functions
    pass


def _subprocess_worker(pipe, func, args, kwargs):  # type: ignore[override]
    try:
        result = func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - exercised in error paths
        exc_type = exc.__class__
        pipe.send(("error", exc_type.__module__, exc_type.__name__, exc.args, traceback.format_exc()))
    else:
        pipe.send(("ok", result))
    finally:
        pipe.close()


def _run_in_subprocess(func, *args, **kwargs):
    """Execute ``func`` in a fresh process and return its result."""

    available = mp.get_all_start_methods()
    method = 'fork' if 'fork' in available else 'spawn'
    ctx = mp.get_context(method)
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    process = ctx.Process(target=_subprocess_worker, args=(child_conn, func, args, kwargs))
    process.start()
    message = parent_conn.recv()
    process.join()

    status = message[0]
    if status == 'ok':
        return message[1]

    _, module, name, exc_args, formatted = message
    try:
        exc_module = importlib.import_module(module)
        exc_type = getattr(exc_module, name)
    except Exception:  # pragma: no cover - defensive
        raise RuntimeError(formatted)
    raise exc_type(*exc_args)


def _load_into_memory(path: Path, use_cache: bool = True):
    """Load dataset into memory, keeping sparse format if possible.
    
    Parameters
    ----------
    path : Path
        Path to the h5ad file
    use_cache : bool
        If True, cache the loaded data to avoid reloading for subsequent calls.
        Cache is keyed by absolute path.
    
    Returns
    -------
    ad.AnnData
        Dataset with matrix in memory (sparse if original was sparse)
    """
    import anndata as ad
    import scipy.sparse as sp
    global _SCANPY_REFERENCE_CACHE
    
    cache_key = str(path.resolve())
    
    if use_cache and cache_key in _SCANPY_REFERENCE_CACHE:
        return _SCANPY_REFERENCE_CACHE[cache_key]
    
    adata = ad.read_h5ad(str(path))
    
    # Keep sparse matrices sparse to save memory
    # Only convert to dense if the matrix is already dense
    if sp.issparse(adata.X):
        # Ensure CSR format for Scanpy compatibility
        if not sp.isspmatrix_csr(adata.X):
            adata.X = adata.X.tocsr()
        # Ensure proper dtype
        if adata.X.dtype != np.float32:
            adata.X = adata.X.astype(np.float32)
    else:
        # Already dense, just ensure proper type
        adata.X = np.asarray(adata.X, dtype=np.float64)
    
    if use_cache:
        _SCANPY_REFERENCE_CACHE[cache_key] = adata
    
    return adata


def _normalise_metric_array(values: object, length: int) -> np.ndarray:
    if length == 0:
        return np.array([], dtype=float)
    if values is None:
        return np.full(length, np.nan, dtype=float)
    array = np.asarray(values)
    if array.ndim == 0:
        return np.full(length, float(array), dtype=float)
    array = array.reshape(-1)
    if array.size >= length:
        return array[:length].astype(float)
    padded = np.full(length, np.nan, dtype=float)
    padded[: array.size] = array.astype(float)
    return padded


def _stream_results_to_frame(results: Mapping[str, object]) -> pd.DataFrame:
    columns = ["perturbation", "gene", "effect_size", "statistic", "pvalue"]
    frames: list[pd.DataFrame] = []
    for label, result in results.items():
        genes = getattr(result, "genes", None)
        if genes is None:
            continue
        gene_index = pd.Index(genes).astype(str)
        n_genes = len(gene_index)
        if n_genes == 0:
            continue
        data = {
            "perturbation": [str(label)] * n_genes,
            "gene": gene_index,
            "effect_size": _normalise_metric_array(getattr(result, "effect_size", None), n_genes),
            "statistic": _normalise_metric_array(getattr(result, "statistic", None), n_genes),
            "pvalue": _normalise_metric_array(getattr(result, "pvalue", None), n_genes),
        }
        frames.append(pd.DataFrame(data))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=columns)


def run_scanpy_qc(
    dataset_path: Path,
    *,
    min_genes: int,
    min_cells_per_perturbation: int,
    min_cells_per_gene: int,
    perturbation_column: str,
    control_label: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run Scanpy QC pipeline and save filtered dataset."""
    import scanpy as sc
    import scipy.sparse as sp
    
    adata = sc.read_h5ad(str(dataset_path))
    
    # Ensure CSR
    if sp.issparse(adata.X) and not sp.isspmatrix_csr(adata.X):
        adata.X = adata.X.tocsr()
        
    # Filter cells
    if min_genes > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        
    # Filter perturbations
    if min_cells_per_perturbation > 0:
        labels = adata.obs[perturbation_column].astype(str)
        counts = labels.value_counts()
        keep = labels.eq(control_label) | counts.loc[labels].ge(min_cells_per_perturbation).to_numpy()
        adata = adata[keep].copy()
        
    # Filter genes
    if min_cells_per_gene > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
        
    # Save result
    output_path = output_dir / "scanpy_qc_filtered.h5ad"
    adata.write_h5ad(output_path)
    
    return {
        "result_path": str(output_path),
        "cells_kept": adata.n_obs,
        "genes_kept": adata.n_vars,
    }

def run_scanpy_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    method: str,
    output_dir: Path,
    preprocess: bool = True,
) -> Dict[str, Any]:
    """Run Scanpy DE on filtered dataset."""
    import scanpy as sc
    
    adata = sc.read_h5ad(str(dataset_path))
    
    # Preprocessing
    if preprocess:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    
    # DE
    sc.tl.rank_genes_groups(
        adata,
        groupby=perturbation_column,
        method=method,
        reference=control_label,
        n_genes=adata.n_vars,
    )
    
    df = sc.get.rank_genes_groups_df(adata, None)
    
    # Rename 'group' to perturbation_column if present
    if "group" in df.columns:
        df = df.rename(columns={"group": perturbation_column})
        
    output_path = output_dir / f"scanpy_de_{method.replace('-', '_')}.csv"
    df.to_csv(output_path, index=False)
    
    return {
        "result_path": str(output_path),
        "groups": len(df[perturbation_column].unique()) if not df.empty else 0,
    }

def run_edger_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    output_dir: Path,
    n_jobs: int | None = None,
) -> Dict[str, Any]:
    """Run edgeR DE on filtered dataset."""
    import os
    from benchmarking.env_config import get_global_env_config
    
    # Configure R
    env_config = get_global_env_config()
    r_home = env_config.r_home if env_config else None
    if r_home is None and 'R_HOME' not in os.environ:
        r_home = '/data/miniforge3/envs/pert/lib/R'
    configure_r_environment(r_home)
    
    import scanpy as sc
    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    
    adata = sc.read_h5ad(str(dataset_path))
    
    groups = adata.obs[perturbation_column].astype(str).values
    unique_groups = [g for g in np.unique(groups) if g != control_label]
    
    if not unique_groups:
        raise ValueError("No perturbation groups found")
        
    # R setup
    n_threads = n_jobs if n_jobs is not None and n_jobs > 0 else 1
    ro.r('library(edgeR)')
    ro.r(f'''
    tryCatch({{
        library(RhpcBLASctl)
        blas_set_num_threads({n_threads})
        omp_set_num_threads({n_threads})
    }}, error=function(e) {{ }})
    ''')
    
    # Convert data
    counts = adata.X.T
    if hasattr(counts, 'toarray'):
        counts = counts.toarray()
        
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv['counts'] = counts
        ro.globalenv['groups'] = ro.StrVector(groups)
        ro.globalenv['gene_names'] = ro.StrVector(adata.var_names)
        
        ro.r('''
        rownames(counts) <- gene_names
        y <- DGEList(counts=counts, group=groups)
        y <- calcNormFactors(y)
        design <- model.matrix(~0 + groups)
        colnames(design) <- gsub("groups", "", colnames(design))
        y <- estimateDisp(y, design)
        fit <- glmQLFit(y, design)
        ''')
        
        all_results = []
        for group in unique_groups:
            ro.globalenv['target_group'] = group
            ro.globalenv['control'] = control_label
            
            ro.r('''
            contrast_vec <- makeContrasts(
                contrasts = paste0(target_group, "-", control),
                levels = design
            )
            lrt <- glmQLFTest(fit, contrast=contrast_vec)
            ''')
            
            genes = np.array(ro.r('rownames(lrt$table)'))
            logFC = np.array(ro.r('lrt$table$logFC'))
            logCPM = np.array(ro.r('lrt$table$logCPM'))
            F_stat = np.array(ro.r('lrt$table$F'))
            PValue = np.array(ro.r('lrt$table$PValue'))
            FDR = np.array(ro.r('p.adjust(lrt$table$PValue, method="BH")'))
            
            all_results.append(pd.DataFrame({
                'gene': genes,
                'logFC': logFC,
                'logCPM': logCPM,
                'F': F_stat,
                'PValue': PValue,
                'FDR': FDR,
                'perturbation': group
            }))
            
    final_results = pd.concat(all_results, ignore_index=True)
    output_path = output_dir / "edger_de_glm.csv"
    final_results.to_csv(output_path, index=False)
    
    return {
        "result_path": str(output_path),
        "groups": len(unique_groups),
    }

def run_pertpy_de(
    dataset_path: Path,
    *,
    perturbation_column: str,
    control_label: str,
    backend: str,
    output_dir: Path,
    n_jobs: int | None = None,
) -> Dict[str, Any]:
    """Run Pertpy DE on filtered dataset."""
    import pertpy as pt
    import scanpy as sc
    
    adata = sc.read_h5ad(str(dataset_path))
    
    # Resolve runner
    module = getattr(pt, "tools", None)
    candidate_modules = [
        getattr(module, "differential_expression", None),
        module,
    ]
    try:
        candidate_modules.append(import_module("pertpy.tools._differential_gene_expression"))
    except Exception:
        pass
        
    runner = None
    for candidate in candidate_modules:
        if candidate:
            runner = _resolve_pertpy_runner(candidate, backend) or _resolve_pertpy_class_runner(candidate, backend)
            if runner: break
            
    if not runner:
        raise ValueError(f"Pertpy runner {backend} not found")
        
    # Run
    kwargs = {"groupby": perturbation_column, "control": control_label}
    if n_jobs: kwargs["n_cpus"] = n_jobs
    
    try:
        result = runner(adata, **kwargs)
    except TypeError:
        # Fallback for different API
        kwargs = {"group_key": perturbation_column, "control": control_label}
        if n_jobs: kwargs["n_cpus"] = n_jobs
        result = runner(adata, **kwargs)
        
    df = _convert_reference_result_to_dataframe(result)
    
    # Rename 'contrast' to perturbation_column if present (PyDESeq2)
    if df is not None and "contrast" in df.columns and perturbation_column not in df.columns:
        df = df.rename(columns={"contrast": perturbation_column})
        
    output_path = output_dir / f"pertpy_de_{backend}.csv"
    if df is not None:
        df.to_csv(output_path, index=False)
    
    return {
        "result_path": str(output_path),
        "groups": len(df[perturbation_column].unique()) if df is not None and "perturbation" in df.columns else 0,
    }

# End of scanpy validation utilities
# ============================================================================


@dataclass
class BenchmarkMethod:
    """Description of a method that should be benchmarked."""

    name: str
    description: str
    function: Callable[..., Any]
    kwargs: Dict[str, Any]
    summary: Callable[[Any, Dict[str, Any]], Dict[str, Any]]


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
    skip_existing: bool = True  # Skip methods with existing output files
    environment_config: Optional[EnvironmentConfig] = None
    chunk_size: Optional[int] = None  # Override chunk size for all operations

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

        # Parse environment configuration
        env_data = data.get("environment_config", {})
        environment_config = None
        if env_data:
            environment_config = EnvironmentConfig(
                r_home=env_data.get("r_home"),
                default_n_cores=env_data.get("default_n_cores"),
            )

        force_restandardize = data.get("force_restandardize", False)
        adaptive_qc_mode = data.get("adaptive_qc_mode", "conservative")
        skip_existing = data.get("skip_existing", True)
        chunk_size = data.get("chunk_size")

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
            skip_existing=skip_existing,
            environment_config=environment_config,
            chunk_size=chunk_size,
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
    stream_runtime_seconds: Optional[float] = None
    reference_runtime_seconds: Optional[float] = None
    stream_peak_memory_mb: Optional[float] = None
    stream_avg_memory_mb: Optional[float] = None
    reference_peak_memory_mb: Optional[float] = None
    reference_avg_memory_mb: Optional[float] = None

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

_STATUS_ORDER = ["success", "memory_limit", "timeout", "error", "unknown"]


def _get_expected_output_path(method_name: str, output_dir: Path, data_name: str = None) -> Path | None:
    """Get the expected output path for a benchmark method.
    
    Parameters
    ----------
    method_name : str
        Name of the benchmark method
    output_dir : Path
        Output directory for the dataset
    data_name : str, optional
        Data name prefix (e.g., 'qc', 'de', 'pb') - not used in current implementation
        
    Returns
    -------
    Path | None
        Expected output file path, or None if cannot be determined
    """
    # Phase-based directories
    preprocessing_dir = output_dir / "preprocessing"
    de_dir = output_dir / "de"
    
    # CRISPYx methods with module prefix
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
    
    # Scanpy methods with module prefix
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


def _save_method_result(method_name: str, row_dict: Dict[str, Any], output_dir: Path) -> None:
    """Save individual method benchmark result to cache.
    
    Parameters
    ----------
    method_name : str
        Name of the benchmark method
    row_dict : Dict[str, Any]
        Dictionary containing benchmark results (status, runtime, memory, etc.)
    output_dir : Path
        Output directory for the dataset
    """
    cache_dir = output_dir / ".benchmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{method_name}.json"
    temp_file = cache_dir / f".{method_name}.json.tmp"
    
    try:
        # Convert non-serializable types to JSON-compatible formats
        serializable_dict = {}
        for key, value in row_dict.items():
            if pd.isna(value):
                serializable_dict[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                serializable_dict[key] = float(value)
            elif isinstance(value, (Path,)):
                serializable_dict[key] = str(value)
            else:
                serializable_dict[key] = value
        
        # Atomic write: write to temp file, then rename
        with temp_file.open('w') as f:
            json.dump(serializable_dict, f, indent=2, sort_keys=True)
        temp_file.rename(cache_file)
    except Exception as exc:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        # Log warning but don't fail the benchmark
        print(f"Warning: Failed to save cache for {method_name}: {exc}")


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
        with cache_file.open('r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Warning: Corrupted cache file for {method_name}, will re-run: {exc}")
        # Delete corrupted cache file
        try:
            cache_file.unlink()
        except Exception:
            pass
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
        # Skip config.json, temp files, and comparison cache files
        if (cache_file.name in ("config.json", ".config.json.tmp") or 
            cache_file.name.startswith(".") or
            cache_file.name.endswith("_comparison.json")):
            continue
        
        try:
            with cache_file.open('r') as f:
                result = json.load(f)
                # Only add results that have a method field (exclude comparison metadata)
                if "method" in result:
                    cached_results.append(result)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: Skipping corrupted cache file {cache_file.name}: {exc}")
            continue
    
    return cached_results


def _save_cache_config(output_dir: Path, qc_params: Optional[Dict[str, Any]], standardized_path: str) -> None:
    """Save cache configuration for validation.
    
    Parameters
    ----------
    output_dir : Path
        Output directory for the dataset
    qc_params : Optional[Dict[str, Any]]
        QC parameters used (or None if adaptive)
    standardized_path : str
        Path to standardized dataset
    """
    cache_dir = output_dir / ".benchmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = cache_dir / "config.json"
    temp_file = cache_dir / ".config.json.tmp"
    
    config = {
        "qc_params": qc_params,
        "standardized_dataset_path": standardized_path,
        "timestamp": time.time(),
    }
    
    try:
        with temp_file.open('w') as f:
            json.dump(config, f, indent=2, sort_keys=True)
        temp_file.rename(config_file)
    except Exception as exc:
        if temp_file.exists():
            temp_file.unlink()
        print(f"Warning: Failed to save cache config: {exc}")


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
        with config_file.open('r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _invalidate_cache(output_dir: Path, reason: str = "") -> None:
    """Clear the benchmark cache directory.
    
    Parameters
    ----------
    output_dir : Path
        Output directory for the dataset
    reason : str
        Reason for invalidation (for logging)
    """
    cache_dir = output_dir / ".benchmark_cache"
    
    if not cache_dir.exists():
        return
    
    if reason:
        print(f"  Invalidating cache: {reason}")
    
    # Remove all files in cache directory
    for cache_file in cache_dir.glob("*"):
        try:
            if cache_file.is_file():
                cache_file.unlink()
        except Exception as exc:
            print(f"Warning: Failed to remove cache file {cache_file.name}: {exc}")


def _check_output_exists(method_name: str, output_dir: Path) -> bool:
    """Check if output file for a method already exists.
    
    Parameters
    ----------
    method_name : str
        Name of the benchmark method
    output_dir : Path
        Output directory for the dataset
        
    Returns
    -------
    bool
        True if output exists and is non-empty, or if valid cache exists
    """
    # First check if data output exists
    expected_path = _get_expected_output_path(method_name, output_dir)
    if expected_path is not None and expected_path.exists():
        # Check file is non-empty (at least 1KB for h5ad files, any size for json)
        min_size = 1 if expected_path.suffix == '.json' else 1024
        if expected_path.stat().st_size >= min_size:
            return True
    
    # If no data output, check if cached benchmark result exists
    cached_result = _load_method_result(method_name, output_dir)
    return cached_result is not None


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
    """Return a stable sort key that groups methods by task prefix."""

    # Extract task order from method name
    name = method.name
    if "_qc_" in name:
        task_order = 0
    elif "_pb_" in name:
        task_order = 1
    elif "_de_" in name:
        task_order = 2
    else:
        task_order = 3
    return (task_order, name)


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
        "stream_peak_memory_mb",
        "stream_avg_memory_mb",
        "reference_peak_memory_mb",
        "reference_avg_memory_mb",
    ]
    numeric_columns.extend(
        key for key in DE_METRIC_KEYS if key not in numeric_columns
    )
    for column in numeric_columns:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

    # Sort by method name
    table = table.sort_values(["method"], kind="stable") if "method" in table.columns else table

    # Logical column grouping for results CSV
    preferred_order = [
        # 1. Method identification
        "method",
        "description",
        "status",
        # 2. Performance metrics
        "runtime_seconds",
        "peak_memory_mb",
        # 3. Data dimensions
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
        # 4. Comparison metadata
        "comparison_category",
        "test_type",
        "reference_tool",
        # 5. Comparison metrics - Effect size
        "effect_pearson_corr",
        "effect_spearman_corr",
        "effect_top_k_overlap",
        "effect_max_abs_diff",
        # 6. Comparison metrics - Statistics
        "statistic_pearson_corr",
        "statistic_spearman_corr",
        "statistic_top_k_overlap",
        "statistic_max_abs_diff",
        # 7. Comparison metrics - P-values
        "pvalue_pearson_corr",
        "pvalue_spearman_corr",
        "pvalue_top_k_overlap",
        "pvalue_max_abs_diff",
        "pvalue_stream_auroc",
        "pvalue_reference_auroc",
        # 8. Detailed comparison data
        "stream_peak_memory_mb",
        "stream_avg_memory_mb",
        "reference_peak_memory_mb",
        "reference_avg_memory_mb",
        "stream_timing_breakdown",
        "reference_timing_breakdown",
        "runtime_diff_seconds",
        "runtime_diff_pct",
        "memory_diff_mb",
        "memory_diff_pct",
        # 9. File paths
        "result_path",
        "streaming_result_path",
        "reference_result_path",
        # 10. Errors
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

    # Remove category-based summaries since categories no longer exist
    summary["categories"] = []

    if "error" in df.columns:
        error_rows = df[df["error"].notna()]
        details = []
        for _, row in error_rows.iterrows():
            details.append(
                {
                    "method": str(row.get("method", "")),
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


def _anndata_to_de_dict_legacy(adata: Any) -> Dict[str, Any]:
    """Convert AnnData with DE results to a dictionary of result objects (legacy version - not used).
    
    This function has been replaced by the updated _anndata_to_de_dict with proper
    sparse matrix handling. Kept for reference only.
    """
    from types import SimpleNamespace
    
    result_dict = {}
    
    # CRISPYx stores results in layers: logfoldchange, statistic, pvalue
    # with obs representing perturbations
    if "logfoldchange" in adata.layers and "pvalue" in adata.layers:
        perturbations = adata.obs["perturbation"].tolist()
        for idx, group in enumerate(perturbations):
            result_dict[group] = SimpleNamespace(
                genes=adata.var_names.to_numpy(),
                effect_size=adata.layers["logfoldchange"][idx, :],
                statistic=adata.layers.get("statistic", [None] * adata.n_vars)[idx, :] if "statistic" in adata.layers else None,
                pvalue=adata.layers["pvalue"][idx, :],
            )
    
    return result_dict


def _anndata_to_de_dict(adata) -> Dict[str, Any]:
    """Convert AnnData with DE results to dictionary format.
    
    Sparse Matrix Handling
    ----------------------
    Properly extracts 1D dense arrays from sparse matrices in adata.X and adata.layers.
    Uses np.asarray() to convert sparse row slices to dense without full matrix densification.
    
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
            # Convert to dense and flatten to 1D
            # For sparse matrices, this extracts only one row (memory efficient)
            return np.asarray(row).flatten()
        except Exception:
            return None
    
    stream_result_dict = {}
    
    # crispyx stores results in layers with obs=perturbations
    # logfoldchange layer is optional (not present for t-test which stores effect in .X)
    if "pvalue" in adata.layers and "perturbation" in adata.obs.columns:
        perturbations = adata.obs["perturbation"].tolist()
        for idx, group in enumerate(perturbations):
            # Determine which statistic layer to use
            if "z_score" in adata.layers:
                statistic_values = _extract_row(adata.layers["z_score"], idx)
            elif "statistic" in adata.layers:
                statistic_values = _extract_row(adata.layers["statistic"], idx)
            else:
                statistic_values = None
            
            # Determine which effect size layer/matrix to use
            if "logfoldchange" in adata.layers:
                effect_size_values = _extract_row(adata.layers["logfoldchange"], idx)
            elif adata.X is not None:
                # t_test stores effect sizes in .X matrix
                effect_size_values = _extract_row(adata.X, idx)
            else:
                effect_size_values = None
            
            stream_result_dict[group] = SimpleNamespace(
                genes=adata.var_names.to_numpy(),
                effect_size=effect_size_values,
                statistic=statistic_values,
                pvalue=_extract_row(adata.layers["pvalue"], idx),
            )
    # Fallback for Scanpy format
    elif "rank_genes_groups" in adata.uns:
        rgg = adata.uns["rank_genes_groups"]
        # Get group names from structured array
        if hasattr(rgg["names"], "dtype") and hasattr(rgg["names"].dtype, "names"):
            groups = list(rgg["names"].dtype.names)
        else:
            groups = []
        
        for group in groups:
            # Use names from rank_genes_groups if available, otherwise fallback
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


def _compare_de_frames(
    streaming: pd.DataFrame, reference: pd.DataFrame
) -> Dict[str, Optional[float]]:
    """Return comparison metrics between streaming and reference DE results."""

    return compute_de_comparison_metrics(streaming, reference)





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


def evaluate_benchmarks(output_dir: Path) -> None:
    """Evaluate benchmark results and generate report."""
    from crispyx.comparison import compute_de_comparison_metrics
    
    results_dir = output_dir / ".benchmark_cache"
    if not results_dir.exists():
        return
        
    # Load all results
    results = []
    for f in results_dir.glob("*.json"):
        if f.name == "config.json": continue
        try:
            with open(f) as fh:
                data = json.load(fh)
                if "summary" in data:
                    # Flatten summary into main dict
                    summary = data.pop("summary")
                    data.update(summary)
                results.append(data)
        except Exception:
            pass
            
    if not results:
        return
        
    df = pd.DataFrame(results)
    
    # Rename max_memory_mb to peak_memory_mb if needed
    if "max_memory_mb" in df.columns and "peak_memory_mb" not in df.columns:
        df = df.rename(columns={"max_memory_mb": "peak_memory_mb"})
    
    # Generate Performance Table
    perf_cols = [
        "method", "status", "elapsed_seconds", "peak_memory_mb", "avg_memory_mb",
        "cells_kept", "genes_kept", "groups"
    ]
    perf_df = df[[c for c in perf_cols if c in df.columns]].copy()
    if "method" in perf_df.columns:
        perf_df = perf_df.sort_values("method")
    
    # Generate Accuracy Table
    accuracy_results = []
    
    # Generate Performance Comparison Table
    perf_comp_results = []
    
    # Define comparisons
    comparisons = [
        # QC
        ("crispyx_qc_filtered", "scanpy_qc_filtered", "qc"),
        # DE GLM
        ("crispyx_de_nb_glm", "edger_de_glm", "de"),
        ("crispyx_de_nb_glm", "pertpy_de_pydeseq2", "de"),
        # DE Tests
        ("crispyx_de_t_test", "scanpy_de_t_test", "de"),
        ("crispyx_de_wilcoxon", "scanpy_de_wilcoxon", "de"),
    ]
    
    for stream_method, ref_method, comp_type in comparisons:
        # Check if both exist and succeeded
        stream_res = df[df["method"] == stream_method]
        ref_res = df[df["method"] == ref_method]
        
        if stream_res.empty or ref_res.empty:
            continue
            
        if stream_res.iloc[0]["status"] != "success" or ref_res.iloc[0]["status"] != "success":
            continue
            
        # Performance Comparison
        s_row = stream_res.iloc[0]
        r_row = ref_res.iloc[0]
        
        s_time = s_row.get("elapsed_seconds", np.nan)
        r_time = r_row.get("elapsed_seconds", np.nan)
        s_mem = s_row.get("peak_memory_mb", np.nan)
        r_mem = r_row.get("peak_memory_mb", np.nan)
        
        comp = {
            "comparison": f"{stream_method} vs {ref_method}",
            "crispyx_time_s": s_time,
            "other_time_s": r_time,
            "time_diff_s": s_time - r_time if pd.notna(s_time) and pd.notna(r_time) else None,
            "time_pct": (s_time / r_time * 100) if pd.notna(s_time) and pd.notna(r_time) and r_time > 0 else None,
            "crispyx_mem_mb": s_mem,
            "other_mem_mb": r_mem,
            "mem_diff_mb": s_mem - r_mem if pd.notna(s_mem) and pd.notna(r_mem) else None,
            "mem_pct": (s_mem / r_mem * 100) if pd.notna(s_mem) and pd.notna(r_mem) and r_mem > 0 else None,
        }
        perf_comp_results.append(comp)
        
        # Load result files
        try:
            stream_path_val = stream_res.iloc[0]["result_path"]
            ref_path_val = ref_res.iloc[0]["result_path"]
            
            if pd.isna(stream_path_val) or pd.isna(ref_path_val):
                print(f"Skipping comparison {stream_method} vs {ref_method}: missing result path")
                continue
                
            stream_path = output_dir / str(stream_path_val)
            ref_path = output_dir / str(ref_path_val)
            
            if comp_type == "qc":
                # Compare cell/gene counts
                acc = {
                    "comparison": f"{stream_method} vs {ref_method}",
                    "cells_diff": float(stream_res.iloc[0]["cells_kept"] - ref_res.iloc[0]["cells_kept"]),
                    "genes_diff": float(stream_res.iloc[0]["genes_kept"] - ref_res.iloc[0]["genes_kept"]),
                }
                accuracy_results.append(acc)
                
            elif comp_type == "de":
                # Load DE results
                import anndata as ad
                
                # Load stream (h5ad)
                stream_adata = ad.read_h5ad(str(stream_path))
                stream_dict = _anndata_to_de_dict(stream_adata)
                stream_df = _streaming_de_to_frame(stream_dict)
                
                # Load ref (csv)
                ref_df = pd.read_csv(ref_path)
                ref_df = _standardise_de_dataframe(ref_df)
                
                # Compute metrics
                metrics = compute_de_comparison_metrics(stream_df, ref_df)
                # Create new dict to avoid type errors
                acc = {"comparison": f"{stream_method} vs {ref_method}"}
                acc.update(metrics)
                accuracy_results.append(acc)
                
        except Exception as e:
            print(f"Error comparing {stream_method} vs {ref_method}: {e}")
            
    # Save tables
    perf_df.to_csv(output_dir / "performance_summary.csv", index=False)
    if accuracy_results:
        acc_df = pd.DataFrame(accuracy_results)
        acc_df.to_csv(output_dir / "accuracy_summary.csv", index=False)
        
    # Generate Markdown
    md = "# Benchmark Results\n\n"
    
    md += "## Performance\n\n"
    md += _frame_to_markdown_table(perf_df)
    md += "\n\n"
    
    if perf_comp_results:
        md += "## Performance Comparison\n\n"
        md += _frame_to_markdown_table(pd.DataFrame(perf_comp_results))
        md += "\n\n"
    
    if accuracy_results:
        md += "## Accuracy\n\n"
        md += _frame_to_markdown_table(pd.DataFrame(accuracy_results))
        md += "\n"
        
    with open(output_dir / "results.md", "w") as f:
        f.write(md)


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


def _summarise_runner_result(result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise result from a standalone runner function."""
    # Handle dictionary result (new format)
    if isinstance(result, dict):
        summary = result.copy()
        if "result_path" in summary:
            summary["result_path"] = _normalise_path(summary["result_path"], context)
        return summary

    # Handle tuple result (old format - if any)
    # Result is tuple: (dataframe, path, error, runtime, peak_mem, avg_mem)
    if isinstance(result, tuple) and len(result) >= 6:
        df, path, error, runtime, peak_mem, avg_mem = result[:6]
        
        summary = {}
        if path:
            summary["result_path"] = _normalise_path(path, context)
            
        if error:
            summary["error"] = error
            
        # Pass through internal metrics
        if runtime is not None:
            summary["internal_runtime_seconds"] = runtime
        if peak_mem is not None:
            summary["internal_peak_memory_mb"] = peak_mem
        if avg_mem is not None:
            summary["internal_avg_memory_mb"] = avg_mem
            
        if df is not None and hasattr(df, "shape"):
            summary["rows"] = df.shape[0]
            summary["columns"] = df.shape[1]
            
            # Extract QC metrics if present in attrs
            if hasattr(df, "attrs"):
                for key in ["cells_kept", "genes_kept", "original_cells", "original_genes"]:
                    if key in df.attrs:
                        summary[key] = df.attrs[key]
                        
        return summary

    return {"error": "Invalid result format"}


def _worker(
    queue: mp.Queue,
    method: BenchmarkMethod,
    context: Dict[str, Any],
    memory_limit: int | None,
    time_limit: int | None,
    n_threads: int = 1,
) -> None:
    """Execute ``method`` with optional resource limits and report the outcome.
    
    Memory Baseline Strategy
    ------------------------
    Captures baseline memory before method execution to ensure fair comparison
    with reference methods. Both peak and average memory are reported as deltas
    from this baseline.
    
    For streaming methods:
    - QC methods: Baseline captures environment + minimal data loading overhead
    - DE methods (t-test, wilcoxon): Baseline includes QC-filtered normalized data in memory
    - nb_glm: Baseline includes QC-filtered count data (not normalized)
    
    This matches reference method baselines which capture after _prepare_reference_anndata
    loads filtered datasets.
    
    Parameters
    ----------
    n_threads : int
        Number of threads to use for BLAS/OpenMP operations. Defaults to 1.
    """

    def _apply_resource_limit(limit_value: int | None, limit_type: int, label: str) -> None:
        """Apply ``resource.setrlimit`` defensively.

        Some platforms (notably macOS) raise ``ValueError`` when attempting to
        reduce both the soft and hard limits in a single call, reporting that
        the current limit exceeds the maximum. To avoid crashing the worker
        process, we first lower the soft limit while keeping the existing hard
        cap, then attempt to lower the hard limit. If the OS still rejects the
        change we fall back to best-effort limits and continue execution.
        """

        if not limit_value or limit_value <= 0:
            return

        desired = int(limit_value)

        try:
            current_soft, current_hard = resource.getrlimit(limit_type)

            # First, clamp the soft limit to the current hard ceiling to avoid
            # ``ValueError: current limit exceeds maximum limit``.
            soft_limit = min(
                desired,
                current_hard if current_hard != resource.RLIM_INFINITY else desired,
            )
            if current_soft != soft_limit:
                resource.setrlimit(limit_type, (soft_limit, current_hard))

            # If possible, also lower the hard limit so we get consistent
            # enforcement even if the OS ignores the soft cap.
            if current_hard == resource.RLIM_INFINITY or desired < current_hard:
                try:
                    resource.setrlimit(limit_type, (soft_limit, soft_limit))
                except ValueError:
                    print(
                        f"Warning: unable to reduce {label} hard limit; "
                        "soft limit applied only.",
                        file=sys.stderr,
                    )
        except (ValueError, OSError) as exc:
            print(
                f"Warning: could not apply {label} limit ({exc}); proceeding without it.",
                file=sys.stderr,
            )

    # Set thread limits for BLAS/OpenMP to control parallelism
    set_thread_env_vars(n_threads)

    _apply_resource_limit(memory_limit, resource.RLIMIT_AS, "virtual memory")
    _apply_resource_limit(time_limit, resource.RLIMIT_CPU, "CPU time")

    # Capture baseline memory before method execution
    # This ensures consistent baseline with reference methods
    baseline_memory_bytes = _get_peak_memory_bytes()

    # Start background memory sampling thread
    stop_event = threading.Event()
    memory_samples = []
    memory_thread = threading.Thread(
        target=_sample_memory_continuously,
        args=(stop_event, memory_samples, 0.1),
        daemon=True
    )
    memory_thread.start()
    
    start = time.perf_counter()
    try:
        result = method.function(**method.kwargs)
        elapsed = time.perf_counter() - start
        
        # Stop memory sampling
        stop_event.set()
        memory_thread.join(timeout=1.0)
        
        # Calculate peak memory as delta from baseline
        peak_memory_mb = _peak_memory_delta_mb(baseline_memory_bytes)
        
        # Calculate average memory from samples (current RSS in bytes) as delta from baseline
        avg_memory_mb = None
        if memory_samples:
            # Filter out None values
            valid_samples = [s for s in memory_samples if s is not None]
            if valid_samples:
                avg_rss_bytes = np.mean(valid_samples)
                avg_memory_mb = max(0.0, (avg_rss_bytes - baseline_memory_bytes) / (1024.0 * 1024.0))
        
        summary = method.summary(result, context)
        queue.put(
            {
                "status": "success",
                "elapsed_seconds": elapsed,
                "max_memory_mb": peak_memory_mb,
                "avg_memory_mb": avg_memory_mb,
                "summary": summary,
            }
        )
    except MemoryError as exc:
        # Stop memory sampling
        stop_event.set()
        memory_thread.join(timeout=1.0)
        
        elapsed = time.perf_counter() - start
        queue.put(
            {
                "status": "memory_limit",
                "elapsed_seconds": elapsed,
                "peak_memory_mb": None,
                "avg_memory_mb": None,
                "error": f"MemoryError: {exc}",
            }
        )
    except Exception as exc:  # pragma: no cover - defensive reporting
        # Stop memory sampling
        stop_event.set()
        memory_thread.join(timeout=1.0)
        
        elapsed = time.perf_counter() - start
        queue.put(
            {
                "status": "error",
                "elapsed_seconds": elapsed,
                "peak_memory_mb": None,
                "avg_memory_mb": None,
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
    # Extract n_jobs/n_cores from method kwargs to set thread limits
    n_threads = method.kwargs.get('n_jobs') or method.kwargs.get('n_cores') or 1
    if n_threads is None or n_threads <= 0:
        n_threads = 1
    
    # Special handling for edgeR: run directly without multiprocessing to avoid R/fork issues
    # Still need to set environment variables for this process
    if 'edger_direct' in method.name.lower():
        set_thread_env_vars(n_threads)
        
        # Capture baseline memory before method execution
        baseline_memory_bytes = _get_peak_memory_bytes()
        
        # Start background memory sampling thread
        stop_event = threading.Event()
        memory_samples = []
        memory_thread = threading.Thread(
            target=_sample_memory_continuously,
            args=(stop_event, memory_samples, 0.1),
            daemon=True
        )
        memory_thread.start()
        
        start = time.perf_counter()
        try:
            result = method.function(**method.kwargs)
            elapsed = time.perf_counter() - start
            
            # Stop memory sampling
            stop_event.set()
            memory_thread.join(timeout=1.0)
            
            # Calculate peak memory as delta from baseline
            peak_memory_mb = _peak_memory_delta_mb(baseline_memory_bytes)
            
            # Calculate average memory from samples (current RSS in bytes) as delta from baseline
            avg_memory_mb = None
            if memory_samples:
                # Filter out None values
                valid_samples = [s for s in memory_samples if s is not None]
                if valid_samples:
                    avg_rss_bytes = np.mean(valid_samples)
                    avg_memory_mb = max(0.0, (avg_rss_bytes - baseline_memory_bytes) / (1024.0 * 1024.0))
            
            summary = method.summary(result, context)
            return {
                "status": "success",
                "elapsed_seconds": elapsed,
                "peak_memory_mb": peak_memory_mb,
                "avg_memory_mb": avg_memory_mb,
                "summary": summary,
            }
        except Exception as exc:
            # Stop memory sampling
            stop_event.set()
            memory_thread.join(timeout=1.0)
            
            elapsed = time.perf_counter() - start
            
            avg_memory_mb = None
            if memory_samples:
                # Filter out None values and convert to MB
                valid_samples = [s for s in memory_samples if s is not None]
                if valid_samples:
                    avg_memory_mb = np.mean(valid_samples) / (1024.0 * 1024.0)
            
            return {
                "status": "error",
                "elapsed_seconds": elapsed,
                "max_memory_mb": None,
                "avg_memory_mb": avg_memory_mb,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
    
    # Use spawn context for R/rpy2 compatibility and to avoid OpenMP fork issues
    # triggered by some scanpy workflows.
    needs_spawn = 'pertpy' in method.name.lower() or 'scanpy' in method.name.lower()
    mp_context = mp.get_context('spawn') if needs_spawn else mp
    
    # Track wall-clock time at parent process level for accuracy
    parent_start_time = time.perf_counter()
    
    queue = mp_context.Queue()
    process = mp_context.Process(
        target=_worker,
        args=(queue, method, context, memory_limit, time_limit, n_threads),
        name=f"benchmark-{method.name}",
    )
    process.start()
    join_timeout = None
    if time_limit and time_limit > 0:
        join_timeout = time_limit + 5
    process.join(timeout=join_timeout)
    
    # Calculate actual wall-clock elapsed time from parent process
    parent_elapsed_time = time.perf_counter() - parent_start_time
    
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "status": "timeout",
            "elapsed_seconds": parent_elapsed_time,
            "peak_memory_mb": None,
            "avg_memory_mb": None,
            "summary": {},
            "error": f"Exceeded time limit of {time_limit} seconds",
        }

    if not queue.empty():
        payload = queue.get()
    else:
        payload = {
            "status": "error",
            "elapsed_seconds": parent_elapsed_time,
            "peak_memory_mb": None,
            "avg_memory_mb": None,
            "error": f"Process exited with code {process.exitcode}",
        }

    payload.setdefault("summary", {})
    
    # Ensure peak_memory_mb is present (already computed as delta in _worker)
    payload.setdefault("peak_memory_mb", None)
    
    # Ensure avg_memory_mb is present (may be None)
    payload.setdefault("avg_memory_mb", None)
    
    # Override elapsed_seconds with parent process timing for accuracy
    # (subprocess timing can be incorrect for methods that spawn external processes)
    payload["elapsed_seconds"] = parent_elapsed_time
    
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
    chunk_size: int | None = None,
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
    chunk_size
        Optional fixed chunk size to use. Overrides qc_params.chunk_size if provided.
    """
    import anndata as ad
    
    # Use provided QC params or calculate adaptively
    if qc_params is None:
        # Calculate adaptive QC parameters
        adata_temp = ad.read_h5ad(dataset_path, backed='r')
        try:
            adaptive_thresholds = calculate_adaptive_qc_thresholds(
                adata_temp, perturbation_column, mode='conservative', chunk_size=chunk_size
            )
            min_genes = adaptive_thresholds['min_genes']
            min_cells_per_perturbation = adaptive_thresholds['min_cells_per_perturbation']
            min_cells_per_gene = adaptive_thresholds['min_cells_per_gene']
            final_chunk_size = adaptive_thresholds['chunk_size']
        finally:
            adata_temp.file.close()
    else:
        min_genes = qc_params.min_genes
        min_cells_per_perturbation = qc_params.min_cells_per_perturbation
        min_cells_per_gene = qc_params.min_cells_per_gene
        final_chunk_size = chunk_size if chunk_size is not None else qc_params.chunk_size

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

    # Create phase-based subdirectories for streaming outputs
    preprocessing_dir = output_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    de_dir = output_dir / "de"
    de_dir.mkdir(parents=True, exist_ok=True)

    # Create preprocessed dataset for t-test benchmarks
    preprocessed_path = preprocessing_dir / f"preprocessed_{dataset_path.name}"
    if not preprocessed_path.exists():
        import scanpy as sc
        import scipy.sparse as sp
        print(f"Generating preprocessed dataset for t-test benchmarks: {preprocessed_path}")
        adata_pp = sc.read_h5ad(dataset_path)
        sc.pp.normalize_total(adata_pp)
        sc.pp.log1p(adata_pp)
        if not sp.issparse(adata_pp.X):
            adata_pp.X = sp.csr_matrix(adata_pp.X)
        adata_pp.write(preprocessed_path)

    methods = {
        "crispyx_qc_filtered": BenchmarkMethod(
            name="crispyx_qc_filtered",
            description="Streaming quality control filters",
            function=quality_control_summary,
            kwargs={
                "path": dataset_path,
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "chunk_size": final_chunk_size,
                **shared_kwargs,
                "output_dir": preprocessing_dir,
                "data_name": "qc_filtered",
            },
            summary=_summarise_quality_control,
        ),
        "crispyx_pb_avg_log": BenchmarkMethod(
            name="crispyx_pb_avg_log",
            description="Average log-normalised expression per perturbation",
            function=compute_average_log_expression,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": preprocessing_dir,
                "data_name": "pb_avg_log",
            },
            summary=_summarise_dataframe,
        ),
        "crispyx_pb_pseudobulk": BenchmarkMethod(
            name="crispyx_pb_pseudobulk",
            description="Pseudo-bulk log fold-change per perturbation",
            function=compute_pseudobulk_expression,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": preprocessing_dir,
                "data_name": "pb_pseudobulk",
            },
            summary=_summarise_dataframe,
        ),
        "crispyx_de_t_test": BenchmarkMethod(
            name="crispyx_de_t_test",
            description="t-test differential expression test",
            function=t_test,
            kwargs={
                "path": preprocessed_path,
                **shared_kwargs,
                "output_dir": de_dir,
                "data_name": "de_t_test",
                "n_jobs": n_cores,
            },
            summary=_summarise_de_mapping,
        ),
        "crispyx_de_wilcoxon": BenchmarkMethod(
            name="crispyx_de_wilcoxon",
            description="Wilcoxon rank-sum differential expression",
            function=wilcoxon_test,
            kwargs={
                "path": preprocessed_path,
                **shared_kwargs,
                "output_dir": de_dir,
                "data_name": "de_wilcoxon",
                "n_jobs": n_cores,
            },
            summary=_summarise_de_mapping,
        ),
        "crispyx_de_nb_glm": BenchmarkMethod(
            name="crispyx_de_nb_glm",
            description="Negative binomial GLM differential expression",
            function=nb_glm_test,
            kwargs={
                "path": dataset_path,
                **shared_kwargs,
                "output_dir": de_dir,
                "data_name": "de_nb_glm",
                "n_jobs": n_cores,
            },
            summary=_summarise_de_mapping,
        ),
        "scanpy_qc_filtered": BenchmarkMethod(
            name="scanpy_qc_filtered",
            description="Quality control using Scanpy",
            function=run_scanpy_qc,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "min_genes": min_genes,
                "min_cells_per_perturbation": min_cells_per_perturbation,
                "min_cells_per_gene": min_cells_per_gene,
                "output_dir": preprocessing_dir,
            },
            summary=_summarise_runner_result,
        ),
        "scanpy_de_t_test": BenchmarkMethod(
            name="scanpy_de_t_test",
            description="Differential expression using Scanpy (t-test)",
            function=run_scanpy_de,
            kwargs={
                "dataset_path": preprocessed_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "method": "t-test",
                "output_dir": de_dir,
                "preprocess": False,
            },
            summary=_summarise_runner_result,
        ),
        "scanpy_de_wilcoxon": BenchmarkMethod(
            name="scanpy_de_wilcoxon",
            description="Differential expression using Scanpy (Wilcoxon)",
            function=run_scanpy_de,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "method": "wilcoxon",
                "output_dir": de_dir,
            },
            summary=_summarise_runner_result,
        ),
        "edger_de_glm": BenchmarkMethod(
            name="edger_de_glm",
            description="Differential expression using edgeR (GLM)",
            function=run_edger_de,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "output_dir": de_dir,
                "n_jobs": n_cores,
            },
            summary=_summarise_runner_result,
        ),
        "pertpy_de_pydeseq2": BenchmarkMethod(
            name="pertpy_de_pydeseq2",
            description="Differential expression using Pertpy (PyDESeq2)",
            function=run_pertpy_de,
            kwargs={
                "dataset_path": dataset_path,
                "perturbation_column": shared_kwargs["perturbation_column"],
                "control_label": shared_kwargs["control_label"],
                "backend": "pydeseq2",
                "output_dir": de_dir,
                "n_jobs": n_cores,
            },
            summary=_summarise_runner_result,
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
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for streaming operations (overrides config and adaptive calculation)",
    )
    
    # Environment configuration arguments
    parser.add_argument(
        "--n-cores",
        type=int,
        default=None,
        help="Number of CPU cores to use for parallel operations (overrides config file)",
    )
    parser.add_argument(
        "--r-home",
        type=str,
        default=None,
        help="Path to R installation directory (overrides config and auto-detection)",
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

    # Remove category-based runtime summary since categories no longer exist

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


def _extract_task_and_test_type(method_name: str) -> tuple[str, str]:
    """Extract task type and test type from method name.
    
    Returns
    -------
    tuple[str, str]
        (task, test_type) where task is 'preprocessing' or 'differential_expression'
        and test_type is like 'qc', 't_test', 'wilcoxon', 'nb_glm'
    """
    # Handle NaN or non-string values
    if not isinstance(method_name, str):
        return ("other", "unknown")
    
    if "_qc_" in method_name or method_name.endswith("_qc_filtered"):
        return ("preprocessing", "qc")
    elif "_pb_avg" in method_name:
        return ("preprocessing", "pb_avg")
    elif "_pb_pseudobulk" in method_name:
        return ("preprocessing", "pb_pseudobulk")
    elif "_de_t_test" in method_name:
        return ("differential_expression", "t_test")
    elif "_de_wilcoxon" in method_name:
        return ("differential_expression", "wilcoxon")
    elif "_de_nb_glm" in method_name or "_de_glm" in method_name:
        return ("differential_expression", "nb_glm")
    elif "_de_pydeseq2" in method_name:
        return ("differential_expression", "pydeseq2")
    else:
        return ("other", "unknown")


def _dataframe_to_markdown(
    df: pd.DataFrame, summary: Optional[Dict[str, Any]] = None
) -> str:
    """Render benchmark results as task-based Markdown tables with comparisons."""

    computed_summary = summary or _compute_aggregate_statistics(df)
    narrative = _format_summary_markdown(computed_summary)

    if df.empty:
        return narrative + "\n| |\n|---|\n"

    sections: list[str] = []
    
    # Separate crispyx methods from reference methods
    crispyx_methods = df[df["method"].str.startswith("crispyx_", na=False)].copy()
    reference_methods = df[~df["method"].str.startswith("crispyx_", na=False)].copy()
    
    # Group by task (preprocessing vs differential_expression)
    for task in ["preprocessing", "differential_expression"]:
        task_crispyx = crispyx_methods[
            crispyx_methods["method"].apply(lambda x: _extract_task_and_test_type(x)[0] == task)
        ].copy()
        task_reference = reference_methods[
            reference_methods["method"].apply(lambda x: _extract_task_and_test_type(x)[0] == task)
        ].copy()
        
        if task_crispyx.empty and task_reference.empty:
            continue
        
        # Create task header
        task_title = "Preprocessing" if task == "preprocessing" else "Differential Expression"
        sections.append(f"## {task_title}\n")
        
        # Table 1: crispyx basic info
        if not task_crispyx.empty:
            sections.append("### crispyx Methods\n")
            
            # Select relevant columns for crispyx table
            crispyx_cols = ["method", "description", "status", "runtime_seconds", "peak_memory_mb"]
            if task == "preprocessing":
                crispyx_cols.extend(["cells_kept", "genes_kept", "result_path"])
            else:  # differential_expression
                crispyx_cols.extend(["groups", "genes", "result_path"])
            
            # Keep only columns that exist
            crispyx_table_cols = [col for col in crispyx_cols if col in task_crispyx.columns]
            crispyx_table = task_crispyx[crispyx_table_cols].copy()
            
            # Sort by test type
            crispyx_table["_sort_key"] = crispyx_table["method"].apply(
                lambda x: _extract_task_and_test_type(x)[1]
            )
            crispyx_table = crispyx_table.sort_values("_sort_key").drop(columns=["_sort_key"])
            
            # Drop all-NA columns
            crispyx_table = crispyx_table.dropna(axis=1, how="all")
            
            sections.append(_frame_to_markdown_table(crispyx_table) + "\n")
        
        # Table 2: Reference comparisons
        if not task_reference.empty:
            sections.append("### Reference Comparisons\n")
            
            # Select relevant columns for comparison table
            comparison_cols = [
                "method", "reference_tool", "test_type", "status",
                "runtime_seconds", "peak_memory_mb"
            ]
            
            # Add comparison-specific metrics
            if task == "preprocessing":
                # Add performance difference columns for preprocessing
                perf_diff_cols = [
                    "runtime_diff_seconds", "runtime_diff_pct",
                    "memory_diff_mb", "memory_diff_pct"
                ]
                comparison_cols.extend([col for col in perf_diff_cols if col in task_reference.columns])
                
                comparison_specific = [
                    "normalization_max_abs_diff", "log1p_max_abs_diff",
                    "avg_log_effect_max_abs_diff", "pseudobulk_effect_max_abs_diff"
                ]
            else:  # differential_expression
                # For DE comparisons, add performance difference columns and correlation metrics
                perf_diff_cols = [
                    "runtime_diff_seconds", "runtime_diff_pct",
                    "stream_peak_memory_mb", "stream_avg_memory_mb",
                    "reference_peak_memory_mb", "reference_avg_memory_mb",
                    "memory_diff_mb", "memory_diff_pct"
                ]
                comparison_cols.extend([col for col in perf_diff_cols if col in task_reference.columns])
                
                comparison_specific = [
                    "effect_pearson_corr", "effect_spearman_corr", "effect_top_k_overlap",
                    "statistic_pearson_corr", "statistic_spearman_corr", "statistic_top_k_overlap",
                    "pvalue_pearson_corr", "pvalue_spearman_corr", "pvalue_top_k_overlap"
                ]
            comparison_cols.extend([col for col in comparison_specific if col in task_reference.columns])
            
            # Add result paths
            comparison_cols.extend(["streaming_result_path", "reference_result_path"])
            
            # Keep only columns that exist
            comparison_table_cols = [col for col in comparison_cols if col in task_reference.columns]
            comparison_table = task_reference[comparison_table_cols].copy()
            
            # Sort by test type then by method
            comparison_table["_sort_key"] = comparison_table["method"].apply(
                lambda x: (_extract_task_and_test_type(x)[1], x)
            )
            comparison_table = comparison_table.sort_values("_sort_key").drop(columns=["_sort_key"])
            
            # Drop all-NA columns
            comparison_table = comparison_table.dropna(axis=1, how="all")
            
            sections.append(_frame_to_markdown_table(comparison_table) + "\n")
    
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
                adata_temp, "perturbation", mode=config.adaptive_qc_mode, chunk_size=config.chunk_size
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
            "chunk_size": config.chunk_size if config.chunk_size is not None else config.qc_params.chunk_size,
        }
    
    # Check cache validity and invalidate if needed
    cached_config = _load_cache_config(config.output_dir)
    should_invalidate = False
    invalidate_reason = ""
    
    if config.force_restandardize:
        should_invalidate = True
        invalidate_reason = "force_restandardize=True"
    elif cached_config is not None:
        # Check if QC params changed
        if cached_config.get("qc_params") != qc_params_used:
            should_invalidate = True
            invalidate_reason = "QC parameters changed"
        # Check if standardized dataset path changed
        elif cached_config.get("standardized_dataset_path") != str(standardized_path):
            should_invalidate = True
            invalidate_reason = "standardized dataset path changed"
    
    if should_invalidate:
        _invalidate_cache(config.output_dir, invalidate_reason)
    
    # Save current config to cache
    _save_cache_config(config.output_dir, qc_params_used, str(standardized_path))

    available_methods = create_benchmark_suite(
        dataset_path=standardized_path,
        output_dir=config.output_dir,
        perturbation_column="perturbation",  # Always 'perturbation' after standardization
        control_label="control",  # Always 'control' after standardization
        gene_name_column=config.gene_name_column,
        qc_params=config.qc_params,  # Will use adaptive if None
        n_cores=config.n_cores,
        chunk_size=config.chunk_size,
    )
    
    methods_to_run = config.methods_to_run
    if methods_to_run:
        selected_names: list[str] = []
        for name in methods_to_run:
            if name in available_methods:
                selected_names.append(name)
                continue

            # Allow shorthand prefixes (e.g., "scanpy" -> all scanpy_* methods)
            prefix_matches = [
                method_name
                for method_name in available_methods
                if method_name.startswith(f"{name}_")
            ]

            if prefix_matches:
                selected_names.extend(prefix_matches)
                continue

            raise ValueError(
                f"Unknown method '{name}'. Available methods: {sorted(available_methods)}"
            )
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
        
        # Check if output already exists (for resuming interrupted benchmarks)
        if config.skip_existing and _check_output_exists(method.name, config.output_dir):
            # Try to load cached benchmark metadata
            cached_result = _load_method_result(method.name, config.output_dir)
            
            if cached_result is not None:
                # Use cached result
                rows.append(cached_result)
                
                # Display cached stats
                runtime = cached_result.get("elapsed_seconds") or cached_result.get("runtime_seconds")
                memory = cached_result.get("max_memory_mb") or cached_result.get("peak_memory_mb")
                
                if not config.quiet:
                    stats_str = ""
                    if runtime is not None:
                        stats_str += f"{runtime:.1f}s"
                    if memory is not None:
                        if stats_str:
                            stats_str += ", "
                        stats_str += f"{memory:.1f}GB"
                    
                    if stats_str:
                        print(f"  ⏭️  Skipping {method.name} (cached: {stats_str})")
                    else:
                        print(f"  ⏭️  Skipping {method.name} (cached)")
                
                # Update progress bar with cached stats
                if show_progress:
                    postfix_dict = {"status": "cached"}
                    if memory is not None:
                        postfix_dict["memory"] = f"{memory:.1f}GB"
                    if runtime is not None:
                        postfix_dict["time"] = f"{runtime:.1f}s"
                    method_iterator.set_postfix(postfix_dict, refresh=False)  # type: ignore
            else:
                # No cache, but output exists - create minimal row
                if not config.quiet:
                    print(f"  ⏭️  Skipping {method.name} (output exists)")
                
                existing_path = _get_expected_output_path(method.name, config.output_dir)
                row = {
                    "method": method.name,
                    "description": method.description,
                    "status": "skipped_existing",
                    "elapsed_seconds": None,
                    "max_memory_mb": None,
                    "result_path": _normalise_path(existing_path, context) if existing_path else None,
                }
                rows.append(row)
                
                # Update progress bar for skipped items
                if show_progress:
                    method_iterator.set_postfix(  # type: ignore
                        status="skipped",
                        memory="--",
                        time="--",
                        refresh=False
                    )
            continue
        
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
            "method": method.name,
            "description": method.description,
            "status": result.get("status"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "max_memory_mb": result.get("max_memory_mb"),
            "avg_memory_mb": result.get("avg_memory_mb"),
        }
        summary = result.get("summary", {})
        if summary:
            row.update(summary)
        if result.get("error"):
            row["error"] = result["error"]
        
        rows.append(row)
        
        # Save result to cache immediately after completion
        _save_method_result(method.name, row, config.output_dir)
    
    # Load any cached results that weren't re-run and merge with new results
    cached_results = _load_cached_results(config.output_dir)
    executed_methods = {row["method"] for row in rows}
    
    # Add cached results for methods that weren't executed this run
    for cached_row in cached_results:
        if cached_row.get("method") not in executed_methods:
            rows.append(cached_row)
    
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
    
    # Initialize global environment configuration from CLI args
    from benchmarking.env_config import set_global_env_config, EnvironmentConfig
    
    env_config = EnvironmentConfig(
        r_home=args.r_home if hasattr(args, 'r_home') else None,
        default_n_cores=args.n_cores if hasattr(args, 'n_cores') else None,
    )
    set_global_env_config(env_config)
    
    # Load configuration from YAML or command-line arguments
    if args.config:
        config_result = BenchmarkConfig.from_yaml(args.config)
        configs = config_result if isinstance(config_result, list) else [config_result]
        
        # Override environment config from YAML with CLI args if provided
        for config in configs if isinstance(configs, list) else [configs]:
            if config.environment_config is None:
                config.environment_config = env_config
            else:
                # CLI args take precedence over YAML
                if args.r_home:
                    config.environment_config.r_home = args.r_home
                if args.n_cores:
                    config.environment_config.default_n_cores = args.n_cores
            
            # Update global config with the final merged config
            set_global_env_config(config.environment_config)
            
            # Also update n_cores in config if specified via CLI
            if args.n_cores and config.n_cores is None:
                config.n_cores = args.n_cores

            # Update chunk_size if specified via CLI
            if hasattr(args, 'chunk_size') and args.chunk_size is not None:
                config.chunk_size = args.chunk_size

            # Allow CLI to override the methods list when using a config file
            if args.methods is not None:
                config.methods_to_run = args.methods
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
        
        # Set memory limit for resource limits
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
            n_cores=args.n_cores if hasattr(args, 'n_cores') else None,
            environment_config=env_config,
            chunk_size=args.chunk_size if hasattr(args, 'chunk_size') else None,
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
        
        # Run evaluation
        evaluate_benchmarks(config.output_dir)
        
        # Evaluate benchmarks (generate comparison tables)
        evaluate_benchmarks(config.output_dir)

        if not config.quiet:
            print(f"\nBenchmark complete for {config.dataset_name}")
            print(f"  Results: {csv_path}")
            print(f"  Summary: {md_path}")
            print(f"  JSON: {summary_json_path}")
            if metadata.get("adaptive_qc"):
                print(f"\n  Adaptive QC parameters saved to summary.json")


if __name__ == "__main__":
    main()
