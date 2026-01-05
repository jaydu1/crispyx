"""Profiling utilities for benchmarking (memory tracking and timing analysis).

This module provides:
1. MemoryTracker class for measuring memory usage during benchmark execution
2. Timing profile functions for analyzing NB-GLM bottlenecks
3. Platform-specific memory measurement APIs

.. note::
    For new code, consider using :class:`crispyx.profiling.Profiler` which
    provides unified timing and memory profiling with additional features
    like visualization. ``MemoryTracker`` is retained for backward compatibility
    with existing benchmarking infrastructure.

Usage (Memory Tracking):
    from benchmarking.tools.profiling import MemoryTracker
    
    with MemoryTracker() as mt:
        # Run benchmark code
        result = expensive_computation()
    
    print(f"Peak memory: {mt.get_peak_mb():.2f} MB")
    print(f"Average memory: {mt.get_average_mb():.2f} MB")

Usage (Timing Profile):
    python -m benchmarking.tools.profiling --mode timing
    python -m benchmarking.tools.profiling --mode compare

Alternative (unified profiler from crispyx):
    from crispyx.profiling import Profiler
    
    with Profiler(timing=True, memory=True, sampling=True) as p:
        p.start("computation")
        result = expensive_computation()
        p.stop("computation")
    
    print(p.get_report())
"""
from __future__ import annotations

import logging
import os
import resource
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

# Re-export unified Profiler for new code
try:
    from crispyx.profiling import Profiler, MemoryProfiler as UnifiedMemoryProfiler
except ImportError:
    # crispyx not installed in editable mode during development
    Profiler = None
    UnifiedMemoryProfiler = None


logger = logging.getLogger(__name__)


# =============================================================================
# Memory Measurement Functions
# =============================================================================

# Global flag to track if current RSS warning has been logged (one-time warning)
_CURRENT_RSS_WARNING_LOGGED = False


def get_peak_memory_bytes() -> float:
    """Return the current process peak RSS in bytes.
    
    Uses getrusage(RUSAGE_SELF).ru_maxrss which tracks the high-water mark
    of resident set size for the process. This includes memory from all
    threads and embedded interpreters (e.g., R via rpy2).
    
    Returns
    -------
    float
        Peak RSS in bytes
        
    Notes
    -----
    - Linux: ru_maxrss is in kilobytes, so we multiply by 1024
    - macOS: ru_maxrss is in bytes
    - This captures memory from embedded R (rpy2) since it runs in-process
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage)  # macOS returns bytes
    return float(usage) * 1024.0  # Linux returns KB


def get_current_rss_bytes() -> Optional[float]:
    """Return the current process RSS in bytes (not peak).
    
    Platform-specific implementation:
    - Linux: Reads /proc/self/statm (VmRSS field) - includes all threads
    - macOS: Uses psutil if available, else falls back to peak RSS
    - Windows: Uses psutil if available
    
    Returns
    -------
    float | None
        Current RSS in bytes, or None if unavailable
        
    Notes
    -----
    On Linux, /proc/self/statm provides the total RSS for the process,
    which includes memory from all threads. This correctly captures memory
    usage when methods use multiple threads/cores (e.g., parallel DE tests,
    embedded R via rpy2 with BLAS parallelism).
    
    On macOS without psutil, this returns peak RSS (same as get_peak_memory_bytes())
    which means average memory will equal peak memory. A warning is logged once.
    """
    global _CURRENT_RSS_WARNING_LOGGED
    
    if sys.platform.startswith('linux'):
        # Linux: Read /proc/self/statm
        # Format: size resident shared text lib data dt
        # We want resident (field 2), which is in pages
        # This includes all threads in the process
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
        # macOS: Try psutil if available for current RSS, else fall back to peak RSS
        try:
            import psutil
            process = psutil.Process()
            return float(process.memory_info().rss)
        except ImportError:
            if not _CURRENT_RSS_WARNING_LOGGED:
                print("Warning: macOS does not support current RSS without psutil. "
                      "Average memory will equal peak memory. Install psutil for accurate tracking.")
                _CURRENT_RSS_WARNING_LOGGED = True
            return get_peak_memory_bytes()
    
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


def get_peak_memory_mb() -> float:
    """Return the current process peak RSS in megabytes.
    
    Convenience function for quick peak memory checks.
    
    Returns
    -------
    float
        Peak RSS in megabytes
    """
    return get_peak_memory_bytes() / (1024.0 * 1024.0)


def peak_memory_delta_mb(baseline_bytes: float) -> float:
    """Return peak memory delta from baseline in megabytes.
    
    Parameters
    ----------
    baseline_bytes : float
        Baseline memory in bytes (from earlier get_peak_memory_bytes() call)
    
    Returns
    -------
    float
        Peak memory delta in MB, or 0.0 if current is less than baseline
    """
    return max(0.0, (get_peak_memory_bytes() - baseline_bytes) / (1024.0 * 1024.0))


# =============================================================================
# MemoryTracker Class
# =============================================================================

class MemoryTracker:
    """Context manager for tracking memory usage during benchmark execution.
    
    Starts a background thread that samples current RSS at regular intervals
    to calculate average memory usage. Also tracks peak memory using getrusage.
    
    The tracker correctly captures memory from all threads in the process,
    including parallel workers (joblib, multiprocessing threads) and embedded
    interpreters like R via rpy2.
    
    Parameters
    ----------
    sample_interval : float
        Time between memory samples in seconds. Default is 0.1 (100ms).
        Smaller intervals give more accurate average but higher overhead.
    
    Attributes
    ----------
    baseline_bytes : float
        Peak RSS at start of tracking
    samples : List[float]
        Memory samples collected during tracking (in bytes)
    
    Examples
    --------
    >>> with MemoryTracker() as mt:
    ...     # Run memory-intensive computation
    ...     data = load_large_dataset()
    ...     result = process(data)
    >>> print(f"Peak: {mt.get_peak_mb():.1f} MB, Avg: {mt.get_average_mb():.1f} MB")
    
    Notes
    -----
    Thread Safety:
        Each MemoryTracker instance has its own sampling thread and sample list.
        Multiple trackers can run simultaneously in different contexts, but a
        single tracker should only be used by one thread (the one that created it).
    
    Multi-threaded Workloads:
        When the tracked code uses multiple threads (e.g., parallel DE tests,
        BLAS operations in numpy/R), the memory tracking correctly captures
        the total process memory including all child threads because:
        - /proc/self/statm reports total process RSS (all threads)
        - psutil.Process().memory_info().rss reports total process RSS
        - getrusage(RUSAGE_SELF).ru_maxrss tracks process peak
    """
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.baseline_bytes: float = 0.0
        self.samples: List[float] = []
        self._stop_event: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started: bool = False
        self._stopped: bool = False
    
    def _sample_loop(self) -> None:
        """Background thread function to sample memory usage continuously."""
        while not self._stop_event.is_set():
            rss = get_current_rss_bytes()
            if rss is not None:
                self.samples.append(rss)
            self._stop_event.wait(self.sample_interval)
    
    def start(self) -> None:
        """Start memory tracking.
        
        Records baseline memory and starts background sampling thread.
        
        Raises
        ------
        RuntimeError
            If tracker has already been started
        """
        if self._started:
            raise RuntimeError("MemoryTracker has already been started")
        
        self._started = True
        self.baseline_bytes = get_peak_memory_bytes()
        self.samples = []
        self._stop_event.clear()
        
        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
            name="MemoryTracker-sampler"
        )
        self._thread.start()
    
    def stop(self) -> None:
        """Stop memory tracking.
        
        Signals the sampling thread to stop and waits for it to finish.
        
        Raises
        ------
        RuntimeError
            If tracker was not started or has already been stopped
        """
        if not self._started:
            raise RuntimeError("MemoryTracker was not started")
        if self._stopped:
            raise RuntimeError("MemoryTracker has already been stopped")
        
        self._stopped = True
        self._stop_event.set()
        
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
    
    def get_peak_mb(self) -> float:
        """Return peak memory usage in megabytes (delta from baseline).
        
        Returns
        -------
        float
            Peak memory usage in MB, relative to baseline at start.
            Returns 0.0 if current peak is less than baseline.
        """
        current_peak = get_peak_memory_bytes()
        delta_bytes = max(0.0, current_peak - self.baseline_bytes)
        return delta_bytes / (1024.0 * 1024.0)
    
    def get_average_mb(self) -> Optional[float]:
        """Return average memory usage in megabytes (delta from baseline).
        
        Returns
        -------
        float | None
            Average memory usage in MB across all samples, relative to baseline.
            Returns None if no samples were collected.
        """
        if not self.samples:
            return None
        
        # Use first sample as baseline for average calculation
        # This is more accurate than peak baseline for average memory
        baseline_for_avg = self.samples[0]
        avg_bytes = sum(self.samples) / len(self.samples)
        delta_bytes = max(0.0, avg_bytes - baseline_for_avg)
        return delta_bytes / (1024.0 * 1024.0)
    
    def get_peak_bytes(self) -> float:
        """Return peak memory usage in bytes (delta from baseline)."""
        return max(0.0, get_peak_memory_bytes() - self.baseline_bytes)
    
    def get_peak_absolute_mb(self) -> float:
        """Return absolute peak memory usage in megabytes.
        
        Unlike get_peak_mb() which returns delta from baseline, this returns
        the total process peak RSS. Use for compatibility with code that
        expects absolute memory values.
        
        Returns
        -------
        float
            Absolute peak RSS in megabytes
        """
        return get_peak_memory_bytes() / (1024.0 * 1024.0)
    
    def get_average_absolute_mb(self) -> Optional[float]:
        """Return absolute average memory usage in megabytes.
        
        Unlike get_average_mb() which returns delta from baseline, this returns
        the average of all memory samples (absolute values).
        
        Returns
        -------
        float | None
            Average absolute RSS in megabytes, or None if no samples collected
        """
        if not self.samples:
            return None
        avg_bytes = sum(self.samples) / len(self.samples)
        return avg_bytes / (1024.0 * 1024.0)
    
    def get_sample_count(self) -> int:
        """Return the number of memory samples collected."""
        return len(self.samples)
    
    def reset(self) -> None:
        """Reset the memory tracker for a new measurement phase.
        
        This clears accumulated samples and re-captures the baseline RSS,
        allowing accurate per-step peak measurement within a single process.
        Useful for sequential operations where you want separate memory
        metrics for each step.
        
        The sampling thread continues running - only the samples and baseline
        are reset.
        
        Raises
        ------
        RuntimeError
            If tracker has not been started or has already been stopped
            
        Examples
        --------
        >>> tracker = MemoryTracker()
        >>> tracker.start()
        >>> # Run first step
        >>> result1 = step1()
        >>> peak1 = tracker.get_peak_mb()
        >>> tracker.reset()  # Reset for second step
        >>> # Run second step
        >>> result2 = step2()
        >>> peak2 = tracker.get_peak_mb()
        >>> tracker.stop()
        """
        if not self._started:
            raise RuntimeError("MemoryTracker has not been started")
        if self._stopped:
            raise RuntimeError("MemoryTracker has already been stopped")
        
        # Clear samples and re-capture baseline
        self.samples.clear()
        self.baseline_bytes = get_peak_memory_bytes()
    
    def __enter__(self) -> "MemoryTracker":
        """Context manager entry - starts tracking."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops tracking."""
        self.stop()


# =============================================================================
# Subprocess Memory Sampling
# =============================================================================

def sample_subprocess_memory(
    pid: int,
    stop_event: threading.Event,
    memory_samples: List[float],
    sample_interval: float = 0.1
) -> None:
    """Background thread function to sample subprocess memory from parent process.
    
    Uses psutil to track memory of a subprocess, which works correctly for
    spawned processes where getrusage resets the peak RSS counter.
    Falls back to /proc/{pid}/statm on Linux if psutil is unavailable.
    
    Parameters
    ----------
    pid : int
        Process ID of the subprocess to monitor
    stop_event : threading.Event
        Event to signal when to stop sampling
    memory_samples : List[float]
        List to store memory samples in bytes
    sample_interval : float
        Time interval between samples in seconds (default: 0.1)
    """
    proc = None
    use_proc_fallback = False
    
    try:
        import psutil
        proc = psutil.Process(pid)
    except ImportError:
        # psutil not available, try /proc fallback on Linux
        if sys.platform.startswith('linux'):
            use_proc_fallback = True
            print(
                "Warning: psutil not installed. Using /proc fallback for memory tracking. "
                "Install psutil for more accurate memory measurements: pip install psutil",
                file=sys.stderr
            )
        else:
            print(
                "Warning: psutil not installed and no fallback available for this platform. "
                "Memory tracking will be unavailable. Install psutil: pip install psutil",
                file=sys.stderr
            )
            return
    except Exception:
        # psutil.NoSuchProcess or other errors
        return
    
    while not stop_event.is_set():
        try:
            if use_proc_fallback:
                # Read RSS from /proc/{pid}/statm (second field is RSS in pages)
                statm_path = f'/proc/{pid}/statm'
                try:
                    with open(statm_path, 'r') as f:
                        fields = f.read().split()
                        if len(fields) >= 2:
                            rss_pages = int(fields[1])
                            page_size = os.sysconf('SC_PAGE_SIZE')
                            memory_samples.append(float(rss_pages * page_size))
                except (FileNotFoundError, ProcessLookupError):
                    # Process terminated
                    break
            elif proc is not None:
                mem_info = proc.memory_info()
                memory_samples.append(float(mem_info.rss))
        except Exception:
            # Process terminated or access denied
            break
        stop_event.wait(sample_interval)


# =============================================================================
# Timing Profile Functions
# =============================================================================

def _load_adamson_subset():
    """Load Adamson_subset data and prepare for profiling.
    
    Returns
    -------
    tuple
        (adata, pert_col, labels, control_label, size_factors, backed_path)
    """
    import numpy as np
    import scanpy as sc
    
    # Load Adamson_subset
    data_path = Path(__file__).parents[1] / ".." / "data" / "Adamson_subset.h5ad"
    data_path = data_path.resolve()
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return None
    
    logger.info(f"Loading data from {data_path}")
    adata = sc.read_h5ad(data_path)
    logger.info(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Identify perturbation column
    pert_col = None
    for col in ["perturbation", "gene", "target", "condition"]:
        if col in adata.obs.columns:
            pert_col = col
            break
    
    if pert_col is None:
        logger.error("Could not find perturbation column")
        logger.info(f"Available columns: {list(adata.obs.columns)}")
        return None
    
    logger.info(f"Using perturbation column: {pert_col}")
    
    # Identify control label
    labels = adata.obs[pert_col].values.astype(str)
    unique_labels = np.unique(labels)
    control_label = None
    for candidate in ["control", "Control", "NT", "non-targeting", "ctrl", "CTRL"]:
        if candidate in unique_labels:
            control_label = candidate
            break
    
    if control_label is None:
        logger.error("Could not find control label")
        logger.info(f"Available labels: {unique_labels[:10]}")
        return None
    
    logger.info(f"Control label: {control_label}")
    
    # Compute simple size factors (total count normalization)
    logger.info("Computing size factors...")
    if hasattr(adata.X, 'toarray'):
        totals = np.asarray(adata.X.sum(axis=1)).flatten()
    else:
        totals = np.sum(adata.X, axis=1)
    size_factors = totals / np.median(totals)
    size_factors = np.maximum(size_factors, 1e-10)
    
    # Save to backed format for streaming
    backed_path = Path("/tmp/adamson_subset_backed.h5ad")
    adata.write(backed_path)
    
    return adata, pert_col, labels, control_label, size_factors, backed_path


def profile_joint_model():
    """Profile the joint NB-GLM model on Adamson_subset.
    
    Runs the joint NB-GLM with timing profiling enabled and outputs
    a detailed breakdown of where time is spent.
    """
    import numpy as np
    from crispyx.glm import estimate_joint_model_lbfgsb
    from crispyx.data import read_backed
    
    result = _load_adamson_subset()
    if result is None:
        return
    
    adata, pert_col, labels, control_label, size_factors, backed_path = result
    
    # Run joint model with timing profiling
    logger.info("=" * 60)
    logger.info("Running joint NB-GLM with timing profiling...")
    logger.info("=" * 60)
    
    backed = read_backed(backed_path)
    try:
        model_result = estimate_joint_model_lbfgsb(
            backed,
            obs_df=adata.obs,
            perturbation_labels=labels,
            control_label=control_label,
            covariate_columns=[],
            size_factors=size_factors,
            chunk_size="auto",
            max_iter=25,
            tol=1e-6,
            dispersion_method="moments",  # Faster for profiling
            shrink_dispersion=True,
            per_comparison_dispersion=True,
            use_map_dispersion=True,
            cook_filter=False,
            lfc_shrinkage_type="none",
            n_jobs=-1,
            profile_memory=False,
            profile_timing=True,  # Enable timing profiling
            size_factor_scope="global",
        )
    finally:
        backed.file.close()
    
    logger.info("=" * 60)
    logger.info("Profiling complete!")
    logger.info(f"Fitted {len(model_result.perturbation_labels)} perturbations")
    logger.info(f"Converged: {model_result.converged.sum()}/{len(model_result.converged)} genes")
    logger.info("=" * 60)
    
    # Show profiling stats if available
    if model_result.profiling_stats:
        logger.info("Profiling stats stored in result.profiling_stats")
    
    # Clean up
    backed_path.unlink(missing_ok=True)


def profile_shared_vs_per_comparison():
    """Compare shared vs per-comparison dispersion accuracy.
    
    Runs the joint NB-GLM model twice - once with shared dispersion and once
    with per-comparison dispersion - and compares the results.
    """
    import numpy as np
    from scipy.stats import spearmanr
    from crispyx.glm import estimate_joint_model_lbfgsb
    from crispyx.data import read_backed
    
    result = _load_adamson_subset()
    if result is None:
        return
    
    adata, pert_col, labels, control_label, size_factors, backed_path = result
    
    # Run with per-comparison dispersion (baseline)
    logger.info("=" * 60)
    logger.info("Running with per_comparison_dispersion=True...")
    logger.info("=" * 60)
    
    backed = read_backed(backed_path)
    t0 = time.perf_counter()
    try:
        result_per = estimate_joint_model_lbfgsb(
            backed,
            obs_df=adata.obs,
            perturbation_labels=labels,
            control_label=control_label,
            covariate_columns=[],
            size_factors=size_factors,
            per_comparison_dispersion=True,
            use_map_dispersion=True,
            profile_timing=True,
        )
    finally:
        backed.file.close()
    t_per = time.perf_counter() - t0
    
    # Run with shared dispersion
    logger.info("=" * 60)
    logger.info("Running with per_comparison_dispersion=False...")
    logger.info("=" * 60)
    
    backed = read_backed(backed_path)
    t0 = time.perf_counter()
    try:
        result_shared = estimate_joint_model_lbfgsb(
            backed,
            obs_df=adata.obs,
            perturbation_labels=labels,
            control_label=control_label,
            covariate_columns=[],
            size_factors=size_factors,
            per_comparison_dispersion=False,
            use_map_dispersion=True,
            profile_timing=True,
        )
    finally:
        backed.file.close()
    t_shared = time.perf_counter() - t0
    
    # Compare results
    logger.info("=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info(f"Time with per-comparison dispersion: {t_per:.2f}s")
    logger.info(f"Time with shared dispersion: {t_shared:.2f}s")
    logger.info(f"Speedup: {t_per/t_shared:.2f}x")
    
    # Compute correlation of LFC estimates
    lfc_per = result_per.beta_perturbation.ravel()
    lfc_shared = result_shared.beta_perturbation.ravel()
    
    # Filter out NaN/Inf values
    valid = np.isfinite(lfc_per) & np.isfinite(lfc_shared)
    corr, pval = spearmanr(lfc_per[valid], lfc_shared[valid])
    logger.info(f"LFC Spearman correlation: {corr:.4f} (p={pval:.2e})")
    
    # Compute correlation of dispersions
    disp_per = result_per.dispersion
    disp_shared = result_shared.dispersion
    valid_disp = np.isfinite(disp_per) & np.isfinite(disp_shared)
    corr_disp, _ = spearmanr(disp_per[valid_disp], disp_shared[valid_disp])
    logger.info(f"Dispersion Spearman correlation: {corr_disp:.4f}")
    
    logger.info("=" * 60)
    if corr >= 0.90:
        logger.info(f"✓ Accuracy >= 90% ({corr:.1%}) - shared dispersion can be default")
    else:
        logger.info(f"✗ Accuracy < 90% ({corr:.1%}) - keep per-comparison as default")
    logger.info("=" * 60)
    
    # Clean up
    backed_path.unlink(missing_ok=True)


def profile_with_memory_tracking():
    """Profile the joint NB-GLM model with both timing and memory tracking.
    
    Uses the unified Profiler class to collect both timing and memory stats.
    """
    import numpy as np
    from crispyx.glm import estimate_joint_model_lbfgsb
    from crispyx.data import read_backed
    
    result = _load_adamson_subset()
    if result is None:
        return
    
    adata, pert_col, labels, control_label, size_factors, backed_path = result
    
    logger.info("=" * 60)
    logger.info("Running joint NB-GLM with full profiling (timing + memory)...")
    logger.info("=" * 60)
    
    backed = read_backed(backed_path)
    try:
        model_result = estimate_joint_model_lbfgsb(
            backed,
            obs_df=adata.obs,
            perturbation_labels=labels,
            control_label=control_label,
            covariate_columns=[],
            size_factors=size_factors,
            chunk_size="auto",
            max_iter=25,
            tol=1e-6,
            dispersion_method="moments",
            shrink_dispersion=True,
            per_comparison_dispersion=True,
            use_map_dispersion=True,
            cook_filter=False,
            lfc_shrinkage_type="none",
            n_jobs=-1,
            profile_memory=True,   # Enable memory profiling
            profile_timing=True,   # Enable timing profiling
            size_factor_scope="global",
        )
    finally:
        backed.file.close()
    
    logger.info("=" * 60)
    logger.info("Profiling complete!")
    logger.info(f"Fitted {len(model_result.perturbation_labels)} perturbations")
    logger.info(f"Converged: {model_result.converged.sum()}/{len(model_result.converged)} genes")
    logger.info("=" * 60)
    
    # Show profiling stats
    if model_result.profiling_stats:
        import json
        logger.info("Profiling stats:")
        logger.info(json.dumps(model_result.profiling_stats, indent=2, default=str))
    
    # Clean up
    backed_path.unlink(missing_ok=True)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for running profiling from command line."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    parser = argparse.ArgumentParser(
        description="Profile NB-GLM timing and memory usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m benchmarking.tools.profiling --mode timing
    python -m benchmarking.tools.profiling --mode compare
    python -m benchmarking.tools.profiling --mode full
        """
    )
    parser.add_argument(
        "--mode",
        choices=["timing", "compare", "full"],
        default="timing",
        help="Mode: 'timing' for detailed timing profile, "
             "'compare' for shared vs per-comparison dispersion, "
             "'full' for both timing and memory profiling",
    )
    args = parser.parse_args()
    
    if args.mode == "timing":
        profile_joint_model()
    elif args.mode == "compare":
        profile_shared_vs_per_comparison()
    else:  # full
        profile_with_memory_tracking()


if __name__ == "__main__":
    main()
