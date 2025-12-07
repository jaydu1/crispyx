"""Memory tracking utilities for benchmarking.

This module provides a MemoryTracker class for measuring memory usage during
benchmark execution. It uses platform-specific APIs to capture both peak and
average memory consumption.

Usage:
    from benchmarking.tools.memory import MemoryTracker
    
    with MemoryTracker() as mt:
        # Run benchmark code
        result = expensive_computation()
    
    print(f"Peak memory: {mt.get_peak_mb():.2f} MB")
    print(f"Average memory: {mt.get_average_mb():.2f} MB")
"""
from __future__ import annotations

import os
import resource
import sys
import threading
import time
from typing import List, Optional


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
    import os
    
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
