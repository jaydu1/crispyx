#!/usr/bin/env python
"""Docker worker entrypoint for crispyx benchmarking.

This script is the entrypoint for Docker containers running individual benchmark
methods. It reads method configuration from a JSON file, executes the benchmark,
and writes results to an output JSON file.

Usage:
    python docker_worker.py --input /workspace/input.json --output /workspace/output.json

Input JSON format:
    {
        "method_name": "crispyx_de_nb_glm",
        "function_module": "crispyx.de",
        "function_name": "nb_glm_test",
        "kwargs": {...},
        "context": {...}
    }

Output JSON format:
    {
        "status": "success" | "error" | "timeout" | "memory_limit",
        "elapsed_seconds": 123.45,
        "peak_memory_mb": 1024.5,
        "avg_memory_mb": 512.3,
        "result": {...},
        "summary": {...},
        "stdout": "...",
        "stderr": "...",
        "error": null | "error message"
    }
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import sys
import time
import traceback
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

# Ensure workspace is in path
WORKSPACE = Path("/workspace")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

# Import MemoryTracker - handle both running from /app and from /workspace
_script_dir = Path(__file__).parent
if _script_dir.name == "app":
    # Running from /app: import memory.py directly (copied alongside docker_worker.py)
    sys.path.insert(0, str(_script_dir))
    from memory import MemoryTracker
else:
    # Running from /workspace: import from benchmarking package
    from benchmarking.tools.memory import MemoryTracker


def _set_thread_env_vars(n_threads: int) -> None:
    """Set environment variables to control thread parallelism."""
    n_threads_str = str(n_threads)
    os.environ['OMP_NUM_THREADS'] = n_threads_str
    os.environ['MKL_NUM_THREADS'] = n_threads_str
    os.environ['OPENBLAS_NUM_THREADS'] = n_threads_str
    os.environ['NUMBA_NUM_THREADS'] = n_threads_str
    os.environ['VECLIB_MAXIMUM_THREADS'] = n_threads_str


def _import_function(module_name: str, function_name: str):
    """Dynamically import a function from a module."""
    # Handle module path differences between host and container
    # On host, modules might be src.crispyx.xxx, but in container they're crispyx.xxx
    original_name = module_name
    if module_name.startswith("src."):
        module_name = module_name[4:]  # Strip "src." prefix
    
    # Debug: print what we're trying to import
    print(f"DEBUG: Importing module '{original_name}' -> '{module_name}' function '{function_name}'", file=sys.stderr)
    print(f"DEBUG: sys.path[:5] = {sys.path[:5]}", file=sys.stderr)
    module = import_module(module_name)
    return getattr(module, function_name)


def _convert_paths_in_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string paths in kwargs to Path objects where appropriate."""
    result = kwargs.copy()
    
    path_keys = ['path', 'output_dir', 'dataset_path', 'input_path']
    for key in path_keys:
        if key in result and isinstance(result[key], str):
            result[key] = Path(result[key])
    
    return result


def _extract_summary_from_output(output: Any) -> Dict[str, Any]:
    """Extract summary information from benchmark function output.
    
    Handles different output types:
    - dict: Direct result from reference runners (run_scanpy_de, run_edger_de, etc.)
    - QualityControlResult: Has filtered_path, cell_mask, gene_mask attributes
    - DE mapping: Dict of perturbation -> SimpleNamespace with genes, effect_size, etc.
    - Other objects: Check for common attributes
    
    Parameters
    ----------
    output : Any
        The output from the benchmark function
        
    Returns
    -------
    Dict[str, Any]
        Summary dictionary with result_path, groups, genes, cells_kept, genes_kept
    """
    import numpy as np
    from collections.abc import Mapping
    
    summary = {}
    
    # Case 1: Direct dict output (e.g., from reference runners like run_scanpy_de)
    if isinstance(output, dict):
        summary_keys = [
            'result_path', 'shrunk_result_path',  # Both base and shrunk paths
            'groups', 'genes', 'cells_kept', 'genes_kept',
            'import_seconds', 'load_seconds', 'process_seconds', 'save_seconds',
            'base_seconds', 'shrinkage_seconds',  # Timing breakdown for integrated shrinkage
            'joint', 'shrinkage_type',  # Metadata from integrated methods
        ]
        for k in summary_keys:
            if k in output:
                v = output[k]
                if isinstance(v, Path):
                    summary[k] = str(v)
                else:
                    summary[k] = v
        return summary
    
    # Case 2: Mapping but not dict - likely DE result (dict of perturbation -> SimpleNamespace)
    if isinstance(output, Mapping) and not isinstance(output, dict):
        groups = list(output.keys())
        summary['groups'] = len(groups)
        if groups:
            first_result = output[groups[0]]
            if hasattr(first_result, 'genes'):
                genes = getattr(first_result, 'genes')
                if hasattr(genes, '__len__'):
                    summary['genes'] = len(genes)
            # Check for result_path on first result
            if hasattr(first_result, 'result_path'):
                val = getattr(first_result, 'result_path')
                summary['result_path'] = str(val) if isinstance(val, Path) else val
        return summary
    
    # Case 3: Object with attributes (QualityControlResult, other result objects)
    if hasattr(output, '__dict__') or hasattr(output, '__slots__'):
        # QualityControlResult: has filtered_path property, cell_mask, gene_mask
        if hasattr(output, 'filtered_path'):
            try:
                summary['result_path'] = str(output.filtered_path)
            except Exception:
                pass
        
        if hasattr(output, 'cell_mask'):
            try:
                cell_mask = getattr(output, 'cell_mask')
                if cell_mask is not None:
                    summary['cells_kept'] = int(np.asarray(cell_mask).sum())
            except Exception:
                pass
        
        if hasattr(output, 'gene_mask'):
            try:
                gene_mask = getattr(output, 'gene_mask')
                if gene_mask is not None:
                    summary['genes_kept'] = int(np.asarray(gene_mask).sum())
            except Exception:
                pass
        
        # Direct result_path attribute (if not already set)
        if 'result_path' not in summary and hasattr(output, 'result_path'):
            val = getattr(output, 'result_path')
            if val is not None:
                summary['result_path'] = str(val) if isinstance(val, Path) else val
        
        # Other common attributes
        for attr in ['groups', 'genes']:
            if attr not in summary and hasattr(output, attr):
                val = getattr(output, attr)
                if val is not None:
                    if hasattr(val, '__len__') and not isinstance(val, str):
                        summary[attr] = len(val)
                    elif isinstance(val, (int, float)):
                        summary[attr] = val
    
    return summary


@contextlib.contextmanager
def capture_output():
    """Context manager to capture stdout and stderr.
    
    Yields
    ------
    tuple[io.StringIO, io.StringIO]
        (stdout_buffer, stderr_buffer) containing captured output
    """
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Create a tee-like class that writes to both the buffer and the original stream
    class TeeWriter:
        def __init__(self, buffer, original):
            self.buffer = buffer
            self.original = original
        
        def write(self, text):
            self.buffer.write(text)
            self.original.write(text)
        
        def flush(self):
            self.buffer.flush()
            self.original.flush()
    
    try:
        sys.stdout = TeeWriter(stdout_buffer, old_stdout)
        sys.stderr = TeeWriter(stderr_buffer, old_stderr)
        yield stdout_buffer, stderr_buffer
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def execute_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a benchmark method and return results.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - method_name: Name of the benchmark method
        - function_module: Module containing the function
        - function_name: Name of the function to call
        - kwargs: Keyword arguments for the function
        - context: Additional context (dataset info, etc.)
        - n_threads: Number of threads to use (optional)
    
    Returns
    -------
    dict
        Result dictionary with status, timing, memory, stdout, stderr, and output
    """
    method_name = config.get("method_name", "unknown")
    function_module = config["function_module"]
    function_name = config["function_name"]
    kwargs = config.get("kwargs", {})
    context = config.get("context", {})
    n_threads = config.get("n_threads", 1)
    
    # Set thread limits
    _set_thread_env_vars(n_threads)
    
    # Prepare result
    result = {
        "method": method_name,
        "status": "error",
        "elapsed_seconds": None,
        "peak_memory_mb": None,
        "avg_memory_mb": None,
        "result": None,
        "summary": {},
        "stdout": "",
        "stderr": "",
        "error": None,
    }
    
    # Use MemoryTracker for memory measurement
    tracker = MemoryTracker(sample_interval=0.1)
    
    with capture_output() as (stdout_buf, stderr_buf):
        tracker.start()
        start_time = time.perf_counter()
        
        try:
            # Import and execute the function
            func = _import_function(function_module, function_name)
            
            # Convert paths in kwargs
            kwargs = _convert_paths_in_kwargs(kwargs)
            
            # Execute
            gc.collect()
            output = func(**kwargs)
            gc.collect()
            
            elapsed = time.perf_counter() - start_time
            
            # Stop memory tracking
            tracker.stop()
            
            # Get memory stats (using delta-based for docker as it's isolated)
            peak_memory_mb = tracker.get_peak_mb()
            avg_memory_mb = tracker.get_average_mb()
            
            # Extract summary from output
            summary = _extract_summary_from_output(output)
            
            result.update({
                "status": "success",
                "elapsed_seconds": elapsed,
                "peak_memory_mb": peak_memory_mb,
                "avg_memory_mb": avg_memory_mb,
                "summary": summary,
            })
            
        except MemoryError as exc:
            try:
                tracker.stop()
            except RuntimeError:
                pass
            elapsed = time.perf_counter() - start_time
            result.update({
                "status": "memory_limit",
                "elapsed_seconds": elapsed,
                "error": f"MemoryError: {exc}",
            })
            
        except Exception as exc:
            try:
                tracker.stop()
            except RuntimeError:
                pass
            elapsed = time.perf_counter() - start_time
            result.update({
                "status": "error",
                "elapsed_seconds": elapsed,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            })
        
        # Capture stdout/stderr
        result["stdout"] = stdout_buf.getvalue()
        result["stderr"] = stderr_buf.getvalue()
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Docker worker for crispyx benchmarking"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input JSON configuration file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output JSON result file"
    )
    parser.add_argument(
        "--help-format",
        action="store_true",
        help="Show input/output JSON format and exit"
    )
    
    args = parser.parse_args()
    
    if args.help_format:
        print(__doc__)
        sys.exit(0)
    
    # Read input configuration
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(input_path, 'r') as f:
        config = json.load(f)
    
    # Execute benchmark
    result = execute_benchmark(config)
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # Exit with appropriate code
    if result["status"] == "success":
        sys.exit(0)
    elif result["status"] == "memory_limit":
        sys.exit(137)  # Standard OOM exit code
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
