"""Environment configuration utilities for benchmarking."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvironmentConfig:
    """Configuration for execution environment and parallelization.
    
    Attributes
    ----------
    r_home : str | None
        Path to R installation directory. If None, will attempt auto-detection.
    default_n_cores : int | None
        Default number of cores to use for parallel operations. 
        If None, uses os.cpu_count().
    """
    
    r_home: Optional[str] = None
    default_n_cores: Optional[int] = None
    
    def get_n_cores(self) -> int:
        """Return the number of cores to use, with auto-detection fallback."""
        if self.default_n_cores is not None and self.default_n_cores > 0:
            return self.default_n_cores
        return os.cpu_count() or 1


# Global environment configuration singleton
_global_env_config: Optional[EnvironmentConfig] = None


def get_global_env_config() -> EnvironmentConfig:
    """Get the global environment configuration, creating default if needed."""
    global _global_env_config
    if _global_env_config is None:
        _global_env_config = EnvironmentConfig()
    return _global_env_config


def set_global_env_config(config: EnvironmentConfig) -> None:
    """Set the global environment configuration."""
    global _global_env_config
    _global_env_config = config


def detect_r_home() -> Optional[str]:
    """Attempt to auto-detect R installation directory.
    
    Tries the following methods in order:
    1. Check R_HOME environment variable
    2. Run 'R RHOME' command
    3. Try conda environment path patterns
    
    Returns
    -------
    str | None
        Path to R installation, or None if detection failed.
    """
    # Method 1: Check R_HOME environment variable
    r_home = os.environ.get('R_HOME')
    if r_home and os.path.isdir(r_home):
        return r_home
    
    # Method 2: Run 'R RHOME' command
    try:
        result = subprocess.run(
            ['R', 'RHOME'],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            r_home = result.stdout.strip()
            if r_home and os.path.isdir(r_home):
                return r_home
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    
    # Method 3: Try conda environment patterns
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        candidate = os.path.join(conda_prefix, 'lib', 'R')
        if os.path.isdir(candidate):
            return candidate
    
    return None


def configure_r_environment(r_home: Optional[str] = None) -> None:
    """Configure R environment by setting R_HOME.
    
    Parameters
    ----------
    r_home : str | None
        Path to R installation. If None, attempts auto-detection.
        If auto-detection fails, R_HOME is left unchanged.
    
    Notes
    -----
    This should be called before importing rpy2 to ensure proper R initialization.
    """
    if r_home is None:
        # Try auto-detection
        r_home = detect_r_home()
    
    if r_home is not None:
        # Validate that the path exists
        if not os.path.isdir(r_home):
            print(f"Warning: R_HOME path does not exist: {r_home}")
            return
        
        # Only set if not already set, to respect user's environment
        if 'R_HOME' not in os.environ:
            os.environ['R_HOME'] = r_home


def set_thread_env_vars(n_threads: int) -> None:
    """Set environment variables to control thread/core usage for numerical libraries.
    
    Parameters
    ----------
    n_threads : int
        Number of threads to use. Should be >= 1.
    
    Notes
    -----
    Sets the following environment variables:
    - OMP_NUM_THREADS: OpenMP parallelization
    - MKL_NUM_THREADS: Intel MKL BLAS library
    - OPENBLAS_NUM_THREADS: OpenBLAS library
    - NUMEXPR_NUM_THREADS: NumExpr library
    - VECLIB_MAXIMUM_THREADS: macOS Accelerate framework
    - R_THREADS: R threading control
    """
    if n_threads < 1:
        n_threads = 1
    
    n_threads_str = str(n_threads)
    os.environ['OMP_NUM_THREADS'] = n_threads_str
    os.environ['MKL_NUM_THREADS'] = n_threads_str
    os.environ['OPENBLAS_NUM_THREADS'] = n_threads_str
    os.environ['NUMEXPR_NUM_THREADS'] = n_threads_str
    os.environ['VECLIB_MAXIMUM_THREADS'] = n_threads_str
    os.environ['R_THREADS'] = n_threads_str
