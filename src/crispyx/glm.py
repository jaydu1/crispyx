"""Generalized linear models utilities for differential expression."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence, Literal, Tuple

import numba as nb
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve, solve as scipy_solve
from scipy.optimize import minimize_scalar, minimize, brentq
from scipy.special import gammaln, digamma, polygamma

# Import Numba kernels from separate module
from ._kernels import (
    gammaln_nb,
    _nb_loglik_grid_numba,
    _accumulate_perturbation_blocks_numba,
    _batch_schur_solve_numba,
    _nb_ll_for_alpha,
    _compute_mle_dispersion_numba,
    _nb_map_grid_search_numba,
    _wls_solve_2x2_numba,
    _irls_batch_numba,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Memory Profiling Instrumentation
# =============================================================================

class MemoryProfiler:
    """Context manager for memory profiling with tracemalloc snapshots.
    
    Usage:
        with MemoryProfiler(enabled=True) as profiler:
            # ... code to profile ...
            profiler.snapshot("after_data_load")
            # ... more code ...
            profiler.snapshot("after_irls")
        
        # Access results
        profiler.get_report()  # Human-readable report
        profiler.get_stats()   # Dict for storage in adata.uns
    """
    
    def __init__(self, enabled: bool = False, top_n: int = 10):
        self.enabled = enabled
        self.top_n = top_n
        self.snapshots: dict[str, tuple] = {}  # label -> (snapshot, timestamp, current_mb, peak_mb)
        self._start_time = None
    
    def __enter__(self):
        if self.enabled:
            import tracemalloc
            import time
            tracemalloc.start()
            self._start_time = time.perf_counter()
            self.snapshot("start")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            import tracemalloc
            self.snapshot("end")
            tracemalloc.stop()
        return False
    
    def snapshot(self, label: str) -> None:
        """Take a memory snapshot at the current point."""
        if not self.enabled:
            return
        import tracemalloc
        import time
        snap = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        self.snapshots[label] = (
            snap,
            time.perf_counter() - self._start_time,
            current / 1024 / 1024,  # MB
            peak / 1024 / 1024,      # MB
        )
        logger.debug(
            f"Memory snapshot '{label}': current={current/1024/1024:.1f}MB, "
            f"peak={peak/1024/1024:.1f}MB"
        )
    
    def get_stats(self) -> dict:
        """Get memory statistics as a dict for storage in adata.uns."""
        if not self.enabled or not self.snapshots:
            return {}
        
        stats = {
            "snapshots": {},
            "peak_memory_mb": 0.0,
            "top_allocations": [],
        }
        
        for label, (snap, timestamp, current_mb, peak_mb) in self.snapshots.items():
            stats["snapshots"][label] = {
                "timestamp_s": round(timestamp, 3),
                "current_mb": round(current_mb, 2),
                "peak_mb": round(peak_mb, 2),
            }
            stats["peak_memory_mb"] = max(stats["peak_memory_mb"], peak_mb)
        
        # Get top allocations from final snapshot
        if "end" in self.snapshots:
            end_snap = self.snapshots["end"][0]
            top_stats = end_snap.statistics("lineno")[:self.top_n]
            stats["top_allocations"] = [
                {
                    "file": str(stat.traceback),
                    "size_mb": round(stat.size / 1024 / 1024, 2),
                    "count": stat.count,
                }
                for stat in top_stats
            ]
        
        return stats
    
    def get_report(self) -> str:
        """Generate a human-readable memory profiling report."""
        if not self.enabled or not self.snapshots:
            return "Memory profiling was not enabled."
        
        lines = ["=" * 60, "Memory Profile Report", "=" * 60]
        
        # Snapshot timeline
        lines.append("\nSnapshot Timeline:")
        lines.append("-" * 40)
        for label, (snap, timestamp, current_mb, peak_mb) in self.snapshots.items():
            lines.append(
                f"  {label:25s} t={timestamp:7.2f}s  "
                f"current={current_mb:8.1f}MB  peak={peak_mb:8.1f}MB"
            )
        
        # Memory differences between consecutive snapshots
        labels = list(self.snapshots.keys())
        if len(labels) > 1:
            lines.append("\nMemory Deltas:")
            lines.append("-" * 40)
            for i in range(1, len(labels)):
                prev_label, curr_label = labels[i-1], labels[i]
                prev_mb = self.snapshots[prev_label][2]
                curr_mb = self.snapshots[curr_label][2]
                delta = curr_mb - prev_mb
                sign = "+" if delta >= 0 else ""
                lines.append(
                    f"  {prev_label} -> {curr_label}: {sign}{delta:.1f}MB"
                )
        
        # Top allocations
        if "end" in self.snapshots:
            lines.append(f"\nTop {self.top_n} Memory Allocations (at end):")
            lines.append("-" * 40)
            end_snap = self.snapshots["end"][0]
            top_stats = end_snap.statistics("lineno")[:self.top_n]
            for i, stat in enumerate(top_stats, 1):
                lines.append(
                    f"  {i:2d}. {stat.size/1024/1024:8.2f}MB  {stat.traceback}"
                )
        
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class NBGLMResult:
    """Result of fitting a negative binomial GLM for a single gene."""

    coef: np.ndarray
    se: np.ndarray
    dispersion: float
    converged: bool
    n_iter: int
    deviance: float
    max_cooks: float | None = None


@dataclass
class NBGLMBatchResult:
    """Result of fitting NB GLM for multiple genes in a batch."""

    coef: np.ndarray  # Shape: (n_genes, n_features)
    se: np.ndarray  # Shape: (n_genes, n_features)
    dispersion: np.ndarray  # Shape: (n_genes,)
    converged: np.ndarray  # Shape: (n_genes,) bool
    n_iter: np.ndarray  # Shape: (n_genes,) int
    deviance: np.ndarray  # Shape: (n_genes,)


@dataclass
class ControlStatisticsCache:
    """Cached statistics for control cells to avoid redundant computation.
    
    When fitting independent NB-GLM models for multiple perturbations, each
    comparison includes the same control cells. This cache precomputes and
    stores control cell contributions to the IRLS normal equations, allowing
    them to be reused across all perturbation comparisons instead of being
    redundantly computed.
    
    The cache stores:
    - Control cell intercept (β₀): baseline log-expression per gene
    - Control dispersion: estimated from control cells only
    - XᵀWX contribution from control cells (for intercept column)
    - XᵀWz contribution from control cells
    - Global dispersion (optional): precomputed MAP dispersion using all cells
    
    Memory optimization: We store control_matrix as dense (not mu/weights)
    since mu and weights change during IRLS and must be recomputed anyway.
    Storing the dense matrix avoids repeated .toarray() calls in workers.
    
    This reduces IRLS complexity from O(n_perturbations × n_control × n_genes × n_iter)
    to O(n_control × n_genes × n_iter) for control-related computations.
    """
    
    # Control cell data (stored as dense for efficiency)
    control_matrix: np.ndarray  # Shape: (n_control, n_genes) - always dense
    control_n: int  # Number of control cells
    control_offset: np.ndarray  # Shape: (n_control,) log size factors
    
    # Precomputed intercept (baseline expression for control)
    beta_intercept: np.ndarray  # Shape: (n_genes,)
    
    # Control dispersion (estimated from control cells only)
    control_dispersion: np.ndarray  # Shape: (n_genes,)
    
    # XᵀWX contribution from control (for intercept only, since X[control, perturbation] = 0)
    # For a simple intercept+perturbation design:
    # XᵀWX[0,0] from control = sum of weights for control cells per gene
    control_xtwx_intercept: np.ndarray  # Shape: (n_genes,)
    
    # XᵀWz contribution from control
    # For intercept: XᵀWz[0] from control = sum of (W * z) for control cells per gene  
    control_xtwz_intercept: np.ndarray  # Shape: (n_genes,)
    
    # Mean expression for control (for dispersion trend fitting)
    control_mean_expr: np.ndarray  # Shape: (n_genes,)
    
    # Expression counts for control cells
    control_expr_counts: np.ndarray  # Shape: (n_genes,)
    
    # Proportion of control cells expressing each gene
    pts_rest: np.ndarray  # Shape: (n_genes,)
    
    # Global MAP dispersion (optional): precomputed using all cells
    # When provided, workers skip per-comparison MAP dispersion computation
    global_dispersion: np.ndarray | None = None  # Shape: (n_genes,)
    global_dispersion_trend: np.ndarray | None = None  # Shape: (n_genes,)


def precompute_control_statistics(
    control_matrix: np.ndarray | sp.csr_matrix,
    control_offset: np.ndarray,
    *,
    max_iter: int = 10,
    tol: float = 1e-6,
    min_mu: float = 0.5,
    dispersion_method: Literal["moments", "cox-reid"] = "moments",
) -> ControlStatisticsCache:
    """Precompute control cell statistics for reuse across perturbation comparisons.
    
    This function fits a simple intercept-only model to control cells to estimate
    the baseline expression level (β₀) per gene. The resulting intercept, mean
    expression, weights, and XᵀWX/XᵀWz contributions are cached for reuse.
    
    Parameters
    ----------
    control_matrix
        Expression matrix for control cells, shape (n_control, n_genes).
    control_offset
        Log size factors for control cells, shape (n_control,).
    max_iter
        Maximum IRLS iterations for intercept estimation.
    tol
        Convergence tolerance.
    min_mu
        Minimum fitted mean value.
    dispersion_method
        Method for dispersion estimation.
        
    Returns
    -------
    ControlStatisticsCache
        Cached statistics for control cells.
    """
    # Check sparsity and determine strategy
    is_sparse = sp.issparse(control_matrix)
    if is_sparse:
        nnz = control_matrix.nnz
        total = control_matrix.shape[0] * control_matrix.shape[1]
        sparsity = 1.0 - (nnz / total) if total > 0 else 0.0
    else:
        nnz = np.count_nonzero(control_matrix)
        sparsity = 1.0 - (nnz / control_matrix.size) if control_matrix.size > 0 else 0.0
    
    if is_sparse:
        control_expr_counts = np.asarray(control_matrix.getnnz(axis=0)).ravel()
        n_control, n_genes = control_matrix.shape
        logger.debug(f"Control matrix sparsity: {sparsity:.1%} ({n_control} cells × {n_genes} genes)")
    else:
        control_expr_counts = np.sum(control_matrix > 0, axis=0)
        n_control, n_genes = control_matrix.shape
    
    offset = np.asarray(control_offset, dtype=np.float64)
    
    # Compute pts_rest (proportion of control cells expressing each gene)
    pts_rest = control_expr_counts / n_control if n_control > 0 else np.zeros(n_genes, dtype=np.float32)
    
    # Compute normalized mean expression - use sparse operations if highly sparse
    if is_sparse and sparsity > 0.8:
        # For highly sparse matrices, use sparse mean
        control_mean_expr = np.asarray(control_matrix.mean(axis=0)).ravel() / np.exp(offset).mean()
        mean_counts = np.asarray(control_matrix.mean(axis=0)).ravel()
    else:
        # Densify for IRLS computations
        if is_sparse:
            Y = np.asarray(control_matrix.toarray(), dtype=np.float64)
        else:
            Y = np.asarray(control_matrix, dtype=np.float64)
        normalized = Y / np.exp(offset)[:, None]
        control_mean_expr = normalized.mean(axis=0)
        mean_counts = Y.mean(axis=0)
    
    # Always densify for IRLS (required for efficient vectorized operations)
    if is_sparse:
        Y = np.asarray(control_matrix.toarray(), dtype=np.float64)
    else:
        Y = np.asarray(control_matrix, dtype=np.float64)
    
    # Initialize intercept: log of mean normalized counts
    mean_offset = np.exp(offset).mean()
    beta_intercept = np.log(np.maximum(mean_counts / mean_offset, 1e-10))
    
    # Initialize dispersion
    alpha = np.full(n_genes, 0.1, dtype=np.float64)
    
    # IRLS for intercept-only model
    log_min_mu = np.log(min_mu)
    offset_col = offset[:, None]
    
    mu = np.empty((n_control, n_genes), dtype=np.float64)
    weights = np.empty_like(mu)
    z = np.empty_like(mu)
    
    for iteration in range(max_iter):
        # Compute mu = exp(β₀ + offset)
        eta = beta_intercept[None, :] + offset_col
        np.clip(eta, log_min_mu, 20.0, out=eta)
        np.exp(eta, out=mu)
        np.maximum(mu, min_mu, out=mu)
        
        # Compute weights: W = μ² / (μ + α * μ²)
        variance = mu + alpha[None, :] * mu * mu
        np.divide(mu * mu, np.maximum(variance, min_mu), out=weights)
        
        # Working response: z = η + (Y - μ) / μ
        resid = Y - mu
        z = eta + resid / np.maximum(mu, min_mu)
        
        # Solve for intercept: β₀ = sum(W * (z - offset)) / sum(W)
        z_centered = z - offset_col
        xtwx = np.sum(weights, axis=0)  # (n_genes,)
        xtwz = np.sum(weights * z_centered, axis=0)  # (n_genes,)
        
        beta_new = xtwz / np.maximum(xtwx, 1e-10)
        
        # Check convergence
        if np.max(np.abs(beta_new - beta_intercept)) < tol:
            beta_intercept = beta_new
            break
        
        beta_intercept = beta_new
        
        # Update dispersion (method of moments)
        resid_sq = resid * resid
        numerator = np.sum((resid_sq - Y) / np.maximum(mu * mu, min_mu), axis=0)
        dof = max(n_control - 1, 1)
        alpha_new = np.clip(numerator / dof, 1e-8, 1e6)
        alpha = np.where(np.isfinite(alpha_new), alpha_new, alpha)
    
    # Final mu and weights
    eta = beta_intercept[None, :] + offset_col
    np.clip(eta, log_min_mu, 20.0, out=eta)
    np.exp(eta, out=mu)
    np.maximum(mu, min_mu, out=mu)
    
    variance = mu + alpha[None, :] * mu * mu
    np.divide(mu * mu, np.maximum(variance, min_mu), out=weights)
    
    # Compute XᵀWX and XᵀWz for control (intercept column only)
    z_centered = eta + (Y - mu) / np.maximum(mu, min_mu) - offset_col
    control_xtwx_intercept = np.sum(weights, axis=0)  # (n_genes,)
    control_xtwz_intercept = np.sum(weights * z_centered, axis=0)  # (n_genes,)
    
    # Free temporary arrays (mu, weights, z_centered) - only Y is needed
    del mu, weights, z_centered, eta
    
    return ControlStatisticsCache(
        control_matrix=Y,  # Store dense matrix directly (avoids repeated .toarray())
        control_n=n_control,
        control_offset=offset,
        beta_intercept=beta_intercept,
        control_dispersion=alpha,
        control_xtwx_intercept=control_xtwx_intercept,
        control_xtwz_intercept=control_xtwz_intercept,
        control_mean_expr=control_mean_expr,
        control_expr_counts=control_expr_counts.astype(np.int32),
        pts_rest=pts_rest.astype(np.float32),
    )


class NBGLMFitter:
    """L-BFGS-B solver for negative binomial GLMs.

    Parameters
    ----------
    design:
        The design matrix with shape ``(n_samples, n_features)``.
    offset:
        Optional log-scale offset (e.g. library size) per sample.
    dispersion:
        Optional dispersion (alpha) for the negative binomial. If ``None`` the
        dispersion is re-estimated at each iteration using a method-of-moments
        update similar to the one used in statsmodels.
    max_iter:
        Maximum number of outer iterations for alternating optimization.
    tol:
        Absolute tolerance on the coefficient updates for convergence.
    poisson_init_iter:
        Maximum number of iterations for the Poisson initialisation stage. If
        set to ``0`` the Poisson warm start is skipped and coefficients are
        initialised at zero.
    ridge_penalty:
        Small diagonal ridge penalty added to the weighted normal equations to
        improve numerical stability. This does not change the estimator when
        the system is well conditioned but prevents failures when the Hessian
        is nearly singular.
    min_mu:
        Lower bound on the fitted mean to avoid issues with extremely small
        predicted counts for lowly expressed genes.
    min_total_count:
        Genes with a total count below this threshold are not fitted (the
        resulting ``NBGLMResult`` will report ``converged=False``).
    dispersion_method:
        Method for estimating dispersion when ``dispersion`` is None:
        - "moments": Method-of-moments (fast but less accurate)
        - "cox-reid": Cox-Reid adjusted profile likelihood (slower but more
          accurate, similar to DESeq2)
    """

    def __init__(
        self,
        design: ArrayLike,
        *,
        offset: ArrayLike | None = None,
        dispersion: float | None = None,
        max_iter: int = 50,
        tol: float = 1e-6,
        poisson_init_iter: int = 20,
        ridge_penalty: float = 1e-6,
        min_mu: float = 0.5,
        min_total_count: float = 1.0,
        compute_cooks: bool = False,
        dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
    ) -> None:
        self.design = np.asarray(design, dtype=np.float64)
        if self.design.ndim != 2:
            raise ValueError("design must be a 2D array")
        self.n_samples, self.n_features = self.design.shape
        self.offset = (
            np.asarray(offset, dtype=np.float64)
            if offset is not None
            else np.zeros(self.n_samples, dtype=np.float64)
        )
        if self.offset.shape != (self.n_samples,):
            raise ValueError("offset must have shape (n_samples,)")
        self.dispersion = dispersion
        self.max_iter = int(max_iter)
        self.tol = tol
        self.poisson_init_iter = int(max(0, poisson_init_iter))
        self.ridge_penalty = ridge_penalty
        self.min_mu = min_mu
        self.min_total_count = min_total_count
        self.compute_cooks = compute_cooks
        self.dispersion_method = dispersion_method

    def fit_gene(self, counts: ArrayLike) -> NBGLMResult:
        """Fit NB GLM for a single gene using L-BFGS-B optimization."""
        y = np.asarray(counts, dtype=np.float64)
        if y.shape != (self.n_samples,):
            raise ValueError("counts must have shape (n_samples,)")
        total = float(y.sum())
        if total < self.min_total_count or not np.isfinite(total):
            zeros = np.zeros(self.n_features, dtype=np.float64)
            return NBGLMResult(
                coef=zeros,
                se=np.full(self.n_features, np.inf, dtype=np.float64),
                dispersion=float("nan"),
                converged=False,
                n_iter=0,
                deviance=float("nan"),
                max_cooks=None,
            )
        
        return self._fit_gene_lbfgsb(y)

    def fit_matrix(self, matrix: ArrayLike, *, batch_size: int | None = None) -> list[NBGLMResult]:
        if sp.issparse(matrix):
            sparse_matrix = sp.csc_matrix(matrix, dtype=np.float64)
            n_genes = sparse_matrix.shape[1]
            batch = batch_size or n_genes
            if sparse_matrix.shape[0] != self.n_samples:
                raise ValueError("matrix must have shape (n_samples, n_genes)")
            results: list[NBGLMResult] = []
            for start in range(0, n_genes, batch):
                end = min(start + batch, n_genes)
                dense_block = np.asarray(sparse_matrix[:, start:end].toarray())
                for col in range(dense_block.shape[1]):
                    results.append(self.fit_gene(dense_block[:, col]))
            return results

        y = np.asarray(matrix, dtype=np.float64)
        if y.ndim != 2 or y.shape[0] != self.n_samples:
            raise ValueError("matrix must have shape (n_samples, n_genes)")

        batch = batch_size or y.shape[1]
        results: list[NBGLMResult] = []
        for start in range(0, y.shape[1], batch):
            end = min(start + batch, y.shape[1])
            block = y[:, start:end]
            for col in range(block.shape[1]):
                results.append(self.fit_gene(block[:, col]))
        return results

    def _fit_gene_lbfgsb(self, y: np.ndarray) -> NBGLMResult:
        """Fit NB GLM using L-BFGS-B optimization (PyDESeq2 style).
        
        This method directly optimizes the negative binomial log-likelihood
        using scipy's L-BFGS-B optimizer, which is the approach used by PyDESeq2.
        It alternates between optimizing coefficients (beta) and dispersion (alpha).
        """
        # Initialize with Poisson warm start
        beta = np.zeros(self.n_features, dtype=np.float64)
        if self.poisson_init_iter > 0:
            beta = self._poisson_warm_start(y, beta.copy())
        
        # Initial dispersion estimate using method of moments
        eta = self.offset + self.design @ beta
        mu = np.exp(np.clip(eta, np.log(self.min_mu), 20.0))
        mu = np.maximum(mu, self.min_mu)
        alpha = self._update_alpha(y, mu, 0.1)
        
        converged = False
        n_iter = 0
        
        # Alternate between optimizing beta and alpha
        for outer_iter in range(self.max_iter):
            # Optimize beta given alpha using L-BFGS-B
            def neg_log_likelihood(beta_vec: np.ndarray) -> float:
                eta = self.offset + self.design @ beta_vec
                mu = np.exp(np.clip(eta, np.log(self.min_mu), 20.0))
                mu = np.maximum(mu, self.min_mu)
                r = 1.0 / max(alpha, 1e-10)
                # NB log-likelihood (using numba-accelerated gammaln)
                ll = np.sum(
                    gammaln_nb(y + r)
                    - gammaln_nb(r)
                    - gammaln_nb(y + 1)
                    + r * np.log(r / (r + mu))
                    + y * np.log(mu / (r + mu + 1e-12))
                )
                return -ll
            
            def gradient(beta_vec: np.ndarray) -> np.ndarray:
                eta = self.offset + self.design @ beta_vec
                mu = np.exp(np.clip(eta, np.log(self.min_mu), 20.0))
                mu = np.maximum(mu, self.min_mu)
                r = 1.0 / max(alpha, 1e-10)
                # Gradient: d(-ll)/d(beta) = X^T @ (mu - y * (1 + r) / (mu + r))
                # Simplified: X^T @ (mu * (y + r) / (mu + r) - y)
                w = (y + r) / (mu + r)
                grad = self.design.T @ (mu * w - y)
                return -grad  # Negative because we're minimizing negative LL
            
            beta_old = beta.copy()
            try:
                result = minimize(
                    neg_log_likelihood,
                    beta,
                    method='L-BFGS-B',
                    jac=gradient,
                    options={'maxiter': 50, 'gtol': self.tol, 'ftol': 1e-10}
                )
                if result.success or result.fun < neg_log_likelihood(beta):
                    beta = result.x
                n_iter = outer_iter + 1
            except Exception:
                pass
            
            # Update mu with new beta
            eta = self.offset + self.design @ beta
            mu = np.exp(np.clip(eta, np.log(self.min_mu), 20.0))
            mu = np.maximum(mu, self.min_mu)
            
            # Update dispersion
            if self.dispersion is not None:
                alpha = self.dispersion
            else:
                alpha_old = alpha
                alpha = self._update_alpha(y, mu, alpha)
                
                # Refine with Cox-Reid if requested
                if self.dispersion_method == "cox-reid":
                    variance = mu + alpha * (mu**2)
                    weights = (mu**2) / np.maximum(variance, self.min_mu)
                    alpha = self.estimate_dispersion_cox_reid(
                        y, mu, weights, initial_alpha=alpha
                    )
            
            # Check convergence
            beta_diff = float(np.max(np.abs(beta - beta_old)))
            if beta_diff < self.tol:
                converged = True
                break
        
        # Compute final statistics
        eta = self.offset + self.design @ beta
        mu = np.exp(np.clip(eta, np.log(self.min_mu), 20.0))
        mu = np.maximum(mu, self.min_mu)
        variance = mu + alpha * (mu**2)
        weights = (mu**2) / np.maximum(variance, self.min_mu)
        
        # Compute covariance matrix (inverse Hessian)
        cov_beta = self._hessian_inverse(weights)
        se = np.sqrt(np.maximum(np.diag(cov_beta), self.min_mu))
        
        # Compute deviance
        deviance = self._compute_deviance(y, mu, alpha)
        
        # Cook's distance if requested
        max_cooks = None
        if self.compute_cooks:
            hat_diag = self._hat_diagonal(weights, cov_beta)
            pearson_resid = (y - mu) / np.sqrt(np.maximum(variance, self.min_mu))
            denom = np.maximum((1.0 - hat_diag) ** 2, self.min_mu)
            cooks = (pearson_resid**2 / max(self.n_features, 1)) * (hat_diag / denom)
            max_cooks = float(np.nanmax(cooks)) if cooks.size else None
        
        return NBGLMResult(
            coef=beta,
            se=se,
            dispersion=alpha,
            converged=converged,
            n_iter=n_iter,
            deviance=deviance,
            max_cooks=max_cooks,
        )

    def _poisson_warm_start(self, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        for _ in range(self.poisson_init_iter):
            eta = self.offset + self.design @ beta
            mu = np.exp(np.clip(eta, a_min=np.log(self.min_mu), a_max=None))
            mu = np.maximum(mu, self.min_mu)
            weights = mu
            z = eta + (y - mu) / np.maximum(mu, self.min_mu)
            working_response = z - self.offset
            beta_new, _ = self._weighted_least_squares(weights, working_response)
            if np.max(np.abs(beta_new - beta)) < self.tol:
                return beta_new
            beta = beta_new
        return beta

    def _weighted_least_squares(self, weights: np.ndarray, y_working: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if weights.shape != (self.n_samples,):
            raise ValueError("weights must have shape (n_samples,)")
        w_sqrt = np.sqrt(np.clip(weights, self.min_mu, None))
        x_weighted = self.design * w_sqrt[:, None]
        z_weighted = y_working * w_sqrt
        xtwx = x_weighted.T @ x_weighted
        if self.ridge_penalty:
            xtwx = xtwx + self.ridge_penalty * np.eye(self.n_features)
        xtwz = x_weighted.T @ z_weighted
        try:
            c, lower = cho_factor(xtwx, overwrite_a=False, check_finite=False)
            beta = cho_solve((c, lower), xtwz, check_finite=False)
            inv_hessian = cho_solve((c, lower), np.eye(self.n_features), check_finite=False)
        except np.linalg.LinAlgError:
            try:
                beta = np.linalg.solve(xtwx, xtwz)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(xtwx) @ xtwz
            try:
                inv_hessian = np.linalg.inv(xtwx)
            except np.linalg.LinAlgError:
                inv_hessian = np.linalg.pinv(xtwx)
        return beta, inv_hessian

    def _hessian_inverse(self, weights: np.ndarray) -> np.ndarray:
        w_sqrt = np.sqrt(np.clip(weights, self.min_mu, None))
        x_weighted = self.design * w_sqrt[:, None]
        xtwx = x_weighted.T @ x_weighted
        if self.ridge_penalty:
            xtwx = xtwx + self.ridge_penalty * np.eye(self.n_features)
        try:
            c, lower = cho_factor(xtwx, overwrite_a=False, check_finite=False)
            inv_hessian = cho_solve((c, lower), np.eye(self.n_features), check_finite=False)
        except np.linalg.LinAlgError:
            inv_hessian = np.linalg.pinv(xtwx)
        return inv_hessian

    def _hat_diagonal(self, weights: np.ndarray, inv_hessian: np.ndarray) -> np.ndarray:
        w_sqrt = np.sqrt(np.clip(weights, self.min_mu, None))
        x_weighted = self.design * w_sqrt[:, None]
        projection = x_weighted @ inv_hessian
        hat = np.sum(x_weighted * projection, axis=1)
        return np.clip(hat, 0.0, 1.0)

    @staticmethod
    def _compute_deviance(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
        mu = np.maximum(mu, 1e-12)
        if alpha <= 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu) - (y - mu), -mu)
            return 2.0 * float(np.nansum(terms))
        r = 1.0 / alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = y * np.log(np.maximum(y, 1e-12) / mu)
            term2 = (y + r) * np.log((y + r) / (mu + r))
            dev = 2.0 * float(np.nansum(term1 - term2))
        return dev

    def _update_alpha(self, y: np.ndarray, mu: np.ndarray, current_alpha: float) -> float:
        # Method-of-moments style update used as a cheap approximation to
        # maximize the profile likelihood for alpha.
        resid = y - mu
        denom = np.maximum(mu**2, self.min_mu)
        numerator = np.sum((resid**2 - y) / denom)
        dof = max(y.size - self.n_features, 1)
        alpha = numerator / dof
        if not np.isfinite(alpha):
            alpha = current_alpha
        alpha = float(np.clip(alpha, 1e-8, 1e6))
        if alpha <= 0:
            alpha = 1e-8
        return alpha

    def estimate_dispersion_cox_reid(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        weights: np.ndarray,
        *,
        initial_alpha: float = 0.1,
        bounds: tuple[float, float] = (1e-8, 1e3),
    ) -> float:
        """Estimate dispersion using Cox-Reid adjusted profile likelihood.
        
        This method maximizes the adjusted profile log-likelihood for the
        dispersion parameter, which includes a bias correction term based on
        the Cox-Reid adjustment. This approach is similar to DESeq2's dispersion
        estimation.
        
        Parameters
        ----------
        y
            Count vector of shape (n_samples,).
        mu
            Current fitted mean values of shape (n_samples,).
        weights
            IRLS weights from the current fit.
        initial_alpha
            Starting value for the optimization.
        bounds
            Lower and upper bounds for the dispersion parameter.
            
        Returns
        -------
        float
            Estimated dispersion parameter.
        """
        n = len(y)
        p = self.n_features
        
        def neg_log_likelihood(alpha: float) -> float:
            if alpha <= 0:
                return np.inf
            r = 1.0 / alpha
            
            # Negative binomial log-likelihood (using numba-accelerated gammaln)
            ll = np.sum(
                gammaln_nb(y + r)
                - gammaln_nb(r)
                - gammaln_nb(y + 1)
                + r * np.log(r / (r + mu))
                + y * np.log(mu / (r + mu + 1e-12))
            )
            
            # Cox-Reid adjustment: -0.5 * log(det(X^T W X))
            # This accounts for the fact that we're profiling over beta
            variance = mu + alpha * (mu**2)
            w = (mu**2) / np.maximum(variance, self.min_mu)
            w_sqrt = np.sqrt(np.clip(w, self.min_mu, None))
            x_weighted = self.design * w_sqrt[:, None]
            xtwx = x_weighted.T @ x_weighted
            try:
                sign, log_det = np.linalg.slogdet(xtwx)
                if sign > 0:
                    ll -= 0.5 * log_det
            except np.linalg.LinAlgError:
                pass
            
            return -ll
        
        # Use bounded optimization
        try:
            result = minimize_scalar(
                neg_log_likelihood,
                bounds=bounds,
                method="bounded",
                options={"xatol": 1e-4, "maxiter": 50},
            )
            if result.success and np.isfinite(result.x):
                return float(np.clip(result.x, bounds[0], bounds[1]))
        except Exception:
            pass
        
        # Fallback to method-of-moments
        return self._update_alpha(y, mu, initial_alpha)


def build_design_matrix(
    obs_frame,
    *,
    covariate_columns: Sequence[str],
    perturbation_indicator: np.ndarray,
    intercept: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Construct a design matrix from covariates and a perturbation indicator.

    Parameters
    ----------
    obs_frame:
        Pandas ``DataFrame`` (preferred) or structured numpy array containing
        covariate columns.
    covariate_columns:
        Columns that should be included in the design matrix. Categorical
        columns are expanded using one-hot encoding (dropping the first level).
    perturbation_indicator:
        Binary array of shape ``(n_samples,)`` marking perturbed cells.
    intercept:
        Whether to prepend an intercept column to the design matrix.

    Returns
    -------
    design:
        The numeric design matrix as a ``numpy.ndarray`` of ``float64``.
    column_names:
        The column names corresponding to the design matrix.
    """

    import pandas as pd

    if not isinstance(obs_frame, pd.DataFrame):
        obs_frame = pd.DataFrame(obs_frame)
    if len(obs_frame) != perturbation_indicator.shape[0]:
        raise ValueError("Number of samples in obs_frame and indicator do not match")
    matrices = []
    column_names: list[str] = []
    if intercept:
        matrices.append(np.ones((len(obs_frame), 1), dtype=np.float64))
        column_names.append("intercept")
    matrices.append(perturbation_indicator.reshape(-1, 1).astype(np.float64))
    column_names.append("perturbation")
    for column in covariate_columns:
        if column not in obs_frame.columns:
            raise KeyError(f"Covariate '{column}' not found in obs_frame")
        series = obs_frame[column]
        if series.dtype.kind in {"O", "U"} or str(series.dtype).startswith("category"):
            dummies = pd.get_dummies(series, prefix=column, drop_first=True, dtype=float)
            if dummies.shape[1] == 0:
                continue
            matrices.append(dummies.to_numpy(dtype=np.float64))
            column_names.extend(dummies.columns.astype(str).tolist())
        else:
            matrices.append(series.to_numpy(dtype=np.float64).reshape(-1, 1))
            column_names.append(str(column))
    design = np.hstack(matrices) if matrices else np.empty((len(obs_frame), 0), dtype=np.float64)
    return design, column_names


def fit_dispersion_trend(
    means: ArrayLike,
    dispersions: ArrayLike,
    *,
    min_mean: float = 0.5,
    fit_type: Literal["parametric", "local", "mean"] = "parametric",
    n_iter: int = 10,
) -> np.ndarray:
    """Fit a smooth mean-dispersion trend using DESeq2/PyDESeq2-style Gamma GLM.
    
    The parametric trend models dispersion as:
        dispersion = asymptDisp + extraPois / mean
    
    This matches PyDESeq2's fitDispersionTrend which uses iteratively reweighted
    least squares (IRLS) with a Gamma family and log link, fitting:
        E[disp] = a0 + a1 / mean
    
    Outliers are iteratively removed based on prediction ratio bounds.
    
    Parameters
    ----------
    means
        Mean expression values per gene (normalized counts).
    dispersions
        Raw genewise dispersion estimates.
    min_mean
        Minimum mean value for fitting (genes below this are excluded).
    fit_type
        Type of trend fitting:
        - "parametric": DESeq2/PyDESeq2-style Gamma GLM (recommended)
        - "local": Weighted local regression (LOWESS-like)
        - "mean": Simple median (fallback)
    n_iter
        Number of iterations for IRLS fitting.
    
    Returns
    -------
    np.ndarray
        Fitted trend values for each gene.
    """
    means_arr = np.asarray(means, dtype=np.float64)
    disp_arr = np.asarray(dispersions, dtype=np.float64)
    
    # For fitting, use all genes with valid dispersion (PyDESeq2 style)
    # Don't filter by min_mean - only exclude genes with inf covariate (mean=0)
    # or invalid dispersion values
    valid_for_fit = (
        np.isfinite(means_arr)
        & np.isfinite(disp_arr)
        & (means_arr > 0)  # Only exclude truly zero mean (avoids inf in 1/mean)
        & (disp_arr > 0)   # Exclude zero/negative dispersion
    )
    n_valid = valid_for_fit.sum()
    
    if n_valid < 3:
        baseline = np.nanmedian(disp_arr[valid_for_fit]) if np.any(valid_for_fit) else 0.1
        return np.full_like(means_arr, baseline, dtype=np.float64)
    
    x_valid = means_arr[valid_for_fit]
    y_valid = disp_arr[valid_for_fit]
    
    if fit_type == "mean":
        baseline = np.nanmedian(y_valid)
        return np.full_like(means_arr, baseline, dtype=np.float64)
    
    if fit_type == "parametric":
        # PyDESeq2-style Gamma GLM fit: disp = a0 + a1 / mean
        # Uses L-BFGS-B optimization with Gamma deviance loss, matching PyDESeq2
        # The model is: E[disp] = a0 + a1 / mean (identity link)
        # Loss = mean(target / mu + log(mu)) where mu = a0 + a1 * covariate
        from scipy.optimize import minimize
        
        try:
            # Build design matrix: [1, 1/mean] for covariates (PyDESeq2 style)
            covariates = np.column_stack([np.ones_like(x_valid), 1.0 / x_valid])
            targets = y_valid.copy()
            
            # PyDESeq2-style iterative fitting with outlier removal
            # Key difference: remove genes from arrays after each iteration
            old_params = np.array([0.1, 0.1])
            params = np.array([1.0, 1.0])
            X_current = covariates.copy()
            y_current = targets.copy()
            
            # Convergence criterion from PyDESeq2:
            # while (coeffs > 1e-10).all() and (log(|coeffs/old_coeffs|)^2).sum() >= 1e-6
            max_outer_iter = 20
            for outer_iter in range(max_outer_iter):
                # Check convergence
                if not (params > 1e-10).all():
                    break
                log_change = np.log(np.abs(params / old_params)) ** 2
                if log_change.sum() < 1e-6 and outer_iter > 0:
                    break
                
                old_params = params.copy()
                
                # Gamma GLM loss and gradient
                def loss(coeffs):
                    mu = X_current @ coeffs
                    mu = np.maximum(mu, 1e-10)
                    return np.nanmean(y_current / mu + np.log(mu))
                
                def grad(coeffs):
                    mu = X_current @ coeffs
                    mu = np.maximum(mu, 1e-10)
                    return -np.nanmean(
                        ((y_current / mu - 1)[:, None] * X_current) / mu[:, None],
                        axis=0
                    )
                
                try:
                    res = minimize(
                        loss,
                        x0=params,
                        jac=grad,
                        method="L-BFGS-B",
                        bounds=[(1e-12, np.inf), (1e-12, np.inf)],
                    )
                    if not res.success:
                        break
                    params = res.x
                    predictions = X_current @ params
                except Exception:
                    break
                
                # Outlier removal (PyDESeq2 style): keep genes with 1e-4 <= ratio < 15
                pred_ratios = y_current / np.maximum(predictions, 1e-10)
                keep_mask = (pred_ratios >= 1e-4) & (pred_ratios < 15.0)
                
                if keep_mask.sum() < 3:
                    break  # Not enough genes
                
                # Remove outliers from arrays (critical: this is how PyDESeq2 does it)
                X_current = X_current[keep_mask]
                y_current = y_current[keep_mask]
            
            # Compute trend for all genes: a0 + a1 / mean
            # PyDESeq2 doesn't clamp mean here - use small epsilon to avoid division by zero
            trend = params[0] + params[1] / np.maximum(means_arr, 1e-8)
            trend = np.maximum(trend, 1e-8)
            return trend
            
        except Exception:
            # Fall back to polynomial fit
            pass
    
    # Fallback: log-quadratic polynomial fit (original method)
    x = np.log(np.clip(means_arr[valid_for_fit], min_mean, None))
    y = np.log(np.clip(disp_arr[valid_for_fit], 1e-10, None))
    
    # Use robust weights to reduce influence of outliers
    median_y = np.median(y)
    mad_y = np.median(np.abs(y - median_y))
    weights = 1.0 / (1.0 + ((y - median_y) / (1.4826 * mad_y + 1e-8)) ** 2)
    
    try:
        coeffs = np.polyfit(x, y, deg=2, w=weights)
    except np.linalg.LinAlgError:
        coeffs = np.polyfit(x, y, deg=2)
    
    log_means_all = np.log(np.clip(means_arr, min_mean, None))
    trend = np.exp(np.polyval(coeffs, log_means_all))
    return np.maximum(trend, 1e-8)


def shrink_dispersions(
    raw: ArrayLike,
    trend: ArrayLike,
    *,
    prior_df: float | None = None,
    min_prior_df: float = 1.0,
    max_prior_df: float = 100.0,
    outlier_sigma: float = 2.0,
    n_iter: int = 5,
) -> np.ndarray:
    """Shrink dispersions toward fitted trend using empirical Bayes.
    
    This implements a DESeq2/PyDESeq2-style empirical Bayes shrinkage where the
    prior variance is estimated from the distribution of log-dispersion residuals
    around the trend using an iterative trimmed variance estimator.
    
    Genes with dispersions more than `outlier_sigma` standard deviations above 
    the trend keep their MLE (not shrunken), matching PyDESeq2's outlier handling.
    
    The shrinkage formula is:
        log(shrunk) = (prior_df * log(trend) + log(raw)) / (prior_df + 1)
    
    This is equivalent to a posterior mean estimate under a log-normal prior.
    
    Parameters
    ----------
    raw
        Raw MLE dispersion estimates.
    trend
        Fitted mean-dispersion trend values.
    prior_df
        Prior degrees of freedom controlling shrinkage strength.
        If None, estimated empirically from the data using iterative trimming.
    min_prior_df
        Minimum allowed prior degrees of freedom.
    max_prior_df
        Maximum allowed prior degrees of freedom.
    outlier_sigma
        Number of standard deviations above trend beyond which genes
        keep their MLE dispersion (not shrunken). Set to inf to disable.
    
    Returns
    -------
    np.ndarray
        Shrunken dispersion estimates.
    """
    raw_arr = np.asarray(raw, dtype=np.float64)
    trend_arr = np.asarray(trend, dtype=np.float64)
    shrunk = np.array(raw_arr, copy=True)
    mask = (
        np.isfinite(raw_arr)
        & np.isfinite(trend_arr)
        & (raw_arr > 0)
        & (trend_arr > 0)
    )
    if not np.any(mask):
        return shrunk
    
    log_raw = np.log(raw_arr[mask])
    log_trend = np.log(trend_arr[mask])
    residuals = log_raw - log_trend
    
    # PyDESeq2-style iterative trimmed variance estimator
    # Iteratively exclude residuals outside 2.5 MAD of median to get robust prior_var
    trim_threshold = 2.5
    use_for_var = np.ones(len(residuals), dtype=bool)
    
    for _ in range(n_iter):
        resid_subset = residuals[use_for_var]
        if len(resid_subset) < 3:
            break
            
        median_resid = np.median(resid_subset)
        mad = np.median(np.abs(resid_subset - median_resid))
        sigma_resid = 1.4826 * mad  # Scale MAD to approximate std dev
        
        if sigma_resid < 1e-8:
            # Fallback to sample std
            sigma_resid = max(np.std(resid_subset, ddof=1), 1e-4)
        
        # Update mask: include only residuals within trim_threshold * sigma
        abs_dev = np.abs(residuals - median_resid)
        use_for_var = abs_dev < trim_threshold * sigma_resid
        
        if use_for_var.sum() < 3:
            use_for_var = np.ones(len(residuals), dtype=bool)
            break
    
    # Final variance estimate from trimmed residuals
    resid_subset = residuals[use_for_var]
    median_resid = np.median(resid_subset)
    mad = np.median(np.abs(resid_subset - median_resid))
    sigma_resid = 1.4826 * mad
    
    # If variance is very small, use sample variance from trimmed set
    if sigma_resid < 1e-4:
        sigma_resid = max(np.std(resid_subset, ddof=1), 1e-4)
    
    prior_var = sigma_resid ** 2
    
    if prior_df is None:
        # PyDESeq2 estimates prior_df from the variance of log-dispersion residuals
        # Using the relationship: Var(log_disp) = trigamma(prior_df)
        # Approximate: prior_df ≈ 1 / prior_var for log-normal approximation
        if prior_var > 1e-8:
            # Use trigamma inverse approximation for better accuracy
            # For small prior_var, prior_df is large
            prior_df = 1.0 / prior_var
        else:
            prior_df = max_prior_df
        
        prior_df = float(np.clip(prior_df, min_prior_df, max_prior_df))
    
    # Identify outliers: genes with dispersion > outlier_sigma * sigma above trend
    # These genes keep their MLE (not shrunken) - PyDESeq2 behavior
    is_outlier = residuals > outlier_sigma * sigma_resid
    
    # Apply shrinkage to non-outliers
    log_post = (prior_df * log_trend + log_raw) / (prior_df + 1.0)
    
    # Non-outliers get shrunken values, outliers keep MLE
    shrunk_values = np.where(is_outlier, np.exp(log_raw), np.exp(log_post))
    shrunk[mask] = shrunk_values
    
    return shrunk


def _refine_dispersion_brent(
    j: int,
    Y_col: np.ndarray,
    mu_col: np.ndarray,
    log_trend_j: float,
    prior_var: float,
    log_min: float,
    log_max: float,
) -> float:
    """Refine dispersion for a single gene using Brent's method.
    
    This is used after grid search to get the exact optimum.
    Memory optimization: computes gammaln(Y + 1) on-the-fly.
    """
    from scipy.optimize import brentq, minimize_scalar
    
    def neg_posterior(log_alpha: float) -> float:
        alpha = np.exp(log_alpha)
        r = 1.0 / alpha
        
        # NB log-likelihood - using numba-accelerated gammaln
        ll = np.sum(
            gammaln_nb(Y_col + r)
            - gammaln_nb(r)
            - gammaln_nb(Y_col + 1.0)
            + r * np.log(r / (r + mu_col + 1e-12))
            + Y_col * np.log(mu_col / (r + mu_col + 1e-12) + 1e-12)
        )
        
        # Log-prior
        log_prior = -0.5 * (log_alpha - log_trend_j) ** 2 / prior_var
        
        return -(ll + log_prior)
    
    # Use bounded scalar optimization with Brent's method
    result = minimize_scalar(
        neg_posterior,
        bounds=(log_min, log_max),
        method='bounded',
        options={'xatol': 1e-4}
    )
    
    if result.success:
        return np.exp(result.x)
    else:
        return np.exp(log_trend_j)  # Fallback to trend


def estimate_dispersion_map(
    Y: np.ndarray,
    mu: np.ndarray,
    trend: np.ndarray,
    *,
    prior_var: float | None = None,
    min_disp: float = 1e-8,
    max_disp: float = 10.0,
    n_grid: int = 30,
    refine: bool = False,
    n_jobs: int = -1,
) -> np.ndarray:
    """Estimate MAP dispersion using vectorized grid search + optional refinement.
    
    This implements PyDESeq2-style MAP estimation where the dispersion
    is estimated by maximizing:
        log L(Y | mu, alpha) + log prior(alpha | trend, prior_var)
    
    The prior is log-normal: log(alpha) ~ N(log(trend), prior_var)
    
    **Optimization (v2)**: Uses Numba-accelerated vectorized grid search over
    all genes in parallel, followed by optional per-gene refinement using
    Brent's method. This is ~10-50× faster than the original per-gene 
    sequential optimization.
    
    **Memory optimization (v3)**: Default changed to refine=False with n_grid=30
    for ~2× speedup with <1% accuracy loss. Use refine=True for highest precision.
    
    Parameters
    ----------
    Y
        Count matrix of shape (n_cells, n_genes).
    mu
        Fitted mean matrix of shape (n_cells, n_genes).
    trend
        Dispersion trend values of shape (n_genes,).
    prior_var
        Variance of the log-normal prior. If None, estimated from data
        using the variance of log-dispersion residuals around trend.
    min_disp
        Minimum allowed dispersion value.
    max_disp
        Maximum allowed dispersion value.
    n_grid
        Number of grid points for initial search. More points = better
        initial estimate but slower grid search. Default increased to 30
        for better accuracy when refine=False.
    refine
        If True, refine the grid search result using Brent's method.
        Default is False for ~2× speedup with <1% accuracy loss.
    n_jobs
        Number of parallel jobs for refinement. -1 uses all cores.
    
    Returns
    -------
    np.ndarray
        MAP dispersion estimates of shape (n_genes,).
    """
    from scipy.special import polygamma
    
    n_cells, n_genes = Y.shape
    # Avoid copies if already float64
    if Y.dtype != np.float64:
        Y = np.asarray(Y, dtype=np.float64)
    if mu.dtype != np.float64:
        mu = np.asarray(mu, dtype=np.float64)
    trend = np.asarray(trend, dtype=np.float64)
    
    # Clip mu in-place to avoid creating a copy
    np.maximum(mu, 1e-10, out=mu)
    
    # Memory-optimized MLE estimation using Numba for speed
    # Computes per-gene MLE dispersion without large intermediate arrays
    dof = max(n_cells - 2, 1)
    alpha_mle = _compute_mle_dispersion_numba(Y, mu, dof)
    alpha_mle = np.clip(alpha_mle, min_disp, max_disp)
    
    # Estimate prior variance if not provided (PyDESeq2 style)
    if prior_var is None:
        log_alpha = np.log(np.maximum(alpha_mle, min_disp))
        log_trend_arr = np.log(np.maximum(trend, min_disp))
        valid = np.isfinite(log_alpha) & np.isfinite(log_trend_arr)
        if np.sum(valid) > 10:
            residuals = log_alpha[valid] - log_trend_arr[valid]
            # Robust estimate using MAD (mean absolute deviation)
            mad = np.median(np.abs(residuals - np.median(residuals)))
            squared_logres = (1.4826 * mad) ** 2
            # PyDESeq2 formula: max(squared_logres - polygamma_correction, 0.25)
            num_vars = 2  # intercept + perturbation
            polygamma_corr = polygamma(1, (n_cells - num_vars) / 2)
            prior_var = max(squared_logres - polygamma_corr, 0.25)
        else:
            prior_var = 0.25
    
    log_trend = np.log(np.maximum(trend, min_disp))
    log_min = np.log(min_disp)
    log_max = np.log(max_disp)
    
    # Create grid of log-alpha values
    log_alpha_grid = np.linspace(log_min, log_max, n_grid)
    
    # =========================================================================
    # Stage 1: Vectorized grid search using Numba
    # Memory optimization: gammaln(Y+1) is computed on-the-fly in the kernel
    # instead of precomputing a full (n_cells, n_genes) array (~144 MB savings)
    # =========================================================================
    best_log_alpha, best_idx = _nb_map_grid_search_numba(
        Y, mu, log_trend, log_alpha_grid, prior_var
    )
    
    # If not refining, return grid search results directly
    if not refine:
        return np.exp(np.clip(best_log_alpha, log_min, log_max))
    
    # =========================================================================
    # Stage 2: Refine using Brent's method (parallel)
    # =========================================================================
    # For each gene, determine refinement bounds based on grid search result
    # Use adjacent grid points as bounds, or the grid boundaries
    lower_bounds = np.where(
        best_idx > 0,
        log_alpha_grid[np.maximum(best_idx - 1, 0).astype(int)],
        log_min
    )
    upper_bounds = np.where(
        best_idx < n_grid - 1,
        log_alpha_grid[np.minimum(best_idx + 1, n_grid - 1).astype(int)],
        log_max
    )
    
    # Parallel refinement - gammaln computed on-the-fly in _refine_dispersion_brent
    def _refine_gene(j):
        return _refine_dispersion_brent(
            j,
            Y[:, j],
            mu[:, j],
            float(log_trend[j]),
            prior_var,
            float(lower_bounds[j]),
            float(upper_bounds[j]),
        )
    
    # Use joblib for parallel refinement
    alpha_map = np.array(
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_refine_gene)(j) for j in range(n_genes)
        ),
        dtype=np.float64
    )
    
    # Clip to bounds
    alpha_map = np.clip(alpha_map, min_disp, max_disp)
    
    return alpha_map


def _estimate_dispersion_map_sequential(
    Y: np.ndarray,
    mu: np.ndarray,
    trend: np.ndarray,
    *,
    prior_var: float | None = None,
    min_disp: float = 1e-8,
    max_disp: float = 10.0,
) -> np.ndarray:
    """Original sequential MAP dispersion estimation (kept for reference/testing).
    
    This is the original per-gene sequential implementation. Use estimate_dispersion_map
    for the optimized vectorized version.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import polygamma
    
    n_cells, n_genes = Y.shape
    Y = np.asarray(Y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    trend = np.asarray(trend, dtype=np.float64)
    
    # Clip mu to avoid numerical issues
    mu = np.maximum(mu, 1e-10)
    
    # Initial estimate: use method of moments for MLE
    resid = Y - mu
    variance = resid ** 2 - Y
    denom = np.maximum(mu ** 2, 1e-10)
    dof = max(n_cells - 2, 1)
    alpha_mle = np.sum(variance / denom, axis=0) / dof
    alpha_mle = np.clip(alpha_mle, min_disp, max_disp)
    
    # Estimate prior variance if not provided (PyDESeq2 style)
    if prior_var is None:
        log_alpha = np.log(np.maximum(alpha_mle, min_disp))
        log_trend_arr = np.log(np.maximum(trend, min_disp))
        valid = np.isfinite(log_alpha) & np.isfinite(log_trend_arr)
        if np.sum(valid) > 10:
            residuals = log_alpha[valid] - log_trend_arr[valid]
            # Robust estimate using MAD (mean absolute deviation)
            mad = np.median(np.abs(residuals - np.median(residuals)))
            squared_logres = (1.4826 * mad) ** 2
            # PyDESeq2 formula: max(squared_logres - polygamma_correction, 0.25)
            num_vars = 2  # intercept + perturbation
            polygamma_corr = polygamma(1, (n_cells - num_vars) / 2)
            prior_var = max(squared_logres - polygamma_corr, 0.25)
        else:
            prior_var = 0.25
    
    log_trend = np.log(np.maximum(trend, min_disp))
    log_min = np.log(min_disp)
    log_max = np.log(max_disp)
    
    # Result array
    alpha_map = np.zeros(n_genes, dtype=np.float64)
    
    def nb_loglik_plus_prior(log_alpha, y_col, mu_col, log_trend_j, prior_var):
        """Negative of (log-likelihood + log-prior) for minimization."""
        alpha = np.exp(log_alpha)
        inv_alpha = 1.0 / alpha
        
        # NB log-likelihood (vectorized over cells, using numba-accelerated gammaln)
        ll = np.sum(
            gammaln_nb(y_col + inv_alpha)
            - gammaln_nb(inv_alpha)
            - gammaln_nb(y_col + 1)
            + inv_alpha * np.log(inv_alpha / (inv_alpha + mu_col))
            + y_col * np.log(mu_col / (inv_alpha + mu_col))
        )
        
        # Log-normal prior on log(alpha)
        log_prior = -0.5 * (log_alpha - log_trend_j) ** 2 / prior_var
        
        # Return negative for minimization
        return -(ll + log_prior)
    
    # Optimize each gene
    for j in range(n_genes):
        y_col = Y[:, j]
        mu_col = mu[:, j]
        log_trend_j = log_trend[j]
        
        # Use bounded scalar optimization
        result = minimize_scalar(
            lambda x: nb_loglik_plus_prior(x, y_col, mu_col, log_trend_j, prior_var),
            bounds=(log_min, log_max),
            method='bounded',
            options={'xatol': 1e-4}
        )
        
        if result.success:
            alpha_map[j] = np.exp(result.x)
        else:
            # Fall back to trend if optimization fails
            alpha_map[j] = trend[j]
    
    # Clip to bounds
    alpha_map = np.clip(alpha_map, min_disp, max_disp)
    
    return alpha_map


def shrink_log_foldchange(
    coef: ArrayLike,
    se: ArrayLike,
    *,
    prior_var: float | None = None,
    shrinkage_type: Literal["normal", "apeglm", "none"] = "normal",
) -> np.ndarray:
    """Apply empirical Bayes shrinkage to log-fold changes.
    
    This implements shrinkage methods similar to DESeq2/PyDESeq2:
    
    - "normal": Normal-normal conjugate shrinkage (DESeq2 default prior to v1.16)
    - "apeglm": Approximate posterior estimation using adaptive shrinkage that
      preserves genes with strong evidence of differential expression
    - "none": No shrinkage (PyDESeq2 default behavior)
    
    For "normal" shrinkage:
        coef ~ N(true_effect, se^2)  (likelihood)
        true_effect ~ N(0, prior_var)  (prior)
        shrunk = coef * prior_var / (prior_var + se^2)  (posterior mean)
    
    Parameters
    ----------
    coef
        Raw log-fold change estimates (natural log scale).
    se
        Standard errors of the coefficients.
    prior_var
        Prior variance for the shrinkage. If None, estimated empirically
        using a robust variance estimator.
    shrinkage_type
        Type of shrinkage to apply:
        - "normal": Standard normal-normal shrinkage
        - "apeglm": Adaptive shrinkage preserving strong signals
        - "none": No shrinkage (return raw coefficients)
    
    Returns
    -------
    np.ndarray
        Shrunken log-fold change estimates (or raw if shrinkage_type="none").
    """
    coef_arr = np.asarray(coef, dtype=np.float64)
    se_arr = np.asarray(se, dtype=np.float64)
    shrunk = np.array(coef_arr, copy=True)
    
    if shrinkage_type == "none":
        return shrunk
    
    mask = np.isfinite(coef_arr) & np.isfinite(se_arr) & (se_arr > 0)
    if not np.any(mask):
        return shrunk
    
    if prior_var is None:
        # Estimate prior variance empirically
        # Use the median of squared coefficients minus median squared SE
        # This is a method-of-moments estimator for prior_var
        coef_sq = coef_arr[mask] ** 2
        se_sq = se_arr[mask] ** 2
        
        # Estimate total variance and subtract observation variance
        total_var = np.median(coef_sq)
        obs_var = np.median(se_sq)
        prior_var = max(total_var - obs_var, 0.01)  # Ensure positive
        
        # Alternative: use robust scale estimate
        if prior_var < 0.01:
            mad = np.median(np.abs(coef_arr[mask]))
            prior_var = (1.4826 * mad) ** 2
    
    prior_var = float(np.maximum(prior_var, 1e-8))
    
    if shrinkage_type == "apeglm":
        # Adaptive shrinkage: reduce shrinkage for genes with high z-scores
        # This preserves strong signals while shrinking weak ones
        z_scores = np.abs(coef_arr[mask] / se_arr[mask])
        # Probability of being non-null based on z-score
        # Use a sigmoid-like weighting (approximates apeglm behavior)
        lfdr = 1.0 / (1.0 + np.exp(2.0 * (z_scores - 2.0)))  # local FDR approximation
        
        # Shrinkage factor interpolates between full shrinkage and no shrinkage
        base_shrink = prior_var / (prior_var + se_arr[mask] ** 2)
        shrink_factor = lfdr * base_shrink + (1.0 - lfdr) * 1.0
        shrunk[mask] = coef_arr[mask] * shrink_factor
    else:
        # Standard normal-normal shrinkage
        shrink_factor = prior_var / (prior_var + se_arr[mask] ** 2)
        shrunk[mask] = coef_arr[mask] * shrink_factor
    
    return shrunk


def _estimate_apeglm_prior_scale(
    mle_lfc: np.ndarray,
    se: np.ndarray,
    init_scale: float = 1.0,
) -> float:
    """Estimate apeGLM prior scale parameter using PyDESeq2's adaptive method.
    
    This implements the prior variance estimation from PyDESeq2's `fit_prior_var`
    function, which finds the scale parameter that balances the data likelihood
    with the Cauchy prior.
    
    Parameters
    ----------
    mle_lfc
        MLE log-fold change estimates.
    se
        Standard errors of the MLE estimates.
    init_scale
        Initial guess for the scale parameter.
    
    Returns
    -------
    float
        Estimated prior scale parameter for the Cauchy prior.
    """
    mask = np.isfinite(mle_lfc) & np.isfinite(se) & (se > 0)
    if not np.any(mask):
        return init_scale
    
    S = mle_lfc[mask] ** 2
    D = se[mask] ** 2
    
    def objective(a: float) -> float:
        """Objective function: find a such that weighted mean of (S-D) equals a."""
        coeff = 1.0 / (2.0 * (a + D) ** 2)
        return float(((S - D) * coeff).sum() / coeff.sum() - a)
    
    # Try to find root, fall back to reasonable default if it fails
    try:
        # Search for root in reasonable range
        low, high = 1e-6, 400.0
        f_low, f_high = objective(low), objective(high)
        
        if f_low * f_high < 0:
            # Root exists in bracket
            scale_sq = brentq(objective, low, high, xtol=1e-6)
        else:
            # No root in bracket, use moment estimate
            scale_sq = max(float(np.median(S) - np.median(D)), 0.01)
    except Exception:
        # Fallback: use robust moment estimate
        scale_sq = max(float(np.median(S) - np.median(D)), 0.01)
    
    return max(np.sqrt(scale_sq), 0.1)


def shrink_lfc_apeglm(
    counts: np.ndarray,
    design_matrix: np.ndarray,
    size_factors: np.ndarray,
    dispersion: np.ndarray,
    mle_coef: np.ndarray,
    mle_se: np.ndarray,
    *,
    shrink_index: int = 1,
    prior_scale: float | None = None,
    prior_no_shrink_scale: float = 15.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply apeGLM LFC shrinkage using Cauchy prior (PyDESeq2-compatible).
    
    This implements the apeGLM (approximate posterior estimation for GLM)
    shrinkage method used by DESeq2/PyDESeq2. Unlike normal shrinkage,
    apeGLM uses a heavy-tailed Cauchy prior that preserves large effects
    while shrinking small/uncertain effects toward zero.
    
    The method re-fits the NB-GLM model with a Cauchy prior penalty on the
    LFC coefficient, finding the MAP (maximum a posteriori) estimate.
    
    Parameters
    ----------
    counts
        Raw count matrix (n_cells, n_genes).
    design_matrix
        Design matrix (n_cells, n_params).
    size_factors
        Size factors for each cell (n_cells,).
    dispersion
        Gene-wise dispersion estimates (n_genes,).
    mle_coef
        MLE coefficient matrix (n_params, n_genes) for warm-starting.
    mle_se
        Standard errors of MLE LFC estimates (n_genes,).
    shrink_index
        Index of the coefficient to shrink (default: 1, the LFC coefficient).
    prior_scale
        Scale parameter for Cauchy prior. If None, estimated adaptively.
    prior_no_shrink_scale
        Scale for normal prior on non-shrunk coefficients (default: 15.0).
    max_iter
        Maximum iterations for L-BFGS-B optimization.
    tol
        Convergence tolerance.
    n_jobs
        Number of parallel jobs for per-gene optimization.
    
    Returns
    -------
    shrunk_coef
        Shrunken coefficient matrix (n_params, n_genes).
    shrunk_se
        Approximate standard errors from inverse Hessian (n_genes,).
    converged
        Boolean array indicating convergence for each gene (n_genes,).
    """
    n_cells, n_genes = counts.shape
    n_params = design_matrix.shape[1]
    
    # Ensure arrays are float64
    counts = np.asarray(counts, dtype=np.float64)
    design_matrix = np.asarray(design_matrix, dtype=np.float64)
    size_factors = np.asarray(size_factors, dtype=np.float64).ravel()
    dispersion = np.asarray(dispersion, dtype=np.float64).ravel()
    mle_coef = np.asarray(mle_coef, dtype=np.float64)
    mle_se = np.asarray(mle_se, dtype=np.float64).ravel()
    
    # Pre-compute log size factors (offset)
    log_size_factors = np.log(size_factors + 1e-10)
    
    # Estimate prior scale if not provided
    if prior_scale is None:
        mle_lfc = mle_coef[shrink_index, :]
        prior_scale = _estimate_apeglm_prior_scale(mle_lfc, mle_se)
    
    prior_scale_sq = prior_scale ** 2
    prior_no_shrink_var = prior_no_shrink_scale ** 2
    
    def _fit_gene_apeglm(j: int) -> tuple[np.ndarray, float, bool]:
        """Fit apeGLM for a single gene."""
        y = counts[:, j]
        disp = dispersion[j]
        beta_init = mle_coef[:, j].copy()
        
        # Skip genes with invalid data
        if not np.isfinite(disp) or disp <= 0 or not np.all(np.isfinite(beta_init)):
            return beta_init, mle_se[j] if np.isfinite(mle_se[j]) else 1.0, False
        
        size = 1.0 / disp  # NB size parameter
        
        def neg_log_posterior(beta: np.ndarray) -> float:
            """Negative log posterior = NB NLL + prior penalties."""
            # Linear predictor
            eta = design_matrix @ beta + log_size_factors
            mu = np.exp(np.clip(eta, -30, 30))
            
            # NB negative log-likelihood (up to constants)
            # -log L = sum(-y*log(mu) + (y + size)*log(mu + size) + const)
            nll = np.sum(-y * np.log(mu + 1e-10) + (y + size) * np.log(mu + size))
            
            # Prior penalties
            # Normal prior on intercept (index 0)
            prior_intercept = beta[0] ** 2 / (2 * prior_no_shrink_var)
            # Cauchy prior on LFC coefficient (shrink_index)
            prior_lfc = np.log1p((beta[shrink_index] / prior_scale) ** 2)
            # Normal priors on other covariates
            prior_other = 0.0
            for k in range(n_params):
                if k != 0 and k != shrink_index:
                    prior_other += beta[k] ** 2 / (2 * prior_no_shrink_var)
            
            return nll + prior_intercept + prior_lfc + prior_other
        
        def gradient(beta: np.ndarray) -> np.ndarray:
            """Gradient of negative log posterior."""
            eta = design_matrix @ beta + log_size_factors
            mu = np.exp(np.clip(eta, -30, 30))
            
            # NB gradient: d(NLL)/d(beta) = X^T @ (mu - y * mu / (mu + eps))
            # Simplified: X^T @ (mu * (1 - y/(mu + size)) * (1 + size/mu)^(-1) * something)
            # Actually: d(NLL)/d(eta) = (mu - y) * mu / (mu + size)
            # But using the form: d(NLL)/d(eta) = (1 - (y + size) / (mu + size)) * mu
            w = mu - y * mu / (mu + size)
            grad_nll = design_matrix.T @ w
            
            # Prior gradients
            grad_prior = np.zeros(n_params)
            grad_prior[0] = beta[0] / prior_no_shrink_var
            # Cauchy prior gradient: 2 * beta / (scale^2 + beta^2)
            grad_prior[shrink_index] = 2 * beta[shrink_index] / (prior_scale_sq + beta[shrink_index] ** 2)
            for k in range(n_params):
                if k != 0 and k != shrink_index:
                    grad_prior[k] = beta[k] / prior_no_shrink_var
            
            return grad_nll + grad_prior
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                neg_log_posterior,
                beta_init,
                method="L-BFGS-B",
                jac=gradient,
                options={"maxiter": max_iter, "gtol": tol},
            )
            beta_map = result.x
            converged = result.success
            
            # Estimate SE from inverse Hessian (approximate)
            # Use finite difference Hessian at MAP
            eta = design_matrix @ beta_map + log_size_factors
            mu = np.exp(np.clip(eta, -30, 30))
            disp_inv = size
            W = mu * (1 + mu / disp_inv) ** (-1)  # NB weights
            XtWX = design_matrix.T @ (design_matrix * W[:, None])
            
            # Add prior curvature
            XtWX[0, 0] += 1.0 / prior_no_shrink_var
            # Cauchy Hessian: 2*(s^2 - beta^2) / (s^2 + beta^2)^2
            cauchy_hess = 2 * (prior_scale_sq - beta_map[shrink_index]**2) / (prior_scale_sq + beta_map[shrink_index]**2)**2
            XtWX[shrink_index, shrink_index] += cauchy_hess
            
            try:
                inv_hess = np.linalg.inv(XtWX)
                se_map = np.sqrt(max(inv_hess[shrink_index, shrink_index], 1e-10))
            except np.linalg.LinAlgError:
                se_map = mle_se[j] if np.isfinite(mle_se[j]) else 1.0
                
        except Exception:
            # Fall back to MLE if optimization fails
            beta_map = beta_init
            se_map = mle_se[j] if np.isfinite(mle_se[j]) else 1.0
            converged = False
        
        return beta_map, se_map, converged
    
    # Parallel optimization over genes
    if n_jobs == 1:
        results = [_fit_gene_apeglm(j) for j in range(n_genes)]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_gene_apeglm)(j) for j in range(n_genes)
        )
    
    # Collect results
    shrunk_coef = np.zeros((n_params, n_genes), dtype=np.float64)
    shrunk_se = np.zeros(n_genes, dtype=np.float64)
    converged = np.zeros(n_genes, dtype=bool)
    
    for j, (beta, se, conv) in enumerate(results):
        shrunk_coef[:, j] = beta
        shrunk_se[j] = se
        converged[j] = conv
    
    return shrunk_coef, shrunk_se, converged


def compute_cooks_distance_batch(
    Y: np.ndarray,
    mu: np.ndarray,
    dispersion: np.ndarray,
    n_params: int = 2,
) -> np.ndarray:
    """Compute Cook's distance for each observation in a batch of genes.
    
    Cook's distance measures the influence of each observation on the fitted
    model. Large values indicate potential outliers that disproportionately
    affect the estimates.
    
    For GLMs, Cook's distance is computed as:
        D_i = (r_i^2 / p) * (h_ii / (1 - h_ii)^2)
    
    where r_i is the Pearson residual, p is the number of parameters,
    and h_ii is the leverage (diagonal of the hat matrix).
    
    Parameters
    ----------
    Y
        Count matrix of shape (n_cells, n_genes).
    mu
        Fitted mean matrix of shape (n_cells, n_genes).
    dispersion
        Dispersion estimates of shape (n_genes,).
    n_params
        Number of model parameters (default 2 for intercept + treatment).
    
    Returns
    -------
    np.ndarray
        Cook's distance matrix of shape (n_cells, n_genes).
    """
    n_cells, n_genes = Y.shape
    
    # Compute variance: V = mu + dispersion * mu^2
    variance = mu + dispersion[None, :] * mu ** 2
    variance = np.maximum(variance, 1e-10)
    
    # Pearson residuals: r = (Y - mu) / sqrt(V)
    resid = (Y - mu) / np.sqrt(variance)
    
    # Weights for NB GLM: W = mu^2 / V
    weights = mu ** 2 / variance
    
    # Approximate leverage using average weight contribution
    # For balanced designs: h_ii ≈ W_i / sum(W)
    sum_weights = np.sum(weights, axis=0, keepdims=True)
    leverage = weights / np.maximum(sum_weights, 1e-10)
    
    # Clip leverage to avoid division by zero
    leverage = np.clip(leverage, 1e-10, 1 - 1e-10)
    
    # Cook's distance: D = (r^2 / p) * (h / (1-h)^2)
    cooks = (resid ** 2 / n_params) * (leverage / (1 - leverage) ** 2)
    
    return cooks


def filter_outliers_cooks(
    Y: np.ndarray,
    mu: np.ndarray,
    dispersion: np.ndarray,
    *,
    n_params: int = 2,
    threshold_quantile: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identify and replace outlier counts based on Cook's distance.
    
    Following DESeq2's approach:
    1. Compute Cook's distance for each observation
    2. Identify outliers where Cook's D > F(threshold_quantile, p, n-p)
    3. Replace outlier counts with trimmed mean from non-outlier samples
    
    Parameters
    ----------
    Y
        Count matrix of shape (n_cells, n_genes).
    mu
        Fitted mean matrix of shape (n_cells, n_genes).
    dispersion
        Dispersion estimates of shape (n_genes,).
    n_params
        Number of model parameters.
    threshold_quantile
        Quantile of F distribution for outlier threshold.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Y_filtered: Count matrix with outliers replaced
        - outlier_mask: Boolean matrix indicating outliers (n_cells, n_genes)
    """
    from scipy import stats
    
    n_cells, n_genes = Y.shape
    
    # Compute Cook's distance
    cooks = compute_cooks_distance_batch(Y, mu, dispersion, n_params)
    
    # F distribution threshold
    dfn = n_params
    dfd = max(n_cells - n_params, 1)
    threshold = stats.f.ppf(threshold_quantile, dfn, dfd)
    
    # Identify outliers
    outlier_mask = cooks > threshold
    
    # Replace outliers with trimmed mean
    Y_filtered = Y.copy()
    
    for g in range(n_genes):
        outliers_g = outlier_mask[:, g]
        if np.any(outliers_g):
            non_outlier_counts = Y[~outliers_g, g]
            if len(non_outlier_counts) > 0:
                # Use trimmed mean (exclude top/bottom 10%)
                trimmed_mean = stats.trim_mean(non_outlier_counts, 0.1)
                Y_filtered[outliers_g, g] = trimmed_mean
    
    return Y_filtered, outlier_mask


def estimate_covariate_effects_streaming(
    backed_adata,
    *,
    obs_df: "pd.DataFrame",
    perturbation_labels: np.ndarray,
    control_label: str,
    covariate_columns: Sequence[str],
    size_factors: np.ndarray,
    chunk_size: int = 2048,
    poisson_iter: int = 10,
    tol: float = 1e-6,
    return_intercept: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Estimate global intercept and covariate effects using control cells only.
    
    This function fits a Poisson regression using only control cells to estimate:
    - An intercept (baseline expression for control cells)
    - Covariate effects (if any covariates are specified)
    
    By using only control cells, the intercept represents the true control baseline
    that can then be used as an offset in per-perturbation fitting. This ensures
    that perturbation effects are properly estimated as deviations from control.
    
    For the intercept-only case (no covariates), the closed-form MLE is used:
        intercept = log(sum(Y) / sum(size_factors))
    
    When covariates are present, IRLS is used with proper per-gene weighting.
    
    Parameters
    ----------
    backed_adata
        Backed AnnData object opened in read mode.
    obs_df
        Full obs DataFrame with all cells (already loaded).
    perturbation_labels
        Array of perturbation labels for all cells.
    control_label
        The label identifying control cells.
    covariate_columns
        List of covariate column names to include.
    size_factors
        Per-cell size factors (length n_cells).
    chunk_size
        Number of cells to process per chunk.
    poisson_iter
        Number of Poisson IRLS iterations.
    tol
        Convergence tolerance.
    return_intercept
        If True, also return the global intercept coefficients. This is useful
        for joint fitting where the intercept should be shared across all
        perturbation comparisons.
        
    Returns
    -------
    np.ndarray or tuple
        If return_intercept is False:
            Covariate effects of shape (n_covariates, n_genes). These are the
            log-scale regression coefficients for each covariate.
        If return_intercept is True:
            Tuple of (covariate_effects, intercept) where:
            - covariate_effects has shape (n_covariates, n_genes)
            - intercept has shape (n_genes,) representing the control baseline
    """
    import pandas as pd
    from .data import iter_matrix_chunks
    
    n_cells = backed_adata.n_obs
    n_genes = backed_adata.n_vars
    
    # Identify control cells
    control_mask = (perturbation_labels == control_label)
    control_indices = np.where(control_mask)[0]
    n_control = len(control_indices)
    
    if n_control == 0:
        raise ValueError(f"No control cells found with label '{control_label}'")
    
    # Build covariate portion of design matrix (for control cells only)
    cov_matrices = []
    cov_names: list[str] = []
    for column in covariate_columns:
        if column not in obs_df.columns:
            raise KeyError(f"Covariate '{column}' not found in obs_df")
        series = obs_df[column].iloc[control_indices]
        if series.dtype.kind in {"O", "U"} or str(series.dtype).startswith("category"):
            dummies = pd.get_dummies(series, prefix=column, drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                cov_matrices.append(dummies.to_numpy(dtype=np.float64))
                cov_names.extend(dummies.columns.astype(str).tolist())
        else:
            cov_matrices.append(series.to_numpy(dtype=np.float64).reshape(-1, 1))
            cov_names.append(str(column))
    
    n_covariates = sum(m.shape[1] for m in cov_matrices) if cov_matrices else 0
    
    # Size factors for control cells
    size_factors_control = size_factors[control_indices]
    
    if n_covariates == 0:
        # No covariates: use closed-form MLE for intercept
        # intercept = log(sum(Y) / sum(size_factors)) for each gene
        # This is the exact MLE for Poisson with offset
        
        sum_counts = np.zeros(n_genes, dtype=np.float64)
        
        control_idx = 0
        for slc, chunk in iter_matrix_chunks(
            backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
        ):
            chunk_control_mask = control_mask[slc]
            if not np.any(chunk_control_mask):
                continue
            
            Y_chunk_control = np.asarray(chunk[chunk_control_mask], dtype=np.float64)
            sum_counts += Y_chunk_control.sum(axis=0)
        
        sum_size_factors = size_factors_control.sum()
        beta_intercept = np.log(np.maximum(sum_counts / sum_size_factors, 1e-12))
        beta_cov = np.zeros((0, n_genes), dtype=np.float64)
        
        if return_intercept:
            return beta_cov, beta_intercept
        return beta_cov
    
    # With covariates: use IRLS with proper per-gene weighting
    # We need to solve separately for each gene since weights differ per gene
    cov_matrix_control = np.hstack(cov_matrices)  # (n_control, n_covariates)
    
    # Design matrix: intercept + covariates (control cells only)
    # Shape: (n_control, 1 + n_covariates)
    n_features = 1 + n_covariates
    
    # Initialize beta coefficients
    beta = np.zeros((n_features, n_genes), dtype=np.float64)
    
    # Log size factors for control cells
    log_size_factors_control = np.log(np.maximum(size_factors[control_indices], 1e-12))
    
    # Poisson IRLS with streaming - only process control cells
    # We need to map global chunk indices to control cell indices
    for iteration in range(poisson_iter):
        # Accumulate X^T W X and X^T W z across chunks
        xtwx_accum = np.zeros((n_features, n_features), dtype=np.float64)
        xtwz_accum = np.zeros((n_features, n_genes), dtype=np.float64)
        
        control_idx = 0  # Track position within control cells
        for slc, chunk in iter_matrix_chunks(
            backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
        ):
            # Find which cells in this chunk are control cells
            chunk_control_mask = control_mask[slc]
            
            if not np.any(chunk_control_mask):
                continue
            
            # Extract control cells from this chunk
            Y_chunk_control = np.asarray(chunk[chunk_control_mask], dtype=np.float64)
            n_chunk_control = Y_chunk_control.shape[0]
            
            # Get indices of control cells in this chunk (relative to all control cells)
            chunk_control_count = chunk_control_mask.sum()
            control_slice = slice(control_idx, control_idx + chunk_control_count)
            control_idx += chunk_control_count
            
            # Build design for control cells in this chunk
            X_chunk = np.empty((n_chunk_control, n_features), dtype=np.float64)
            X_chunk[:, 0] = 1.0  # Intercept
            if n_covariates > 0:
                X_chunk[:, 1:] = cov_matrix_control[control_slice]
            
            offset_chunk = log_size_factors_control[control_slice]
            
            # Compute eta, mu for this chunk
            eta = X_chunk @ beta + offset_chunk[:, None]
            eta = np.clip(eta, -20.0, 20.0)
            mu = np.exp(eta)
            mu = np.maximum(mu, 1e-6)
            
            # Poisson weights = mu
            W = mu  # (n_chunk_control, n_genes)
            
            # Working response: z = eta - offset + (y - mu) / mu
            z = eta - offset_chunk[:, None] + (Y_chunk_control - mu) / np.maximum(mu, 1e-6)
            
            # Accumulate X^T W X: sum over genes, then over samples
            # X^T W X = sum_g sum_i W[i,g] * X[i,:,None] * X[i,None,:]
            # We use average weights across genes for a shared XtWX
            W_sum = W.sum(axis=1)  # (n_chunk_control,)
            xtwx_accum += X_chunk.T @ (W_sum[:, None] * X_chunk)
            
            # Accumulate X^T W z per gene
            Wz = W * z  # (n_chunk_control, n_genes)
            xtwz_accum += X_chunk.T @ Wz  # (n_features, n_genes)
        
        # Solve for beta: (X^T W X) beta = X^T W z
        # Add ridge penalty for stability
        ridge = 1e-6 * np.eye(n_features)
        try:
            beta_new = np.linalg.solve(xtwx_accum + ridge, xtwz_accum)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(xtwx_accum + ridge, xtwz_accum, rcond=None)[0]
        
        # Check convergence
        max_diff = np.max(np.abs(beta_new - beta))
        beta = beta_new
        
        if max_diff < tol:
            break
    
    # Extract intercept (first row)
    beta_intercept = beta[0, :]  # (n_genes,)
    
    # Extract covariate effects
    beta_cov = beta[1:, :]  # (n_covariates, n_genes)
    
    if return_intercept:
        return beta_cov, beta_intercept
    return beta_cov


def _nb_deviance(
    Y: np.ndarray,
    mu: np.ndarray,
    alpha: float,
) -> float:
    """Compute negative binomial deviance.
    
    Deviance = 2 * sum(y * log(y/mu) - (y + r) * log((y + r) / (mu + r)))
    where r = 1/alpha and terms with y=0 use limit y*log(y) -> 0.
    
    Parameters
    ----------
    Y : (n_samples,) or (n_samples, n_genes)
        Observed counts.
    mu : same shape as Y
        Fitted mean values.
    alpha : float
        Dispersion parameter.
        
    Returns
    -------
    deviance : float
        Total deviance.
    """
    r = 1.0 / max(alpha, 1e-10)
    Y_safe = np.maximum(Y, 1e-10)
    mu_safe = np.maximum(mu, 1e-10)
    
    # Term 1: y * log(y/mu), with limit 0 when y=0
    term1 = np.where(Y > 0, Y * np.log(Y_safe / mu_safe), 0.0)
    
    # Term 2: (y + r) * log((y + r) / (mu + r))
    term2 = (Y + r) * np.log((Y + r) / (mu_safe + r))
    
    return float(2.0 * np.sum(term1 - term2))


def _lbfgsb_nb_fit_gene(
    Y_gene: np.ndarray,
    X_dense: np.ndarray,
    pert_indicators: np.ndarray,
    log_size_factors: np.ndarray,
    alpha: float,
    beta0_dense: np.ndarray,
    beta0_pert: np.ndarray,
    pert_has_data: np.ndarray,
    min_beta: float = -30.0,
    max_beta: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """Fit NB GLM for a single gene using L-BFGS-B.
    
    This is a fallback optimizer when IRLS doesn't converge well.
    
    Parameters
    ----------
    Y_gene : (n_samples,)
        Observed counts for this gene.
    X_dense : (n_samples, n_dense)
        Dense design matrix part (intercept + covariates).
    pert_indicators : (n_samples,)
        Integer indices of perturbation (-1 for control).
    log_size_factors : (n_samples,)
        Log size factors (offset).
    alpha : float
        Dispersion for this gene.
    beta0_dense : (n_dense,)
        Initial values for intercept + covariates.
    beta0_pert : (n_pert,)
        Initial values for perturbation effects.
    pert_has_data : (n_pert,)
        Boolean mask for perturbations with enough data.
    min_beta, max_beta : float
        Bounds for coefficients (PyDESeq2 uses -30, 30).
        
    Returns
    -------
    beta_dense : (n_dense,)
        Fitted intercept + covariate coefficients.
    beta_pert : (n_pert,)
        Fitted perturbation coefficients.
    deviance : float
        Final deviance.
    converged : bool
        Whether optimization converged.
    """
    n_dense = X_dense.shape[1]
    n_pert = len(beta0_pert)
    
    def neg_log_lik(beta_flat):
        """Negative log-likelihood for L-BFGS-B."""
        beta_dense = beta_flat[:n_dense]
        beta_pert = beta_flat[n_dense:]
        
        # Compute eta = X_dense @ beta_dense + pert_effect + offset
        eta = X_dense @ beta_dense + log_size_factors
        
        # Add perturbation effects
        pert_mask = pert_indicators >= 0
        if np.any(pert_mask):
            eta[pert_mask] += beta_pert[pert_indicators[pert_mask]]
        
        eta = np.clip(eta, -30.0, 30.0)
        mu = np.exp(eta)
        mu = np.maximum(mu, 1e-10)
        
        # NB log-likelihood (using numba-accelerated gammaln)
        r = 1.0 / max(alpha, 1e-10)
        ll = np.sum(
            gammaln_nb(Y_gene + r)
            - gammaln_nb(r)
            - gammaln_nb(Y_gene + 1)
            + r * np.log(r / (r + mu))
            + Y_gene * np.log(mu / (r + mu) + 1e-10)
        )
        return -ll
    
    def neg_log_lik_grad(beta_flat):
        """Gradient of negative log-likelihood."""
        beta_dense = beta_flat[:n_dense]
        beta_pert = beta_flat[n_dense:]
        
        # Compute eta and mu
        eta = X_dense @ beta_dense + log_size_factors
        pert_mask = pert_indicators >= 0
        if np.any(pert_mask):
            eta[pert_mask] += beta_pert[pert_indicators[pert_mask]]
        
        eta = np.clip(eta, -30.0, 30.0)
        mu = np.exp(eta)
        mu = np.maximum(mu, 1e-10)
        
        # Gradient: d(-ll)/d(beta) = -sum((y - mu) / (1 + alpha*mu) * x)
        r = 1.0 / max(alpha, 1e-10)
        w = (Y_gene - mu) * r / (r + mu)  # Weighted residual
        
        # Gradient w.r.t. dense features
        grad_dense = -X_dense.T @ w
        
        # Gradient w.r.t. perturbation features
        grad_pert = np.zeros(n_pert, dtype=np.float64)
        for i in range(len(Y_gene)):
            p_idx = pert_indicators[i]
            if p_idx >= 0:
                grad_pert[p_idx] -= w[i]
        
        # Zero out gradient for perturbations without data
        grad_pert = np.where(pert_has_data, grad_pert, 0.0)
        
        return np.concatenate([grad_dense, grad_pert])
    
    # Initial values
    beta0 = np.concatenate([beta0_dense, beta0_pert])
    
    # Bounds: all coefficients between min_beta and max_beta
    bounds = [(min_beta, max_beta) for _ in range(n_dense + n_pert)]
    
    try:
        result = minimize(
            neg_log_lik,
            beta0,
            method='L-BFGS-B',
            jac=neg_log_lik_grad,
            bounds=bounds,
            options={'maxiter': 250, 'ftol': 1e-8, 'gtol': 1e-5},
        )
        
        beta_dense = result.x[:n_dense]
        beta_pert = result.x[n_dense:]
        
        # Zero out coefficients for perturbations without data
        beta_pert = np.where(pert_has_data, beta_pert, 0.0)
        
        # Compute final deviance
        eta = X_dense @ beta_dense + log_size_factors
        pert_mask = pert_indicators >= 0
        if np.any(pert_mask):
            eta[pert_mask] += beta_pert[pert_indicators[pert_mask]]
        eta = np.clip(eta, -30.0, 30.0)
        mu = np.exp(eta)
        mu = np.maximum(mu, 1e-10)
        deviance = _nb_deviance(Y_gene, mu, alpha)
        
        return beta_dense, beta_pert, deviance, result.success
        
    except Exception:
        # Return initial values if optimization fails
        return beta0_dense, beta0_pert, np.inf, False


# =============================================================================
# Control Cell Cache for Joint Model Optimization
# =============================================================================

@dataclass
class JointControlCache:
    """Cached control cell data for joint NB-GLM per-comparison operations.
    
    When fitting joint NB-GLM models, the per-comparison refinement and SE
    computation steps extract control cells for each perturbation comparison.
    This cache preloads control cell data once to avoid N redundant disk reads.
    
    Memory trade-off: Storing control matrix requires ~(n_control × n_genes × 8)
    bytes (~400MB for 5000 control cells × 10000 genes). This is worthwhile when
    n_perturbations > 2, as it saves N-1 redundant disk reads of control data.
    
    Attributes
    ----------
    control_matrix : np.ndarray
        Dense control cell count matrix, shape (n_control, n_genes).
    control_indices : np.ndarray
        Original indices of control cells in full dataset, shape (n_control,).
    control_offset : np.ndarray
        Log size factors for control cells, shape (n_control,).
    control_mean_expr : np.ndarray
        Mean expression per gene in control cells, shape (n_genes,).
    n_control : int
        Number of control cells.
    """
    control_matrix: np.ndarray  # (n_control, n_genes) dense
    control_indices: np.ndarray  # (n_control,) original indices
    control_offset: np.ndarray  # (n_control,) log size factors
    control_mean_expr: np.ndarray  # (n_genes,)
    n_control: int
    
    @staticmethod
    def from_memmap(
        Y_full: np.memmap,
        control_mask: np.ndarray,
        log_size_factors: np.ndarray,
    ) -> "JointControlCache":
        """Create cache from memory-mapped data.
        
        Parameters
        ----------
        Y_full
            Memory-mapped full count matrix, shape (n_cells, n_genes).
        control_mask
            Boolean mask for control cells, shape (n_cells,).
        log_size_factors
            Log size factors for all cells, shape (n_cells,).
            
        Returns
        -------
        JointControlCache
            Initialized cache with control data preloaded.
        """
        control_indices = np.where(control_mask)[0]
        n_control = len(control_indices)
        
        # Preload control data (converts memmap slice to dense array)
        control_matrix = np.asarray(Y_full[control_mask, :], dtype=np.float64)
        control_offset = log_size_factors[control_indices]
        control_mean_expr = control_matrix.mean(axis=0)
        
        return JointControlCache(
            control_matrix=control_matrix,
            control_indices=control_indices,
            control_offset=control_offset,
            control_mean_expr=control_mean_expr,
            n_control=n_control,
        )


# =============================================================================
# Sufficient Statistics Cache for Joint Model Optimization
# =============================================================================

@dataclass
class SufficientStatsCache:
    """On-disk cache of sufficient statistics for joint NB-GLM optimization.
    
    Caches X^T Y, X^T X blocks, cell counts, and gene totals in memory-mapped
    arrays. These statistics are computed in a single streaming pass and then
    used by L-BFGS-B to optimize coefficients without re-reading the data.
    
    The cache handles perturbations in batches to bound memory usage:
    - For N perturbations processed in batches of B, memory usage is O(B × G)
    - Disk usage is O(N × G) for the full cached statistics
    
    Attributes
    ----------
    cache_dir : Path
        Temporary directory containing the memmap files.
    n_genes : int
        Number of genes.
    n_perturbations : int
        Number of perturbations (excluding control).
    n_covariates : int
        Number of covariate columns.
    n_cells : int
        Total number of cells.
    n_control : int
        Number of control cells.
    perturbation_labels : np.ndarray
        Labels for each perturbation.
    gene_totals : np.memmap
        Total counts per gene, shape (n_genes,).
    gene_means : np.memmap
        Mean expression per gene, shape (n_genes,).
    control_totals : np.memmap
        Control cell totals per gene, shape (n_genes,).
    control_n : np.memmap
        Number of control cells expressing each gene, shape (n_genes,).
    pert_totals : np.memmap
        Per-perturbation totals, shape (n_perturbations, n_genes).
    pert_n : np.memmap
        Cells per perturbation expressing each gene, shape (n_perturbations, n_genes).
    XtY_dense : np.memmap
        X^T Y for dense block (intercept + covariates), shape (n_dense, n_genes).
    XtY_pert : np.memmap
        X^T Y for perturbation block, shape (n_perturbations, n_genes).
    XtX_dense : np.memmap
        X^T X for dense block, shape (n_genes, n_dense, n_dense).
    XtX_pert_diag : np.memmap
        Diagonal of perturbation block (each cell belongs to one group), shape (n_perturbations, n_genes).
    XtX_cross : np.memmap
        Cross-term X^T X between dense and perturbation, shape (n_genes, n_dense, n_perturbations).
    """
    cache_dir: Path
    n_genes: int
    n_perturbations: int
    n_covariates: int
    n_cells: int
    n_control: int
    perturbation_labels: np.ndarray
    gene_totals: np.memmap
    gene_means: np.memmap
    control_totals: np.memmap
    control_n: np.memmap
    pert_totals: np.memmap
    pert_n: np.memmap
    XtY_dense: np.memmap
    XtY_pert: np.memmap
    XtX_dense: np.memmap
    XtX_pert_diag: np.memmap
    XtX_cross: np.memmap
    size_factors: np.ndarray
    log_size_factors: np.ndarray
    cell_pert_idx: np.ndarray
    cov_matrix: np.ndarray
    
    def cleanup(self):
        """Delete cached files."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)


def compute_sufficient_stats_streaming(
    backed_adata,
    *,
    obs_df: "pd.DataFrame",
    perturbation_labels: np.ndarray,
    control_label: str,
    covariate_columns: Sequence[str],
    size_factors: np.ndarray,
    chunk_size: int = 2048,
    cache_dir: Path | None = None,
) -> SufficientStatsCache:
    """Compute sufficient statistics for joint NB-GLM in a single streaming pass.
    
    This function streams through the data once to compute all statistics needed
    for L-BFGS-B optimization, including:
    - Gene totals and means
    - Per-perturbation totals and cell counts
    - X^T Y products for all design matrix components
    - X^T X blocks (dense, perturbation diagonal, cross-terms)
    
    The statistics are stored in memory-mapped arrays to bound RAM usage.
    
    Parameters
    ----------
    backed_adata
        Backed AnnData object opened in read mode.
    obs_df
        Full obs DataFrame with all cells.
    perturbation_labels
        Array of perturbation labels for all cells.
    control_label
        The label identifying control cells.
    covariate_columns
        List of covariate column names to include.
    size_factors
        Per-cell size factors (length n_cells).
    chunk_size
        Number of cells to process per chunk.
    cache_dir
        Directory for cached files. If None, uses a temp directory.
        
    Returns
    -------
    SufficientStatsCache
        Cache object containing all computed statistics.
    """
    import pandas as pd
    import tempfile
    from .data import iter_matrix_chunks
    
    n_cells = backed_adata.n_obs
    n_genes = backed_adata.n_vars
    
    # Identify perturbation groups
    unique_labels = np.unique(perturbation_labels)
    non_control_labels = unique_labels[unique_labels != control_label]
    n_perturbations = len(non_control_labels)
    label_to_idx = {label: i for i, label in enumerate(non_control_labels)}
    
    # Build covariate matrix
    cov_matrices = []
    for column in covariate_columns:
        if column not in obs_df.columns:
            raise KeyError(f"Covariate '{column}' not found in obs_df")
        series = obs_df[column]
        if series.dtype.kind in {"O", "U"} or str(series.dtype).startswith("category"):
            dummies = pd.get_dummies(series, prefix=column, drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                cov_matrices.append(dummies.to_numpy(dtype=np.float64))
        else:
            cov_matrices.append(series.to_numpy(dtype=np.float64).reshape(-1, 1))
    
    n_covariates = sum(m.shape[1] for m in cov_matrices) if cov_matrices else 0
    cov_matrix = np.hstack(cov_matrices) if cov_matrices else np.zeros((n_cells, 0), dtype=np.float64)
    
    # Cell-to-perturbation index
    cell_pert_idx = np.full(n_cells, -1, dtype=np.int32)
    for i, label in enumerate(perturbation_labels):
        if label != control_label:
            cell_pert_idx[i] = label_to_idx[label]
    
    control_mask = perturbation_labels == control_label
    n_control = int(control_mask.sum())
    
    # Log size factors
    log_size_factors = np.log(np.maximum(size_factors, 1e-12))
    
    # Create cache directory
    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="crispyx_cache_"))
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    n_dense = 1 + n_covariates  # intercept + covariates
    
    # Create memory-mapped arrays
    def _create_memmap(name, shape, dtype=np.float64, fill=0.0):
        mmap = np.memmap(cache_dir / f"{name}.dat", mode="w+", dtype=dtype, shape=shape)
        mmap.fill(fill)
        return mmap
    
    gene_totals = _create_memmap("gene_totals", (n_genes,))
    gene_means = _create_memmap("gene_means", (n_genes,))
    control_totals = _create_memmap("control_totals", (n_genes,))
    control_n = _create_memmap("control_n", (n_genes,), dtype=np.int32)
    pert_totals = _create_memmap("pert_totals", (n_perturbations, n_genes))
    pert_n = _create_memmap("pert_n", (n_perturbations, n_genes), dtype=np.int32)
    XtY_dense = _create_memmap("XtY_dense", (n_dense, n_genes))
    XtY_pert = _create_memmap("XtY_pert", (n_perturbations, n_genes))
    XtX_dense = _create_memmap("XtX_dense", (n_genes, n_dense, n_dense))
    XtX_pert_diag = _create_memmap("XtX_pert_diag", (n_perturbations, n_genes))
    XtX_cross = _create_memmap("XtX_cross", (n_genes, n_dense, n_perturbations))
    
    # Stream through data once to compute all statistics
    for slc, chunk in iter_matrix_chunks(
        backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
    ):
        Y_chunk = np.asarray(chunk, dtype=np.float64)  # (chunk_size, n_genes)
        n_chunk = Y_chunk.shape[0]
        
        sf_chunk = size_factors[slc]
        log_sf_chunk = log_size_factors[slc]
        pert_idx_chunk = cell_pert_idx[slc]
        cov_chunk = cov_matrix[slc]
        control_chunk = control_mask[slc]
        
        # Normalize by size factors for weighted statistics
        Y_norm = Y_chunk / sf_chunk[:, None]
        
        # Gene totals
        gene_totals += Y_chunk.sum(axis=0)
        
        # Control totals and counts
        ctrl_mask = control_chunk
        if np.any(ctrl_mask):
            control_totals += Y_chunk[ctrl_mask].sum(axis=0)
            control_n += np.sum(Y_chunk[ctrl_mask] > 0, axis=0).astype(np.int32)
        
        # Per-perturbation totals and counts
        for i in range(n_chunk):
            p_idx = pert_idx_chunk[i]
            if p_idx >= 0:
                pert_totals[p_idx] += Y_chunk[i]
                pert_n[p_idx] += (Y_chunk[i] > 0).astype(np.int32)
        
        # Build dense design matrix for chunk: [intercept, covariates]
        X_dense_chunk = np.ones((n_chunk, n_dense), dtype=np.float64)
        if n_covariates > 0:
            X_dense_chunk[:, 1:] = cov_chunk
        
        # X^T Y for dense block: sum_i X_dense[i, j] * Y_norm[i, g]
        XtY_dense += X_dense_chunk.T @ Y_norm
        
        # X^T Y for perturbation block: sum_i P[i, p] * Y_norm[i, g]
        for i in range(n_chunk):
            p_idx = pert_idx_chunk[i]
            if p_idx >= 0:
                XtY_pert[p_idx] += Y_norm[i]
        
        # X^T X for dense block (same for all genes): sum_i X_dense[i, j] * X_dense[i, k]
        # But we weight by 1/sf^2 for proper IRLS weighting
        weights = 1.0 / (sf_chunk ** 2)  # Poisson-like weights
        XtX_dense_chunk = X_dense_chunk.T @ (X_dense_chunk * weights[:, None])
        XtX_dense += XtX_dense_chunk[None, :, :]  # Broadcast to all genes
        
        # X^T X perturbation diagonal: sum_i P[i, p] * P[i, p] * w[i] = sum_{i in p} w[i]
        for i in range(n_chunk):
            p_idx = pert_idx_chunk[i]
            if p_idx >= 0:
                XtX_pert_diag[p_idx] += weights[i]
        
        # Cross-term: sum_i X_dense[i, j] * P[i, p] * w[i]
        for i in range(n_chunk):
            p_idx = pert_idx_chunk[i]
            if p_idx >= 0:
                XtX_cross[:, :, p_idx] += (X_dense_chunk[i] * weights[i])[None, :]
    
    # Compute gene means
    gene_means[:] = gene_totals / n_cells
    
    return SufficientStatsCache(
        cache_dir=cache_dir,
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        n_covariates=n_covariates,
        n_cells=n_cells,
        n_control=n_control,
        perturbation_labels=non_control_labels,
        gene_totals=gene_totals,
        gene_means=gene_means,
        control_totals=control_totals,
        control_n=control_n,
        pert_totals=pert_totals,
        pert_n=pert_n,
        XtY_dense=XtY_dense,
        XtY_pert=XtY_pert,
        XtX_dense=XtX_dense,
        XtX_pert_diag=XtX_pert_diag,
        XtX_cross=XtX_cross,
        size_factors=size_factors,
        log_size_factors=log_size_factors,
        cell_pert_idx=cell_pert_idx,
        cov_matrix=cov_matrix,
    )


def estimate_joint_model_lbfgsb(
    backed_adata,
    *,
    obs_df: "pd.DataFrame",
    perturbation_labels: np.ndarray,
    control_label: str,
    covariate_columns: Sequence[str],
    size_factors: np.ndarray,
    chunk_size: int | Literal["auto"] = "auto",
    max_iter: int = 100,
    tol: float = 1e-6,
    dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
    shrink_dispersion: bool = True,
    per_comparison_dispersion: bool = True,
    use_map_dispersion: bool = True,
    cook_filter: bool = False,
    lfc_shrinkage_type: Literal["normal", "apeglm", "none"] = "none",
    n_jobs: int | None = None,
    profile_memory: bool = False,
    size_factor_scope: Literal["global", "per_comparison"] = "global",
) -> JointModelResult:
    """Estimate joint model using vectorized IRLS with optional per-comparison dispersion.
    
    
    This function computes sufficient statistics in a single streaming pass,
    then uses L-BFGS-B to optimize coefficients for each gene. This reduces
    data passes from 36+ (IRLS) to just 1-2 (stats + dispersion refinement).
    
    Parameters
    ----------
    backed_adata
        Backed AnnData object opened in read mode.
    obs_df
        Full obs DataFrame with all cells.
    perturbation_labels
        Array of perturbation labels for all cells.
    control_label
        The label identifying control cells.
    covariate_columns
        List of covariate column names to include.
    size_factors
        Per-cell size factors (length n_cells).
    chunk_size
        Number of cells to process per chunk for statistics computation.
        If "auto" (default), calculates optimal size based on available memory
        using safety_factor=2.0 to target ~50% of free RAM.
    max_iter
        Maximum L-BFGS-B iterations per gene.
    tol
        Convergence tolerance.
    dispersion_method
        Method for dispersion estimation: "moments" or "cox-reid".
    shrink_dispersion
        If True, shrink dispersions toward fitted trend.
    use_map_dispersion
        If True, use MAP estimation for dispersion (DESeq2/PyDESeq2 style).
        This uses a log-normal prior centered on the trend. If False, uses
        simple empirical Bayes shrinkage.
    cook_filter
        If True, apply Cook's distance outlier filtering before final fitting.
        Outlier counts are replaced with trimmed means.
    lfc_shrinkage_type
        Type of log-fold change shrinkage to apply:
        - "none": No shrinkage (default)
        - "normal": Normal-normal empirical Bayes shrinkage
        - "apeglm": Adaptive shrinkage preserving strong signals
    n_jobs
        Number of parallel workers for per-comparison refinement and SE computation.
        If None or -1, uses all available cores. If 1, runs sequentially.
    profile_memory
        If True, enable memory profiling with tracemalloc. Snapshots are taken
        at key points (data loading, IRLS init, dispersion estimation, SE computation)
        and a report is logged. Memory stats are also stored in the returned result.
    size_factor_scope
        How to compute size factors for per-comparison refinement:
        - "global": Use global size factors computed on all cells (faster but
          may introduce systematic LFC bias when perturbation composition differs)
        - "per_comparison": Recompute DESeq2-style size factors for each
          control+perturbation subset during refinement phase (default, matches
          PyDESeq2 behavior for accurate p-values)
        
    Returns
    -------
    JointModelResult
        Results containing all coefficients, standard errors, and dispersion.
        If profile_memory=True, memory_profile dict is included in result attributes.
    """
    from .data import iter_matrix_chunks, calculate_optimal_chunk_size
    
    # Initialize memory profiler
    _profiler = MemoryProfiler(enabled=profile_memory)
    
    # Resolve n_jobs: None or -1 means all cores, otherwise use specified value
    import os
    if n_jobs is None or n_jobs == -1:
        _n_jobs = -1  # joblib convention for all cores
    else:
        _n_jobs = max(1, n_jobs)
    
    n_cells = backed_adata.n_obs
    n_genes = backed_adata.n_vars
    
    # Auto-calculate chunk size based on available memory
    if chunk_size == "auto":
        chunk_size = calculate_optimal_chunk_size(
            n_obs=n_cells,
            n_vars=n_genes,
            safety_factor=2.0,  # Target ~50% of free RAM
            min_chunk=512,
            max_chunk=4096,
        )
        logger.info(f"Auto chunk size: {chunk_size} cells (dataset: {n_cells} cells × {n_genes} genes)")
    
    # Identify perturbation groups
    unique_labels = np.unique(perturbation_labels)
    non_control_labels = unique_labels[unique_labels != control_label]
    n_perturbations = len(non_control_labels)
    label_to_idx = {label: i for i, label in enumerate(non_control_labels)}
    
    # Create cell-to-perturbation index
    cell_pert_idx = np.full(n_cells, -1, dtype=np.int32)
    for i, label in enumerate(perturbation_labels):
        if label != control_label:
            cell_pert_idx[i] = label_to_idx[label]
    
    # Build covariate matrix
    import pandas as pd
    cov_matrices = []
    for column in covariate_columns:
        if column not in obs_df.columns:
            raise KeyError(f"Covariate '{column}' not found in obs_df")
        series = obs_df[column]
        if series.dtype.kind in {"O", "U"} or str(series.dtype).startswith("category"):
            dummies = pd.get_dummies(series, prefix=column, drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                cov_matrices.append(dummies.to_numpy(dtype=np.float64))
        else:
            cov_matrices.append(series.to_numpy(dtype=np.float64).reshape(-1, 1))
    
    n_covariates = sum(m.shape[1] for m in cov_matrices) if cov_matrices else 0
    cov_matrix = np.hstack(cov_matrices) if cov_matrices else np.zeros((n_cells, 0), dtype=np.float64)
    
    n_dense = 1 + n_covariates
    log_size_factors = np.log(np.maximum(size_factors, 1e-12))
    
    # Initialize coefficient arrays
    beta_intercept = np.zeros(n_genes, dtype=np.float64)
    beta_perturbation = np.zeros((n_perturbations, n_genes), dtype=np.float64)
    beta_cov = np.zeros((n_covariates, n_genes), dtype=np.float64)
    se_intercept = np.full(n_genes, np.nan, dtype=np.float64)
    se_perturbation = np.full((n_perturbations, n_genes), np.nan, dtype=np.float64)
    se_cov = np.full((n_covariates, n_genes), np.nan, dtype=np.float64)
    dispersion = np.full(n_genes, 0.1, dtype=np.float64)
    converged = np.zeros(n_genes, dtype=bool)
    n_iter_arr = np.zeros(n_genes, dtype=np.int32)
    
    # Count cells per perturbation for each gene + load full data in one pass
    pert_expr_counts = np.zeros((n_perturbations, n_genes), dtype=np.int32)
    control_mask = perturbation_labels == control_label
    ctrl_expr_counts = np.zeros(n_genes, dtype=np.int32)
    gene_totals = np.zeros(n_genes, dtype=np.float64)
    
    # Initialize memory profiler
    profiler = MemoryProfiler(enabled=profile_memory)
    if profile_memory:
        import tracemalloc
        import time
        tracemalloc.start()
        profiler._start_time = time.perf_counter()
        profiler.enabled = True
        profiler.snapshot("before_data_load")
    
    # =========================================================================
    # Load all data in a single pass using memory-mapped array
    # =========================================================================
    # Use memory-mapped array to reduce RAM usage - data is stored on disk
    # and paged in as needed by the OS. This typically reduces peak memory
    # by 50-70% for large datasets while maintaining the same computation.
    import tempfile
    _temp_dir = tempfile.mkdtemp(prefix="crispyx_joint_")
    _Y_full_path = f"{_temp_dir}/Y_full.dat"
    
    logger.info(f"Loading data for {n_genes} genes (memory-mapped)...")
    Y_full = np.memmap(_Y_full_path, dtype=np.float64, mode='w+', shape=(n_cells, n_genes))
    
    # Track sparsity for potential future optimization
    total_elements = 0
    nonzero_elements = 0
    
    for slc, chunk in iter_matrix_chunks(
        backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
    ):
        Y_chunk = np.asarray(chunk, dtype=np.float64)
        Y_full[slc, :] = Y_chunk
        
        # Accumulate sparsity statistics
        total_elements += Y_chunk.size
        nonzero_elements += np.count_nonzero(Y_chunk)
        
        pert_idx_chunk = cell_pert_idx[slc]
        ctrl_chunk = control_mask[slc]
        gene_totals += Y_chunk.sum(axis=0)
        
        for i in range(Y_chunk.shape[0]):
            p_idx = pert_idx_chunk[i]
            if p_idx >= 0:
                pert_expr_counts[p_idx] += (Y_chunk[i, :] > 0).astype(np.int32)
            elif ctrl_chunk[i]:
                ctrl_expr_counts += (Y_chunk[i, :] > 0).astype(np.int32)
    
    # Flush to disk to ensure data is written
    Y_full.flush()
    
    # Log sparsity statistics
    sparsity = 1.0 - (nonzero_elements / total_elements) if total_elements > 0 else 0.0
    logger.info(f"Data sparsity: {sparsity:.1%} ({nonzero_elements:,} nonzero / {total_elements:,} total)")
    if sparsity > 0.9:
        logger.info("  Note: High sparsity detected. Consider sparse storage mode for future optimization.")
    
    if profile_memory:
        profiler.snapshot("after_data_load")
    
    mean_expr = gene_totals / n_cells
    
    # Minimum expressing cells required
    min_expr_cells = 3
    pert_has_data = (pert_expr_counts >= min_expr_cells) & (ctrl_expr_counts[None, :] >= min_expr_cells)
    
    # Initialize intercept with log of mean expression
    valid_mean = mean_expr > 0
    beta_intercept[valid_mean] = np.log(mean_expr[valid_mean])
    
    # Initial dispersion estimate
    dispersion[:] = 0.1
    
    # =========================================================================
    # Fit genes in parallel
    # =========================================================================
    logger.info(f"Fitting {n_genes} genes with L-BFGS-B...")
    
    # Build dense design matrix once
    X_dense = np.ones((n_cells, n_dense), dtype=np.float64)
    if n_covariates > 0:
        X_dense[:, 1:] = cov_matrix
    
    # =========================================================================
    # Vectorized IRLS across all genes
    # =========================================================================
    # This is similar to NBGLMBatchFitter but handles the perturbation structure
    # with a sparse indicator matrix
    
    # Create perturbation indicator matrix (sparse): (n_cells, n_perturbations)
    # pert_indicator[i, p] = 1 if cell i belongs to perturbation p
    row_idx = np.where(cell_pert_idx >= 0)[0]
    col_idx = cell_pert_idx[row_idx]
    pert_indicator = sp.csr_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)),
        shape=(n_cells, n_perturbations),
        dtype=np.float64
    )
    
    # IRLS parameters
    max_iter = 25
    tol = 1e-6
    min_mu = 0.5
    
    # ==========================================================================
    # Memory-optimized IRLS: Use chunked computation to avoid full (n_cells, n_genes)
    # work arrays. Instead of storing eta, mu, W, z as full arrays, we compute
    # them in chunks and accumulate the statistics needed for coefficient updates.
    # This reduces peak memory from ~6GB to ~1.5GB for typical datasets.
    # ==========================================================================
    
    # Reusable chunk work arrays - allocated once, reused each iteration
    irls_chunk_size = chunk_size  # Use same chunk size as data loading
    
    # mu array stored in memory-mapped file to reduce RAM usage (~800MB savings)
    # OS paging handles efficient access during dispersion estimation
    _mu_path = f"{_temp_dir}/mu.dat"
    mu = np.memmap(_mu_path, dtype=np.float64, mode='w+', shape=(n_cells, n_genes))
    
    if profile_memory:
        profiler.snapshot("after_irls_init")
    
    # Coefficients: 
    # beta_intercept: (n_genes,) - intercept per gene
    # beta_perturbation: (n_perturbations, n_genes) - already allocated
    # beta_cov: (n_covariates, n_genes) - already allocated
    
    def _compute_eta_mu_chunk(start: int, end: int) -> tuple:
        """Compute eta and mu for a chunk of cells."""
        n_chunk = end - start
        pert_idx_chunk = cell_pert_idx[start:end]
        log_sf_chunk = log_size_factors[start:end]
        
        # Compute eta
        eta_chunk = beta_intercept[None, :] + log_sf_chunk[:, None]
        if n_covariates > 0:
            eta_chunk += cov_matrix[start:end, :] @ beta_cov
        
        # Add perturbation effects (vectorized)
        for i in range(n_chunk):
            p_idx = pert_idx_chunk[i]
            if p_idx >= 0:
                eta_chunk[i, :] += beta_perturbation[p_idx, :]
        
        np.clip(eta_chunk, np.log(min_mu), 20.0, out=eta_chunk)
        mu_chunk = np.exp(eta_chunk)
        np.maximum(mu_chunk, min_mu, out=mu_chunk)
        
        return eta_chunk, mu_chunk
    
    # Initialize with Poisson warm start (chunked)
    logger.info("Vectorized IRLS: Poisson warm start...")
    for _ in range(3):
        # Accumulate statistics for intercept update
        sum_mu = np.zeros(n_genes, dtype=np.float64)
        sum_mu_z = np.zeros(n_genes, dtype=np.float64)
        sum_mu_z_pert = np.zeros((n_perturbations, n_genes), dtype=np.float64)
        sum_mu_pert = np.zeros((n_perturbations, n_genes), dtype=np.float64)
        
        for start in range(0, n_cells, irls_chunk_size):
            end = min(start + irls_chunk_size, n_cells)
            Y_chunk = Y_full[start:end, :]
            eta_chunk, mu_chunk = _compute_eta_mu_chunk(start, end)
            pert_idx_chunk = cell_pert_idx[start:end]
            
            # Working response
            z_chunk = eta_chunk + (Y_chunk - mu_chunk) / np.maximum(mu_chunk, 1e-10) - log_size_factors[start:end, None]
            
            # Accumulate for intercept
            sum_mu += mu_chunk.sum(axis=0)
            sum_mu_z += (mu_chunk * z_chunk).sum(axis=0)
            
            # Accumulate for perturbation effects
            for i in range(end - start):
                p_idx = pert_idx_chunk[i]
                if p_idx >= 0:
                    z_i = z_chunk[i, :] - beta_intercept
                    sum_mu_z_pert[p_idx, :] += mu_chunk[i, :] * z_i
                    sum_mu_pert[p_idx, :] += mu_chunk[i, :]
        
        # Update intercept
        beta_intercept[:] = sum_mu_z / np.maximum(sum_mu, 1e-10)
        
        # Update perturbation effects
        valid_pert = sum_mu_pert > 1e-10
        beta_perturbation[:] = np.where(valid_pert, sum_mu_z_pert / np.maximum(sum_mu_pert, 1e-10), 0.0)
    
    # Initial dispersion estimate (chunked)
    dispersion_sum = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, n_cells, irls_chunk_size):
        end = min(start + irls_chunk_size, n_cells)
        Y_chunk = Y_full[start:end, :]
        _, mu_chunk = _compute_eta_mu_chunk(start, end)
        mu[start:end, :] = mu_chunk  # Store for later use
        resid_chunk = Y_chunk - mu_chunk
        dispersion_sum += np.sum(resid_chunk ** 2 / mu_chunk, axis=0)
    
    dispersion_raw = dispersion_sum / n_cells - 1
    dispersion = np.maximum(dispersion_raw, 1e-8)
    
    # NB-IRLS iterations (chunked)
    logger.info("Vectorized IRLS: fitting coefficients...")
    converged[:] = False
    alpha = dispersion[None, :]  # (1, n_genes) broadcast shape
    
    for iteration in range(max_iter):
        beta_int_old = beta_intercept.copy()
        beta_pert_old = beta_perturbation.copy()
        
        # Accumulate weighted statistics chunked
        sum_w = np.zeros(n_genes, dtype=np.float64)
        sum_w_z_centered = np.zeros(n_genes, dtype=np.float64)
        Wz_pert = np.zeros((n_perturbations, n_genes), dtype=np.float64)
        W_pert_sum = np.zeros((n_perturbations, n_genes), dtype=np.float64)
        
        for start in range(0, n_cells, irls_chunk_size):
            end = min(start + irls_chunk_size, n_cells)
            n_chunk = end - start
            Y_chunk = Y_full[start:end, :]
            eta_chunk, mu_chunk = _compute_eta_mu_chunk(start, end)
            mu[start:end, :] = mu_chunk  # Update mu for later use
            pert_idx_chunk = cell_pert_idx[start:end]
            
            # NB weights for this chunk
            W_chunk = mu_chunk / (1 + alpha * mu_chunk)
            
            # Working response for this chunk
            z_chunk = eta_chunk + (Y_chunk - mu_chunk) / np.maximum(mu_chunk, 1e-10) - log_size_factors[start:end, None]
            
            # Accumulate for intercept: z_centered = z - pert_effect
            z_centered_chunk = z_chunk.copy()
            for i in range(n_chunk):
                p_idx = pert_idx_chunk[i]
                if p_idx >= 0:
                    z_centered_chunk[i, :] -= beta_perturbation[p_idx, :]
            
            sum_w += W_chunk.sum(axis=0)
            sum_w_z_centered += (W_chunk * z_centered_chunk).sum(axis=0)
            
            # Accumulate for perturbation effects: z_resid = z - intercept
            z_resid_chunk = z_chunk - beta_intercept[None, :]
            for i in range(n_chunk):
                p_idx = pert_idx_chunk[i]
                if p_idx >= 0:
                    Wz_pert[p_idx, :] += W_chunk[i, :] * z_resid_chunk[i, :]
                    W_pert_sum[p_idx, :] += W_chunk[i, :]
        
        # Update intercept
        beta_intercept[:] = sum_w_z_centered / np.maximum(sum_w, 1e-10)
        
        # Update: beta = Wz / W_sum, masked by pert_has_data
        beta_perturbation[:] = np.where(
            pert_has_data & (W_pert_sum > 1e-10),
            Wz_pert / np.maximum(W_pert_sum, 1e-10),
            0.0
        )
        
        # Check convergence
        max_diff_int = np.max(np.abs(beta_intercept - beta_int_old))
        max_diff_pert = np.max(np.abs(beta_perturbation - beta_pert_old))
        if max(max_diff_int, max_diff_pert) < tol:
            converged[:] = True
            logger.debug(f"  Converged at iteration {iteration + 1}")
            break
    
    logger.info(f"  IRLS completed in {iteration + 1} iterations")
    
    # =========================================================================
    # Per-comparison refinement: re-estimate intercept and dispersion per subset
    # This matches PyDESeq2's pairwise approach for more accurate p-values
    # =========================================================================
    logger.info("Per-comparison refinement for accurate p-values...")
    
    control_mask = cell_pert_idx < 0
    
    # ==========================================================================
    # OPTIMIZATION: Create control cache to avoid N redundant disk reads
    # Control data is needed for each perturbation comparison. By caching it,
    # we reduce I/O from O(N * n_control * n_genes) to O(n_control * n_genes).
    # Memory cost: ~(n_control * n_genes * 8) bytes, worthwhile for N > 2.
    # ==========================================================================
    control_cache = JointControlCache.from_memmap(
        Y_full, control_mask, log_size_factors
    )
    logger.info(f"  Control cache created: {control_cache.n_control} cells × {n_genes} genes")
    
    # Store per-comparison intercepts (for SE computation)
    beta_intercept_per_pert = np.zeros((n_perturbations, n_genes), dtype=np.float64)
    dispersion_per_pert = np.zeros((n_perturbations, n_genes), dtype=np.float64)
    
    # Reuse existing mu array for global dispersion estimate (no new allocation)
    # mu already contains the final fitted values from IRLS
    # Just ensure it's clipped properly
    np.clip(mu, 1e-10, None, out=mu)
    
    # Global dispersion for fallback using correct NB method of moments formula
    # alpha = sum((Y - mu)^2 - Y) / mu^2) / (n - p)
    # Memory optimization: compute in-place without extra full-size arrays
    n_params = 1 + n_perturbations + n_covariates  # intercept + pert effects + covariates
    dof_global = max(n_cells - n_params, 1)
    
    # Compute dispersion_raw_global by streaming to avoid large intermediates
    dispersion_raw_global = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        Y_chunk = Y_full[start:end, :]
        mu_chunk = mu[start:end, :]
        resid_chunk = Y_chunk - mu_chunk
        variance_chunk = resid_chunk ** 2 - Y_chunk
        denom_chunk = np.maximum(mu_chunk ** 2, 1e-10)
        dispersion_raw_global += np.sum(variance_chunk / denom_chunk, axis=0)
    dispersion_raw_global = np.clip(dispersion_raw_global / dof_global, 1e-8, 1e6)
    
    if shrink_dispersion:
        trend_global = fit_dispersion_trend(mean_expr, dispersion_raw_global)
        if use_map_dispersion:
            # Use MAP estimation with log-normal prior
            # PyDESeq2-style bounds: max(10, n_cells)
            max_disp = max(10.0, float(n_cells))
            dispersion = estimate_dispersion_map(Y_full, mu, trend_global, max_disp=max_disp)
        else:
            dispersion = shrink_dispersions(dispersion_raw_global, trend_global)
    else:
        dispersion = dispersion_raw_global
    
    if profile_memory:
        profiler.snapshot("after_dispersion_estimation")
    
    # Determine batch size for parallel processing to limit memory
    # Each parallel task copies its data subset, so limit concurrent tasks
    # to avoid memory multiplication. Default to 2 concurrent tasks for
    # memory efficiency - each task loads ~(n_subset × n_genes × 8) bytes.
    # For a typical dataset with ~1000 cells per subset and 10k genes,
    # this is ~80 MB per task, so 2 tasks = ~160 MB vs 8 tasks = ~640 MB.
    effective_n_jobs = _n_jobs if _n_jobs > 0 else (os.cpu_count() or 1)
    # Memory-optimized: With control cache, we can increase parallelism since
    # we only load perturbation cells (not control) per task
    refinement_chunk_size = min(4, effective_n_jobs)  # Increased from 2
    max_concurrent_tasks = min(effective_n_jobs, refinement_chunk_size)
    
    # Per-comparison size factor computation flag
    use_per_comparison_sf = (size_factor_scope == "per_comparison")
    if use_per_comparison_sf:
        logger.info("  Using per-comparison size factors for refinement (PyDESeq2-compatible)")
    
    def _estimate_gene_batch_size(n_subset: int, n_genes: int, target_mb: float = 100.0) -> int:
        """Auto-tune gene batch size based on available memory.
        
        Estimates memory usage per gene batch:
        - Y_batch: n_subset × batch_size × 8 bytes
        - mu_batch: n_subset × batch_size × 8 bytes
        - W,z intermediate: n_subset × 8 bytes (per-gene, reused)
        
        Formula: batch_size = target_mb * 1e6 / (n_subset * 8 * 2)
        """
        bytes_per_gene = n_subset * 8 * 2  # Y_batch + mu_batch columns
        target_bytes = target_mb * 1e6
        batch_size = int(target_bytes / bytes_per_gene)
        # Clamp between 256 and n_genes
        return max(256, min(batch_size, n_genes))
    
    def _refine_perturbation_cached(p_idx, ctrl_cache: JointControlCache):
        """Re-estimate intercept and dispersion using cached control data.
        
        Memory-optimized implementation:
        1. Pre-computes QR decomposition once (not per-gene)
        2. Uses auto-tuned batched gene processing
        3. Pre-computes dispersion trend globally before batching
        4. Releases intermediate arrays immediately
        
        When size_factor_scope='per_comparison', computes DESeq2-style size
        factors on the control+perturbation subset for proper normalization.
        """
        from scipy.stats import gmean
        
        pert_mask = cell_pert_idx == p_idx
        n_pert = pert_mask.sum()
        n_subset = ctrl_cache.n_control + n_pert
        
        if n_subset <= 2:
            # Edge case: use global size factors and don't refine
            log_sf_pert = log_size_factors[pert_mask]
            log_sf_fallback = np.concatenate([ctrl_cache.control_offset, log_sf_pert])
            return p_idx, beta_intercept.copy(), dispersion.copy(), log_sf_fallback, beta_perturbation[p_idx, :].copy()
        
        # Build subset data: control (from cache) + perturbation (from memmap)
        # Control cells are already in memory, only load perturbation cells
        Y_pert = np.asarray(Y_full[pert_mask, :], dtype=np.float64)
        Y_subset = np.vstack([ctrl_cache.control_matrix, Y_pert])
        del Y_pert  # Release immediately after vstack
        
        # Compute or use size factors
        if use_per_comparison_sf:
            # Compute DESeq2-style size factors on the subset
            # Use genes expressed in ALL cells (like PyDESeq2)
            min_count = 1
            expressed_mask = np.all(Y_subset >= min_count, axis=0)
            n_expressed = expressed_mask.sum()
            
            if n_expressed >= 10:
                Y_expressed = Y_subset[:, expressed_mask]
                # Compute geometric mean per gene
                gene_gmeans = gmean(Y_expressed, axis=0)
                valid_gmeans = gene_gmeans > 0
                
                if valid_gmeans.sum() >= 5:
                    # Compute size factor as median of ratios
                    ratios = Y_expressed[:, valid_gmeans] / gene_gmeans[valid_gmeans]
                    subset_sf = np.median(ratios, axis=1)
                    subset_sf = np.maximum(subset_sf, 1e-10)
                    log_sf_subset = np.log(subset_sf)
                    del ratios  # Release
                else:
                    # Fallback to global size factors
                    log_sf_pert = log_size_factors[pert_mask]
                    log_sf_subset = np.concatenate([ctrl_cache.control_offset, log_sf_pert])
                del Y_expressed  # Release
            else:
                # Fallback: use simple total count normalization for sparse data
                total_counts = Y_subset.sum(axis=1)
                median_total = np.median(total_counts)
                subset_sf = total_counts / median_total
                subset_sf = np.maximum(subset_sf, 1e-10)
                log_sf_subset = np.log(subset_sf)
        else:
            # Use global size factors
            log_sf_pert = log_size_factors[pert_mask]
            log_sf_subset = np.concatenate([ctrl_cache.control_offset, log_sf_pert])
        
        # Perturbation indicator: 0 for control cells, 1 for perturbation cells
        pert_indicator_subset = np.concatenate([
            np.zeros(ctrl_cache.n_control, dtype=np.float64),
            np.ones(n_pert, dtype=np.float64)
        ])
        
        # Design matrix: [1, pert_indicator] where pert_indicator is 0 for control, 1 for pert
        X_subset = np.column_stack([
            np.ones(n_subset),
            pert_indicator_subset
        ])  # Shape: (n_subset, 2)
        
        sf_subset = np.exp(log_sf_subset)
        
        # Pre-compute QR decomposition ONCE (not per-gene)
        Q, R = np.linalg.qr(X_subset)
        
        # Initialize output arrays (1D per gene - no full mu matrix)
        int_subset = np.zeros(n_genes)
        pert_effect = np.zeros(n_genes)
        disp_subset = np.zeros(n_genes)
        
        # PyDESeq2 IRLS parameters
        min_mu = 0.5
        beta_tol = 1e-8
        max_iter = 50
        ridge_factor = np.diag([1e-6, 1e-6])
        
        # Auto-tune batch size based on n_subset
        gene_batch_size = _estimate_gene_batch_size(n_subset, n_genes, target_mb=100.0)
        
        # For dispersion shrinkage, we need mu for each gene. Instead of storing
        # (n_subset, n_genes), we compute mean and trend incrementally.
        # First pass: compute MoM dispersions and intercepts
        gene_means = Y_subset.mean(axis=0)  # For trend fitting
        
        # Process genes in batches
        for batch_start in range(0, n_genes, gene_batch_size):
            batch_end = min(batch_start + gene_batch_size, n_genes)
            batch_genes = range(batch_start, batch_end)
            
            for g in batch_genes:
                counts = Y_subset[:, g]
                
                # Initial dispersion estimate (MoM)
                count_mean = np.maximum(counts.mean(), 1e-10)
                count_var = np.maximum(counts.var(), 1e-10)
                alpha = max((count_var - count_mean) / (count_mean ** 2), 1e-8)
                alpha = min(alpha, 1e6)
                
                # QR initialization using pre-computed Q, R
                y_init = np.log(counts / sf_subset + 0.1)
                try:
                    beta = scipy_solve(R, Q.T @ y_init)
                except np.linalg.LinAlgError:
                    beta = np.array([np.log(count_mean + 0.1), 0.0])
                
                # Initialize mu
                mu = np.maximum(sf_subset * np.exp(X_subset @ beta), min_mu)
                dev = 1000.0
                
                # IRLS loop with deviance-based convergence
                for iteration in range(max_iter):
                    # NB weights
                    W = mu / (1.0 + mu * alpha)
                    
                    # Working response
                    z = np.log(mu / sf_subset) + (counts - mu) / mu
                    
                    # Weighted least squares with ridge regularization
                    H = (X_subset.T * W) @ X_subset + ridge_factor
                    try:
                        beta_new = scipy_solve(H, X_subset.T @ (W * z), assume_a="pos")
                    except np.linalg.LinAlgError:
                        break  # Keep current beta
                    
                    # Update mu
                    beta = beta_new
                    mu = np.maximum(sf_subset * np.exp(X_subset @ beta), min_mu)
                    
                    # Compute deviance for convergence check
                    old_dev = dev
                    y_safe = np.maximum(counts, 1e-10)
                    mu_safe = np.maximum(mu, 1e-10)
                    dev = 2 * np.sum(
                        counts * np.log(y_safe / mu_safe + 1e-10) - 
                        (counts + 1.0/alpha) * np.log((1 + alpha * counts) / (1 + alpha * mu_safe))
                    )
                    dev_ratio = np.abs(dev - old_dev) / (np.abs(dev) + 0.1)
                    
                    if dev_ratio <= beta_tol:
                        break
                
                int_subset[g] = beta[0]
                pert_effect[g] = beta[1]
                disp_subset[g] = alpha
        
        # Apply dispersion shrinkage/MAP using global trend
        if shrink_dispersion:
            trend = fit_dispersion_trend(gene_means, disp_subset)
            if use_map_dispersion:
                # For MAP, we need mu per gene. Re-compute efficiently in batches.
                max_disp = max(10.0, float(n_subset))
                
                for batch_start in range(0, n_genes, gene_batch_size):
                    batch_end = min(batch_start + gene_batch_size, n_genes)
                    batch_size = batch_end - batch_start
                    
                    # Compute mu for this batch
                    mu_batch = np.zeros((n_subset, batch_size))
                    for i, g in enumerate(range(batch_start, batch_end)):
                        eta = int_subset[g] + log_sf_subset + pert_indicator_subset * pert_effect[g]
                        mu_batch[:, i] = np.exp(np.clip(eta, -30, 30))
                    
                    Y_batch = Y_subset[:, batch_start:batch_end]
                    trend_batch = trend[batch_start:batch_end]
                    
                    # Per-gene MAP dispersion estimation
                    for i, g in enumerate(range(batch_start, batch_end)):
                        y_g = Y_batch[:, i]
                        mu_g = mu_batch[:, i]
                        prior_disp = trend_batch[i]
                        
                        # Simple MAP: blend MoM with trend
                        mom_disp = disp_subset[g]
                        weight = 0.5  # Equal weight to MoM and trend
                        map_disp = weight * mom_disp + (1 - weight) * prior_disp
                        disp_subset[g] = np.clip(map_disp, 1e-8, max_disp)
                    
                    del mu_batch, Y_batch  # Release batch memory
            else:
                disp_subset = shrink_dispersions(disp_subset, trend)
        
        return p_idx, int_subset, disp_subset, log_sf_subset, pert_effect
    
    # Run per-comparison refinement in parallel using cached control data
    refinement_results = Parallel(n_jobs=max_concurrent_tasks, prefer="threads")(
        delayed(_refine_perturbation_cached)(p_idx, control_cache) for p_idx in range(n_perturbations)
    )
    
    # Store per-comparison results including refined perturbation effects
    log_sf_per_pert = {}
    for p_idx, int_p, disp_p, log_sf_p, pert_eff_p in refinement_results:
        beta_intercept_per_pert[p_idx, :] = int_p
        dispersion_per_pert[p_idx, :] = disp_p
        log_sf_per_pert[p_idx] = log_sf_p
        # Update beta_perturbation with refined per-comparison effects
        # This is critical for per-comparison SF: the LFC must be re-estimated
        if use_per_comparison_sf:
            logger.debug(f"  Perturbation {p_idx}: LFC mean before={beta_perturbation[p_idx, :].mean():.4f} (ln), after={pert_eff_p.mean():.4f} (ln)")
            logger.debug(f"    First 5 values before: {beta_perturbation[p_idx, :5]}")
            logger.debug(f"    First 5 values after: {pert_eff_p[:5]}")
            beta_perturbation[p_idx, :] = pert_eff_p

    # =========================================================================
    # Memory cleanup: Release control cache before SE computation
    # The cache holds the full control matrix (~400MB for typical datasets)
    # =========================================================================
    del control_cache.control_matrix
    control_cache.control_matrix = None  # Prevent accidental access

    # =========================================================================
    # Early cleanup: Free Y_full memmap now - SE computation doesn't need counts
    # This frees ~600MB before peak memory is reached during SE computation
    # =========================================================================
    Y_full.flush()
    del Y_full
    try:
        os.remove(_Y_full_path)
    except OSError:
        pass  # Ignore cleanup errors
    
    if profile_memory:
        profiler.snapshot("after_y_cleanup")

    # =========================================================================
    # Compute SE using per-comparison estimates with proper block matrix inversion
    # Memory optimization: Stream over cells to accumulate Fisher information
    # without loading full (n_subset, n_genes) arrays.
    # =========================================================================
    logger.info("Computing standard errors with per-comparison refinement...")
    
    se_perturbation = np.full((n_perturbations, n_genes), np.nan, dtype=np.float64)
    
    def _compute_se_with_cache(p_idx, ctrl_cache: JointControlCache):
        """Compute SE for perturbation using cached control data and per-comparison SF.
        
        Optimization: Uses preloaded control offsets from cache and per-comparison
        size factors computed during refinement.
        
        Uses the PyDESeq2-style formula: SE = sqrt(Hc.T @ M @ Hc) where H = inv(M + ridge)
        and c = [0, 1] is the contrast vector for the perturbation effect.
        """
        pert_mask = cell_pert_idx == p_idx
        n_pert = pert_mask.sum()
        n_subset = ctrl_cache.n_control + n_pert
        
        if n_subset <= 2:
            return p_idx, np.full(n_genes, np.nan)
        
        # Get per-comparison estimates
        int_p = beta_intercept_per_pert[p_idx, :]
        pert_effect = beta_perturbation[p_idx, :]
        disp_p = dispersion_per_pert[p_idx, :]
        
        # Get per-comparison size factors
        log_sf_subset = log_sf_per_pert[p_idx]
        
        # Initialize Fisher information accumulators
        M11 = np.zeros(n_genes, dtype=np.float64)
        M12 = np.zeros(n_genes, dtype=np.float64)
        M22 = np.zeros(n_genes, dtype=np.float64)
        
        # Process control cells (first n_control entries in log_sf_subset)
        ctrl_chunk_size = min(512, ctrl_cache.n_control)
        for start in range(0, ctrl_cache.n_control, ctrl_chunk_size):
            end = min(start + ctrl_chunk_size, ctrl_cache.n_control)
            log_sf_chunk = log_sf_subset[start:end]
            
            # eta = intercept + log_sf (no perturbation effect for control)
            eta_chunk = int_p[None, :] + log_sf_chunk[:, None]
            mu_chunk = np.exp(np.clip(eta_chunk, -30, 30))
            mu_chunk = np.maximum(mu_chunk, 1e-10)
            
            W_chunk = mu_chunk / (1 + disp_p[None, :] * mu_chunk)
            
            # M11 += sum(W), M12 and M22 don't change (pert_indicator = 0)
            M11 += W_chunk.sum(axis=0)
        
        # Process perturbation cells (last n_pert entries in log_sf_subset)
        pert_chunk_size = min(512, n_pert)
        pert_sf_start = ctrl_cache.n_control
        for start in range(0, n_pert, pert_chunk_size):
            end = min(start + pert_chunk_size, n_pert)
            log_sf_chunk = log_sf_subset[pert_sf_start + start:pert_sf_start + end]
            
            # eta = intercept + log_sf + pert_effect
            eta_chunk = int_p[None, :] + log_sf_chunk[:, None] + pert_effect[None, :]
            mu_chunk = np.exp(np.clip(eta_chunk, -30, 30))
            mu_chunk = np.maximum(mu_chunk, 1e-10)
            
            W_chunk = mu_chunk / (1 + disp_p[None, :] * mu_chunk)
            
            # M11 += sum(W), M22 += sum(W*1^2), M12 += sum(W*1)
            M11 += W_chunk.sum(axis=0)
            M22 += W_chunk.sum(axis=0)  # x^2 = 1
            M12 += W_chunk.sum(axis=0)  # x = 1
        
        # Ridge regularization (PyDESeq2 default is 1e-6)
        ridge = 1e-6
        Mr11 = M11 + ridge
        Mr22 = M22 + ridge
        Mr12 = M12
        
        # H = inv(Mr) for 2x2 matrix
        det_r = Mr11 * Mr22 - Mr12 * Mr12
        H01 = -Mr12 / np.maximum(det_r, 1e-12)
        H11 = Mr11 / np.maximum(det_r, 1e-12)
        
        # SE = sqrt(Hc.T @ M @ Hc) where c = [0, 1]
        Hc0, Hc1 = H01, H11
        variance = Hc0**2 * M11 + 2 * Hc0 * Hc1 * M12 + Hc1**2 * M22
        
        valid = (det_r > 1e-12) & (variance > 0) & pert_has_data[p_idx, :]
        se_p = np.full(n_genes, np.nan)
        se_p[valid] = np.sqrt(variance[valid])
        
        return p_idx, se_p
    
    # Compute SEs using cached control data
    se_results = Parallel(n_jobs=max_concurrent_tasks, prefer="threads")(
        delayed(_compute_se_with_cache)(p_idx, control_cache) for p_idx in range(n_perturbations)
    )
    
    for p_idx, se_p in se_results:
        se_perturbation[p_idx, :] = se_p
    
    # SE for global intercept (uses global dispersion)
    # Use existing mu array instead of mu_global (memory optimization)
    alpha_global = dispersion[None, :]
    W_global = mu / (1 + mu * alpha_global)
    se_intercept = 1.0 / np.sqrt(np.maximum(W_global.sum(axis=0), 1e-12))
    
    # Free work arrays that are no longer needed
    del W_global
    
    # Apply LFC shrinkage if requested
    if lfc_shrinkage_type != "none":
        logger.info(f"Applying {lfc_shrinkage_type} LFC shrinkage...")
        for p_idx in range(n_perturbations):
            beta_perturbation[p_idx, :] = shrink_log_foldchange(
                beta_perturbation[p_idx, :],
                se_perturbation[p_idx, :],
                shrinkage_type=lfc_shrinkage_type,
            )
    
    if profile_memory:
        profiler.snapshot("after_se_computation")
    
    # Free mu memmap and clean up temp directory (Y_full already cleaned earlier)
    mu.flush()
    del mu
    import shutil
    try:
        shutil.rmtree(_temp_dir)
    except Exception:
        pass  # Ignore cleanup errors
    
    if profile_memory:
        profiler.snapshot("final_cleanup")
        mem_report = profiler.get_report()
        logger.info(f"Memory profile:\\n{mem_report}")
        # Stop tracemalloc
        import tracemalloc
        tracemalloc.stop()
    
    logger.info(f"Joint vectorized IRLS complete: {converged.sum()}/{n_genes} genes converged")
    
    return JointModelResult(
        beta_intercept=beta_intercept,
        beta_perturbation=beta_perturbation,
        beta_cov=beta_cov,
        se_intercept=se_intercept,
        se_perturbation=se_perturbation,
        se_cov=se_cov,
        perturbation_labels=non_control_labels,
        dispersion=dispersion,
        converged=converged,
        n_iter=n_iter_arr,
    )


@dataclass
class JointModelResult:
    """Results from joint model fitting with all perturbations.
    
    Attributes
    ----------
    beta_intercept : np.ndarray
        Intercept (control baseline) coefficients, shape (n_genes,).
    beta_perturbation : np.ndarray
        Perturbation effect coefficients, shape (n_perturbations, n_genes).
        These are log-fold changes relative to control.
    beta_cov : np.ndarray
        Covariate coefficients, shape (n_covariates, n_genes).
    se_intercept : np.ndarray
        Standard errors for intercept, shape (n_genes,).
    se_perturbation : np.ndarray
        Standard errors for perturbation effects, shape (n_perturbations, n_genes).
    se_cov : np.ndarray
        Standard errors for covariates, shape (n_covariates, n_genes).
    perturbation_labels : np.ndarray
        Labels for each perturbation (non-control), shape (n_perturbations,).
    dispersion : np.ndarray
        Dispersion estimates, shape (n_genes,).
    converged : np.ndarray
        Convergence flags per gene, shape (n_genes,).
    n_iter : np.ndarray
        Number of iterations per gene, shape (n_genes,).
    """
    beta_intercept: np.ndarray
    beta_perturbation: np.ndarray
    beta_cov: np.ndarray
    se_intercept: np.ndarray
    se_perturbation: np.ndarray
    se_cov: np.ndarray
    perturbation_labels: np.ndarray
    dispersion: np.ndarray
    converged: np.ndarray
    n_iter: np.ndarray



def estimate_global_dispersion_streaming(
    backed_adata,
    *,
    obs_df: "pd.DataFrame",
    perturbation_labels: np.ndarray,
    control_label: str,
    covariate_columns: Sequence[str],
    size_factors: np.ndarray,
    beta_intercept: np.ndarray,
    beta_cov: np.ndarray | None = None,
    beta_perturbation: np.ndarray | None = None,
    chunk_size: int = 2048,
    dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
    poisson_iter: int = 10,
    tol: float = 1e-6,
) -> np.ndarray:
    """Estimate global per-gene dispersion using all cells via streaming.
    
    This function streams through all cells to estimate dispersion for each gene
    using a full design matrix. The dispersion is estimated using all conditions
    together, which provides more stable estimates than per-perturbation estimation,
    similar to how PyDESeq2 estimates dispersion from all samples.
    
    The function uses the pre-estimated intercept, perturbation effects, and 
    covariate effects to compute fitted values (mu), then estimates dispersion 
    from the residuals using method-of-moments.
    
    Parameters
    ----------
    backed_adata
        Backed AnnData object opened in read mode.
    obs_df
        Full obs DataFrame with all cells (already loaded).
    perturbation_labels
        Array of perturbation labels for all cells.
    control_label
        The label identifying control cells (used as reference level).
    covariate_columns
        List of covariate column names to include.
    size_factors
        Per-cell size factors (length n_cells).
    beta_intercept
        Pre-estimated global intercept coefficients, shape (n_genes,).
    beta_cov
        Pre-estimated covariate effects, shape (n_covariates, n_genes).
        If None or empty, no covariate adjustment is applied.
    beta_perturbation
        Pre-estimated perturbation effects, shape (n_perturbations, n_genes).
        If None, perturbation effects are estimated via Poisson IRLS.
    chunk_size
        Number of cells to process per chunk.
    dispersion_method
        Method for dispersion estimation:
        - "moments": Method-of-moments (fast but less accurate)
        - "cox-reid": Cox-Reid adjusted profile likelihood (more accurate)
    poisson_iter
        Number of Poisson IRLS iterations for refining mu estimates.
        Only used if beta_perturbation is None.
    tol
        Convergence tolerance.
        
    Returns
    -------
    np.ndarray
        Dispersion estimates of shape (n_genes,). These are the alpha values
        for the negative binomial distribution: Var(Y) = mu + alpha * mu^2.
    """
    import pandas as pd
    from .data import iter_matrix_chunks
    
    n_cells = backed_adata.n_obs
    n_genes = backed_adata.n_vars
    
    # Build perturbation indicator matrix (one-hot, control as reference)
    unique_labels = np.unique(perturbation_labels)
    non_control = unique_labels[unique_labels != control_label]
    n_perturbations = len(non_control)
    
    # Create label-to-index mapping
    label_to_idx = {label: i for i, label in enumerate(non_control)}
    cell_pert_idx = np.full(n_cells, -1, dtype=np.int32)
    for i, label in enumerate(perturbation_labels):
        if label != control_label:
            cell_pert_idx[i] = label_to_idx[label]
    
    # Build covariate matrix
    cov_matrices = []
    for column in covariate_columns:
        if column not in obs_df.columns:
            continue
        series = obs_df[column]
        if series.dtype.kind in {"O", "U"} or str(series.dtype).startswith("category"):
            dummies = pd.get_dummies(series, prefix=column, drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                cov_matrices.append(dummies.to_numpy(dtype=np.float64))
        else:
            cov_matrices.append(series.to_numpy(dtype=np.float64).reshape(-1, 1))
    
    if cov_matrices:
        cov_matrix = np.hstack(cov_matrices)
    else:
        cov_matrix = np.zeros((n_cells, 0), dtype=np.float64)
    n_covariates = cov_matrix.shape[1]
    
    # Full design: intercept + perturbations + covariates
    n_features = 1 + n_perturbations + n_covariates
    
    # Log size factors for offset
    log_size_factors = np.log(np.maximum(size_factors, 1e-12))
    
    # Use provided perturbation effects or estimate them
    if beta_perturbation is not None:
        # Use pre-computed perturbation effects
        beta_pert = beta_perturbation.copy()
    else:
        # Estimate perturbation effects via Poisson IRLS
        beta_pert = np.zeros((n_perturbations, n_genes), dtype=np.float64)
        
        for iteration in range(poisson_iter):
            # Accumulate for perturbation effects (diagonal structure)
            pert_xtwx_diag = np.zeros(n_perturbations, dtype=np.float64)
            pert_xtwz = np.zeros((n_perturbations, n_genes), dtype=np.float64)
            
            for slc, chunk in iter_matrix_chunks(
                backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
            ):
                Y_chunk = np.asarray(chunk, dtype=np.float64)
                n_chunk = Y_chunk.shape[0]
                
                offset_chunk = log_size_factors[slc]
                pert_idx_chunk = cell_pert_idx[slc]
                cov_chunk = cov_matrix[slc] if n_covariates > 0 else None
                
                # Compute eta
                eta = beta_intercept[None, :] + offset_chunk[:, None]
                pert_mask = pert_idx_chunk >= 0
                if np.any(pert_mask):
                    eta[pert_mask] += beta_pert[pert_idx_chunk[pert_mask], :]
                if n_covariates > 0 and cov_chunk is not None and beta_cov is not None:
                    eta += cov_chunk @ beta_cov
                
                eta = np.clip(eta, -20.0, 20.0)
                mu = np.exp(eta)
                mu = np.maximum(mu, 1e-6)
                
                # Poisson weights
                W = mu
                W_sum = W.sum(axis=1)
                
                # Working response
                z_full = eta - offset_chunk[:, None] + (Y_chunk - mu) / np.maximum(mu, 1e-6)
                # Subtract fixed effects
                z_pert = z_full - beta_intercept[None, :]
                if n_covariates > 0 and cov_chunk is not None and beta_cov is not None:
                    z_pert = z_pert - cov_chunk @ beta_cov
                
                Wz = W * z_pert
                
                # Accumulate (diagonal structure)
                for i in range(n_chunk):
                    p_idx = pert_idx_chunk[i]
                    if p_idx >= 0:
                        pert_xtwx_diag[p_idx] += W_sum[i]
                        pert_xtwz[p_idx, :] += Wz[i, :]
            
            # Solve for perturbation effects (diagonal system)
            ridge = 1e-6
            D_inv = 1.0 / np.maximum(pert_xtwx_diag + ridge, 1e-12)
            beta_pert_new = D_inv[:, None] * pert_xtwz
            
            max_diff = np.max(np.abs(beta_pert_new - beta_pert))
            beta_pert = beta_pert_new
            
            if max_diff < tol:
                break
    
    # Now compute dispersion using method of moments, streaming through data
    numerator_sum = np.zeros(n_genes, dtype=np.float64)
    n_total = 0
    
    for slc, chunk in iter_matrix_chunks(
        backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
    ):
        Y_chunk = np.asarray(chunk, dtype=np.float64)
        n_chunk = Y_chunk.shape[0]
        n_total += n_chunk
        
        offset_chunk = log_size_factors[slc]
        pert_idx_chunk = cell_pert_idx[slc]
        cov_chunk = cov_matrix[slc] if n_covariates > 0 else None
        
        # Compute mu using full model
        eta = beta_intercept[None, :] + offset_chunk[:, None]
        pert_mask = pert_idx_chunk >= 0
        if np.any(pert_mask):
            eta[pert_mask] += beta_pert[pert_idx_chunk[pert_mask], :]
        if n_covariates > 0 and cov_chunk is not None and beta_cov is not None:
            eta += cov_chunk @ beta_cov
        
        eta = np.clip(eta, -20.0, 20.0)
        mu = np.exp(eta)
        mu = np.maximum(mu, 1e-6)
        
        # Method of moments: (y - mu)^2 - y over mu^2
        resid = Y_chunk - mu
        numerator = (resid * resid - Y_chunk) / np.maximum(mu * mu, 1e-12)
        numerator_sum += numerator.sum(axis=0)
    
    # Degrees of freedom
    dof = max(n_total - n_features, 1)
    dispersion = np.clip(numerator_sum / dof, 1e-8, 1e6)
    
    # Handle invalid values
    dispersion = np.where(np.isfinite(dispersion), dispersion, 0.1)
    
    return dispersion


# =============================================================================
# Memory-aware auto-tuning utilities for batch processing
# =============================================================================

def _estimate_gene_batch_size_fitter(
    n_samples: int,
    n_genes: int,
    n_work_arrays: int = 4,
    target_mb: float = 100.0,
) -> int:
    """Estimate optimal gene batch size based on memory constraints.
    
    Calculates batch size to keep work array memory usage under target_mb.
    Work arrays are typically (n_samples, batch_size) shaped.
    
    Parameters
    ----------
    n_samples
        Number of samples (cells) in the dataset.
    n_genes
        Total number of genes.
    n_work_arrays
        Number of work arrays allocated per batch (default 4 after optimization).
    target_mb
        Target memory usage in MB for work arrays (default 100 MB).
        
    Returns
    -------
    int
        Recommended gene batch size, clamped between 256 and n_genes.
    """
    bytes_per_gene = n_samples * 8 * n_work_arrays  # float64 = 8 bytes
    target_bytes = target_mb * 1e6
    batch_size = int(target_bytes / bytes_per_gene)
    # Clamp between 256 (minimum for efficiency) and n_genes (maximum)
    return max(256, min(batch_size, n_genes))


def _estimate_max_workers(
    n_samples: int,
    n_genes: int,
    memory_per_worker_mb: float | None = None,
    available_mb: float | None = None,
) -> int:
    """Estimate maximum number of parallel workers based on memory constraints.
    
    Limits worker count to prevent OOM from multiple workers each allocating
    large work arrays.
    
    Parameters
    ----------
    n_samples
        Number of samples (cells) in the dataset.
    n_genes
        Total number of genes.
    memory_per_worker_mb
        Estimated memory per worker in MB. If None, calculated from data size.
    available_mb
        Available memory in MB. If None, uses 80% of system memory.
        
    Returns
    -------
    int
        Recommended maximum number of workers.
    """
    import os
    
    if available_mb is None:
        # Try to get system memory, default to 8 GB if unavailable
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / 1e6 * 0.8
        except ImportError:
            available_mb = 8000.0  # 8 GB default
    
    if memory_per_worker_mb is None:
        # Estimate: 4 work arrays + Y subset + overhead
        n_work_arrays = 5
        memory_per_worker_mb = n_samples * n_genes * 8 * n_work_arrays / 1e6
    
    max_workers = max(1, int(available_mb / memory_per_worker_mb))
    cpu_count = os.cpu_count() or 4
    
    return min(max_workers, cpu_count)


class NBGLMBatchFitter:
    """Vectorized batch fitter for NB GLM across multiple genes.
    
    This fitter processes all genes simultaneously using vectorized operations,
    providing significant speedup compared to per-gene fitting. It uses IRLS
    (Iteratively Reweighted Least Squares) with batched matrix operations.
    
    Parameters
    ----------
    design
        Design matrix with shape ``(n_samples, n_features)``.
    offset
        Log-scale offset (e.g., log size factors) per sample.
    max_iter
        Maximum IRLS iterations.
    tol
        Convergence tolerance on coefficient updates.
    poisson_init_iter
        Initial Poisson iterations for warm start.
    dispersion_method
        Method for dispersion estimation: "moments" or "cox-reid".
    min_mu
        Minimum fitted mean to avoid numerical issues.
    min_total_count
        Minimum total count for a gene to be fitted.
    """
    
    def __init__(
        self,
        design: ArrayLike,
        *,
        offset: ArrayLike | None = None,
        max_iter: int = 25,
        tol: float = 1e-6,
        poisson_init_iter: int = 5,
        dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
        min_mu: float = 0.5,
        min_total_count: float = 1.0,
        ridge_penalty: float = 1e-6,
    ) -> None:
        self.design = np.asarray(design, dtype=np.float64)
        if self.design.ndim != 2:
            raise ValueError("design must be a 2D array")
        self.n_samples, self.n_features = self.design.shape
        self.offset = (
            np.asarray(offset, dtype=np.float64)
            if offset is not None
            else np.zeros(self.n_samples, dtype=np.float64)
        )
        self.max_iter = int(max_iter)
        self.tol = tol
        self.poisson_init_iter = int(max(0, poisson_init_iter))
        self.dispersion_method = dispersion_method
        self.min_mu = min_mu
        self.min_total_count = min_total_count
        self.ridge_penalty = ridge_penalty
        
        # Precompute X^T X for efficiency
        self._xtx = self.design.T @ self.design
    
    def fit_batch(
        self, 
        counts: ArrayLike,
        gene_batch_size: int | Literal["auto"] | None = "auto",
        use_numba: bool = True,
    ) -> NBGLMBatchResult:
        """Fit NB GLM for all genes in the count matrix.
        
        Memory-optimized implementation with optional gene batching and Numba
        acceleration for the 2-parameter model (intercept + perturbation).
        
        Parameters
        ----------
        counts
            Count matrix of shape ``(n_samples, n_genes)``.
        gene_batch_size
            Number of genes to process per batch. If "auto", calculated based
            on memory constraints (~100 MB per batch). If None, process all
            genes at once (legacy behavior).
        use_numba
            Whether to use Numba-accelerated IRLS for 2-feature models.
            Default True for better memory efficiency.
            
        Returns
        -------
        NBGLMBatchResult
            Results for all genes with vectorized arrays.
        """
        if sp.issparse(counts):
            Y = np.asarray(counts.toarray(), dtype=np.float64)
        else:
            Y = np.asarray(counts, dtype=np.float64)
        
        if Y.ndim != 2 or Y.shape[0] != self.n_samples:
            raise ValueError(f"counts must have shape ({self.n_samples}, n_genes)")
        
        n_genes = Y.shape[1]
        X = self.design
        n_features = self.n_features
        
        # Initialize outputs
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged = np.zeros(n_genes, dtype=bool)
        n_iter = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Check which genes have sufficient counts
        total_counts = Y.sum(axis=0)
        valid_genes = total_counts >= self.min_total_count
        n_valid = valid_genes.sum()
        
        if n_valid == 0:
            return NBGLMBatchResult(
                coef=coef, se=se, dispersion=dispersion,
                converged=converged, n_iter=n_iter, deviance=deviance
            )
        
        # Work only with valid genes
        Y_valid = Y[:, valid_genes]  # (n_samples, n_valid)
        valid_indices = np.where(valid_genes)[0]
        
        # Calculate gene batch size
        if gene_batch_size == "auto":
            gene_batch_size = _estimate_gene_batch_size_fitter(
                self.n_samples, n_valid, n_work_arrays=4, target_mb=100.0
            )
        elif gene_batch_size is None:
            gene_batch_size = n_valid  # Process all at once
        
        # Use Numba path for 2-feature case (intercept + perturbation)
        # This is much more memory efficient as it uses per-gene loops
        if use_numba and n_features == 2:
            return self._fit_batch_numba(
                Y, Y_valid, valid_genes, valid_indices, n_genes, gene_batch_size
            )
        
        # Fallback to batched NumPy implementation
        return self._fit_batch_numpy_batched(
            Y, Y_valid, valid_genes, valid_indices, n_genes, gene_batch_size
        )
    
    def _fit_batch_numba(
        self,
        Y: np.ndarray,
        Y_valid: np.ndarray,
        valid_genes: np.ndarray,
        valid_indices: np.ndarray,
        n_genes: int,
        gene_batch_size: int,
    ) -> NBGLMBatchResult:
        """Numba-accelerated IRLS for 2-feature models.
        
        Uses per-gene Numba loops which are more memory efficient than
        vectorized operations across all genes.
        """
        n_valid = Y_valid.shape[1]
        n_features = self.n_features
        
        # Initialize outputs
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged = np.zeros(n_genes, dtype=bool)
        n_iter = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Initialize beta
        beta_init = np.zeros((n_features, n_valid), dtype=np.float64)
        
        # Poisson warm start
        if self.poisson_init_iter > 0:
            beta_init = self._poisson_warm_start_batch(Y_valid, beta_init)
        
        # Initial dispersion (MoM)
        alpha = np.full(n_valid, 0.1, dtype=np.float64)
        
        # Run Numba IRLS
        beta_result, se_result, conv_result, iter_result = _irls_batch_numba(
            Y_valid,
            self.design,
            self.offset,
            alpha,
            beta_init,
            self.max_iter,
            self.tol,
            self.min_mu,
            self.ridge_penalty,
        )
        
        # Compute final dispersion using MoM
        mu_final = np.zeros_like(Y_valid)
        for g in range(n_valid):
            eta = self.offset + self.design @ beta_result[:, g]
            eta = np.clip(eta, np.log(self.min_mu), 20.0)
            mu_final[:, g] = np.exp(eta)
        mu_final = np.maximum(mu_final, self.min_mu)
        
        resid = Y_valid - mu_final
        dof = max(self.n_samples - n_features, 1)
        alpha_final = np.sum((resid * resid - Y_valid) / np.maximum(mu_final * mu_final, self.min_mu), axis=0) / dof
        alpha_final = np.clip(alpha_final, 1e-8, 1e6)
        
        # Cox-Reid refinement if requested
        if self.dispersion_method == "cox-reid":
            alpha_final = self._refine_dispersion_cox_reid_batch(Y_valid, mu_final, alpha_final)
        
        # Compute deviance
        dev_valid = self._compute_deviance_batch(Y_valid, mu_final, alpha_final)
        
        # Store results
        coef[valid_indices] = beta_result.T
        se[valid_indices] = se_result.T
        dispersion[valid_indices] = alpha_final
        converged[valid_indices] = conv_result
        n_iter[valid_indices] = iter_result
        deviance[valid_indices] = dev_valid
        
        return NBGLMBatchResult(
            coef=coef, se=se, dispersion=dispersion,
            converged=converged, n_iter=n_iter, deviance=deviance
        )
    
    def _fit_batch_numpy_batched(
        self,
        Y: np.ndarray,
        Y_valid: np.ndarray,
        valid_genes: np.ndarray,
        valid_indices: np.ndarray,
        n_genes: int,
        gene_batch_size: int,
    ) -> NBGLMBatchResult:
        """NumPy-based IRLS with gene batching to reduce memory.
        
        Processes genes in batches to limit work array memory usage.
        Reduced from 7 to 4 work arrays via memory reuse.
        """
        n_valid = Y_valid.shape[1]
        n_features = self.n_features
        X = self.design
        
        # Initialize outputs
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged_arr = np.zeros(n_genes, dtype=bool)
        n_iter_arr = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Initialize beta for all valid genes
        beta_all = np.zeros((n_features, n_valid), dtype=np.float64)
        
        # Poisson warm start
        if self.poisson_init_iter > 0:
            beta_all = self._poisson_warm_start_batch(Y_valid, beta_all)
        
        # Initialize dispersion
        alpha_all = np.full(n_valid, 0.1, dtype=np.float64)
        gene_converged = np.zeros(n_valid, dtype=bool)
        gene_n_iter = np.zeros(n_valid, dtype=np.int32)
        
        # Precompute constants
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        # Process genes in batches
        for batch_start in range(0, n_valid, gene_batch_size):
            batch_end = min(batch_start + gene_batch_size, n_valid)
            batch_size = batch_end - batch_start
            batch_slice = slice(batch_start, batch_end)
            
            Y_batch = Y_valid[:, batch_slice]
            beta_batch = beta_all[:, batch_slice]
            alpha_batch = alpha_all[batch_slice]
            batch_converged = np.zeros(batch_size, dtype=bool)
            
            # Allocate work arrays for this batch only (4 arrays instead of 7)
            eta = np.empty((self.n_samples, batch_size), dtype=np.float64)
            mu = np.empty_like(eta)
            # variance_weights: used for both variance and weights (sequential)
            variance_weights = np.empty_like(eta)
            # z_working: used for z, working_response, and resid (sequential)
            z_working = np.empty_like(eta)
            
            for iteration in range(1, self.max_iter + 1):
                # Compute eta and mu
                np.dot(X, beta_batch, out=eta)
                eta += offset_col
                np.clip(eta, log_min_mu, 20.0, out=eta)
                np.exp(eta, out=mu)
                np.maximum(mu, self.min_mu, out=mu)
                
                # Compute variance: V = mu + alpha * mu^2
                np.multiply(mu, mu, out=variance_weights)
                variance_weights *= alpha_batch[None, :]
                variance_weights += mu
                
                # Compute weights in-place: W = mu^2 / V
                mu_sq = mu * mu  # Temporary for numerator
                np.divide(mu_sq, np.maximum(variance_weights, self.min_mu), out=variance_weights)
                # Now variance_weights contains weights
                
                # Working response: z = eta + (Y - mu) / mu - offset
                np.subtract(Y_batch, mu, out=z_working)  # z_working = Y - mu
                np.divide(z_working, np.maximum(mu, self.min_mu), out=z_working)
                z_working += eta
                z_working -= offset_col
                # Now z_working contains working_response
                
                # Solve weighted least squares
                beta_new = self._weighted_least_squares_batch(variance_weights, z_working)
                
                # Check convergence
                beta_diff = np.max(np.abs(beta_new - beta_batch), axis=0)
                newly_converged = (beta_diff < self.tol) & ~batch_converged
                batch_converged |= newly_converged
                
                # Update iteration count for non-converged genes
                for i in range(batch_size):
                    if not batch_converged[i]:
                        gene_n_iter[batch_start + i] = iteration
                
                beta_batch = beta_new
                
                # Update dispersion (MoM)
                np.subtract(Y_batch, mu, out=z_working)  # resid = Y - mu
                np.multiply(z_working, z_working, out=eta)  # reuse eta as temp
                eta -= Y_batch
                denom = np.maximum(mu * mu, self.min_mu)
                numerator = np.sum(eta / denom, axis=0)
                dof = max(self.n_samples - n_features, 1)
                alpha_new = np.clip(numerator / dof, 1e-8, 1e6)
                alpha_batch = np.where(np.isfinite(alpha_new), alpha_new, alpha_batch)
                
                if np.all(batch_converged):
                    break
            
            # Store batch results
            beta_all[:, batch_slice] = beta_batch
            alpha_all[batch_slice] = alpha_batch
            gene_converged[batch_start:batch_end] = batch_converged
            
            # Clean up batch arrays
            del eta, mu, variance_weights, z_working, mu_sq
        
        # Final mu computation and dispersion refinement
        eta_final = np.dot(X, beta_all) + offset_col
        np.clip(eta_final, log_min_mu, 20.0, out=eta_final)
        mu_final = np.exp(eta_final)
        np.maximum(mu_final, self.min_mu, out=mu_final)
        
        if self.dispersion_method == "cox-reid":
            alpha_all = self._refine_dispersion_cox_reid_batch(Y_valid, mu_final, alpha_all)
        
        # Compute SE and deviance
        variance_final = mu_final + alpha_all[None, :] * mu_final * mu_final
        weights_final = mu_final * mu_final / np.maximum(variance_final, self.min_mu)
        se_valid = self._compute_se_batch(weights_final)
        dev_valid = self._compute_deviance_batch(Y_valid, mu_final, alpha_all)
        
        # Store to output arrays
        coef[valid_indices] = beta_all.T
        se[valid_indices] = se_valid.T
        dispersion[valid_indices] = alpha_all
        converged_arr[valid_indices] = gene_converged
        n_iter_arr[valid_indices] = gene_n_iter
        deviance[valid_indices] = dev_valid
        
        return NBGLMBatchResult(
            coef=coef, se=se, dispersion=dispersion,
            converged=converged_arr, n_iter=n_iter_arr, deviance=deviance
        )
    
    def fit_batch_with_covariate_offset(
        self, 
        counts: ArrayLike, 
        covariate_offset: np.ndarray,
    ) -> NBGLMBatchResult:
        """Fit NB GLM with pre-computed covariate offset.
        
        This method is used in the joint fitting approach where covariate effects
        are estimated globally and then held fixed during per-perturbation fitting.
        The covariate offset is subtracted from the working response during IRLS.
        
        Parameters
        ----------
        counts
            Count matrix of shape ``(n_samples, n_genes)``.
        covariate_offset
            Pre-computed covariate offset of shape ``(n_samples, n_genes)``,
            representing X_cov @ beta_cov for the covariate portion of the design.
            
        Returns
        -------
        NBGLMBatchResult
            Results for all genes with vectorized arrays.
        """
        if sp.issparse(counts):
            Y = np.asarray(counts.toarray(), dtype=np.float64)
        else:
            Y = np.asarray(counts, dtype=np.float64)
        
        if Y.ndim != 2 or Y.shape[0] != self.n_samples:
            raise ValueError(f"counts must have shape ({self.n_samples}, n_genes)")
        
        n_genes = Y.shape[1]
        covariate_offset = np.asarray(covariate_offset, dtype=np.float64)
        if covariate_offset.shape != (self.n_samples, n_genes):
            raise ValueError(
                f"covariate_offset must have shape ({self.n_samples}, {n_genes})"
            )
        
        X = self.design
        n_features = self.n_features
        
        # Initialize outputs
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged = np.zeros(n_genes, dtype=bool)
        n_iter = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Check which genes have sufficient counts
        total_counts = Y.sum(axis=0)
        valid_genes = total_counts >= self.min_total_count
        n_valid = valid_genes.sum()
        
        if n_valid == 0:
            return NBGLMBatchResult(
                coef=coef, se=se, dispersion=dispersion,
                converged=converged, n_iter=n_iter, deviance=deviance
            )
        
        # Work only with valid genes
        Y_valid = Y[:, valid_genes]  # (n_samples, n_valid)
        cov_offset_valid = covariate_offset[:, valid_genes]  # (n_samples, n_valid)
        valid_indices = np.where(valid_genes)[0]
        
        # Initialize beta for all valid genes: (n_features, n_valid)
        beta = np.zeros((n_features, n_valid), dtype=np.float64)
        
        # Poisson warm start with covariate offset
        if self.poisson_init_iter > 0:
            beta = self._poisson_warm_start_with_offset(Y_valid, beta, cov_offset_valid)
        
        # Initialize dispersion estimates (method of moments)
        alpha = np.full(n_valid, 0.1, dtype=np.float64)
        
        # IRLS iterations
        gene_converged = np.zeros(n_valid, dtype=bool)
        gene_n_iter = np.zeros(n_valid, dtype=np.int32)
        
        # Pre-allocate work arrays
        eta = np.empty((self.n_samples, n_valid), dtype=np.float64)
        mu = np.empty_like(eta)
        variance = np.empty_like(eta)
        weights = np.empty_like(eta)
        z = np.empty_like(eta)
        working_response = np.empty_like(eta)
        resid = np.empty_like(eta)
        
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        for iteration in range(1, self.max_iter + 1):
            # Compute eta including covariate offset
            # eta = X @ beta + offset + covariate_offset
            np.dot(X, beta, out=eta)
            eta += offset_col
            eta += cov_offset_valid  # Add pre-computed covariate contribution
            np.clip(eta, log_min_mu, 20.0, out=eta)
            np.exp(eta, out=mu)
            np.maximum(mu, self.min_mu, out=mu)
            
            # Compute variance and weights
            np.multiply(mu, mu, out=variance)
            variance *= alpha[None, :]
            variance += mu
            np.divide(mu * mu, np.maximum(variance, self.min_mu), out=weights)
            
            # Working response (subtract covariate offset from z)
            np.subtract(Y_valid, mu, out=resid)
            np.divide(resid, np.maximum(mu, self.min_mu), out=z)
            z += eta
            # Remove covariate offset and regular offset to get working response for X @ beta
            np.subtract(z, offset_col, out=working_response)
            working_response -= cov_offset_valid
            
            # Solve weighted least squares
            beta_new = self._weighted_least_squares_batch(weights, working_response)
            
            # Check convergence
            beta_diff = np.max(np.abs(beta_new - beta), axis=0)
            newly_converged = (beta_diff < self.tol) & ~gene_converged
            gene_converged |= newly_converged
            gene_n_iter[~gene_converged] = iteration
            
            beta = beta_new
            
            # Update dispersion using method of moments
            np.subtract(Y_valid, mu, out=resid)
            np.multiply(resid, resid, out=variance)
            variance -= Y_valid
            denom = np.maximum(mu * mu, self.min_mu)
            numerator = np.sum(variance / denom, axis=0)
            dof = max(self.n_samples - n_features, 1)
            alpha_new = np.clip(numerator / dof, 1e-8, 1e6)
            alpha = np.where(np.isfinite(alpha_new), alpha_new, alpha)
            
            if np.all(gene_converged):
                break
        
        # Final dispersion refinement with Cox-Reid if requested
        if self.dispersion_method == "cox-reid":
            alpha = self._refine_dispersion_cox_reid_batch(Y_valid, mu, alpha)
        
        # Compute final mu, weights, and standard errors
        np.dot(X, beta, out=eta)
        eta += offset_col
        eta += cov_offset_valid
        np.clip(eta, log_min_mu, 20.0, out=eta)
        np.exp(eta, out=mu)
        np.maximum(mu, self.min_mu, out=mu)
        np.multiply(mu, mu, out=variance)
        variance *= alpha[None, :]
        variance += mu
        np.divide(mu * mu, np.maximum(variance, self.min_mu), out=weights)
        
        # Compute SE
        se_valid = self._compute_se_batch(weights)
        
        # Compute deviance
        dev_valid = self._compute_deviance_batch(Y_valid, mu, alpha)
        
        # Store results
        coef[valid_indices] = beta.T
        se[valid_indices] = se_valid.T
        dispersion[valid_indices] = alpha
        converged[valid_indices] = gene_converged
        n_iter[valid_indices] = gene_n_iter
        deviance[valid_indices] = dev_valid
        
        return NBGLMBatchResult(
            coef=coef, se=se, dispersion=dispersion,
            converged=converged, n_iter=n_iter, deviance=deviance
        )
    
    def fit_batch_with_joint_offsets(
        self, 
        counts: ArrayLike, 
        *,
        intercept_offset: np.ndarray | None = None,
        covariate_offset: np.ndarray | None = None,
        fixed_dispersion: np.ndarray | None = None,
    ) -> NBGLMBatchResult:
        """Fit NB GLM with pre-computed intercept and covariate offsets.
        
        This method is used in the joint fitting approach where the global
        intercept (and optionally covariates and dispersion) are estimated 
        using all cells and then held fixed during per-perturbation fitting.
        
        The design matrix should NOT include an intercept column when using
        intercept_offset, as the global intercept is added as an offset.
        
        Parameters
        ----------
        counts
            Count matrix of shape ``(n_samples, n_genes)``.
        intercept_offset
            Pre-computed global intercept of shape ``(n_genes,)``. If provided,
            the design matrix should not include an intercept column.
        covariate_offset
            Pre-computed covariate offset of shape ``(n_samples, n_genes)``,
            representing X_cov @ beta_cov for the covariate portion.
        fixed_dispersion
            If provided, use these dispersion values instead of estimating.
            Shape ``(n_genes,)``.
            
        Returns
        -------
        NBGLMBatchResult
            Results for all genes with vectorized arrays.
        """
        if sp.issparse(counts):
            Y = np.asarray(counts.toarray(), dtype=np.float64)
        else:
            Y = np.asarray(counts, dtype=np.float64)
        
        if Y.ndim != 2 or Y.shape[0] != self.n_samples:
            raise ValueError(f"counts must have shape ({self.n_samples}, n_genes)")
        
        n_genes = Y.shape[1]
        X = self.design
        n_features = self.n_features
        
        # Validate and prepare offsets
        if intercept_offset is not None:
            intercept_offset = np.asarray(intercept_offset, dtype=np.float64)
            if intercept_offset.shape != (n_genes,):
                raise ValueError(f"intercept_offset must have shape ({n_genes},)")
        
        if covariate_offset is not None:
            covariate_offset = np.asarray(covariate_offset, dtype=np.float64)
            if covariate_offset.shape != (self.n_samples, n_genes):
                raise ValueError(
                    f"covariate_offset must have shape ({self.n_samples}, {n_genes})"
                )
        
        if fixed_dispersion is not None:
            fixed_dispersion = np.asarray(fixed_dispersion, dtype=np.float64)
            if fixed_dispersion.shape != (n_genes,):
                raise ValueError(f"fixed_dispersion must have shape ({n_genes},)")
        
        # Initialize outputs
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged = np.zeros(n_genes, dtype=bool)
        n_iter = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Check which genes have sufficient counts
        total_counts = Y.sum(axis=0)
        valid_genes = total_counts >= self.min_total_count
        n_valid = valid_genes.sum()
        
        if n_valid == 0:
            return NBGLMBatchResult(
                coef=coef, se=se, dispersion=dispersion,
                converged=converged, n_iter=n_iter, deviance=deviance
            )
        
        # Work only with valid genes
        Y_valid = Y[:, valid_genes]
        valid_indices = np.where(valid_genes)[0]
        
        # Prepare valid gene offsets
        intercept_valid = intercept_offset[valid_genes] if intercept_offset is not None else None
        cov_offset_valid = covariate_offset[:, valid_genes] if covariate_offset is not None else None
        
        # Initialize beta
        beta = np.zeros((n_features, n_valid), dtype=np.float64)
        
        # Poisson warm start
        if self.poisson_init_iter > 0:
            beta = self._poisson_warm_start_with_joint_offsets(
                Y_valid, beta, intercept_valid, cov_offset_valid
            )
        
        # Initialize or use fixed dispersion
        if fixed_dispersion is not None:
            alpha = fixed_dispersion[valid_genes].copy()
            use_fixed_dispersion = True
        else:
            alpha = np.full(n_valid, 0.1, dtype=np.float64)
            use_fixed_dispersion = False
        
        # IRLS iterations
        gene_converged = np.zeros(n_valid, dtype=bool)
        gene_n_iter = np.zeros(n_valid, dtype=np.int32)
        
        # Pre-allocate work arrays
        eta = np.empty((self.n_samples, n_valid), dtype=np.float64)
        mu = np.empty_like(eta)
        variance = np.empty_like(eta)
        weights = np.empty_like(eta)
        z = np.empty_like(eta)
        working_response = np.empty_like(eta)
        resid = np.empty_like(eta)
        
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        for iteration in range(1, self.max_iter + 1):
            # Compute eta = X @ beta + offset + intercept_offset + covariate_offset
            np.dot(X, beta, out=eta)
            eta += offset_col
            if intercept_valid is not None:
                eta += intercept_valid[None, :]  # Broadcast (n_genes,) to (n_samples, n_genes)
            if cov_offset_valid is not None:
                eta += cov_offset_valid
            np.clip(eta, log_min_mu, 20.0, out=eta)
            np.exp(eta, out=mu)
            np.maximum(mu, self.min_mu, out=mu)
            
            # Compute variance and weights
            np.multiply(mu, mu, out=variance)
            variance *= alpha[None, :]
            variance += mu
            np.divide(mu * mu, np.maximum(variance, self.min_mu), out=weights)
            
            # Working response
            np.subtract(Y_valid, mu, out=resid)
            np.divide(resid, np.maximum(mu, self.min_mu), out=z)
            z += eta
            # Remove all offsets to get working response for X @ beta
            np.subtract(z, offset_col, out=working_response)
            if intercept_valid is not None:
                working_response -= intercept_valid[None, :]
            if cov_offset_valid is not None:
                working_response -= cov_offset_valid
            
            # Solve weighted least squares
            beta_new = self._weighted_least_squares_batch(weights, working_response)
            
            # Check convergence
            beta_diff = np.max(np.abs(beta_new - beta), axis=0)
            newly_converged = (beta_diff < self.tol) & ~gene_converged
            gene_converged |= newly_converged
            gene_n_iter[~gene_converged] = iteration
            
            beta = beta_new
            
            # Update dispersion if not fixed
            if not use_fixed_dispersion:
                np.subtract(Y_valid, mu, out=resid)
                np.multiply(resid, resid, out=variance)
                variance -= Y_valid
                denom = np.maximum(mu * mu, self.min_mu)
                numerator = np.sum(variance / denom, axis=0)
                dof = max(self.n_samples - n_features, 1)
                alpha_new = np.clip(numerator / dof, 1e-8, 1e6)
                alpha = np.where(np.isfinite(alpha_new), alpha_new, alpha)
            
            if np.all(gene_converged):
                break
        
        # Final dispersion refinement if not using fixed dispersion
        if not use_fixed_dispersion and self.dispersion_method == "cox-reid":
            alpha = self._refine_dispersion_cox_reid_batch(Y_valid, mu, alpha)
        
        # Compute final mu, weights, and standard errors
        np.dot(X, beta, out=eta)
        eta += offset_col
        if intercept_valid is not None:
            eta += intercept_valid[None, :]
        if cov_offset_valid is not None:
            eta += cov_offset_valid
        np.clip(eta, log_min_mu, 20.0, out=eta)
        np.exp(eta, out=mu)
        np.maximum(mu, self.min_mu, out=mu)
        np.multiply(mu, mu, out=variance)
        variance *= alpha[None, :]
        variance += mu
        np.divide(mu * mu, np.maximum(variance, self.min_mu), out=weights)
        
        # Compute SE
        se_valid = self._compute_se_batch(weights)
        
        # Compute deviance
        dev_valid = self._compute_deviance_batch(Y_valid, mu, alpha)
        
        # Store results
        coef[valid_indices] = beta.T
        se[valid_indices] = se_valid.T
        dispersion[valid_indices] = alpha
        converged[valid_indices] = gene_converged
        n_iter[valid_indices] = gene_n_iter
        deviance[valid_indices] = dev_valid
        
        return NBGLMBatchResult(
            coef=coef, se=se, dispersion=dispersion,
            converged=converged, n_iter=n_iter, deviance=deviance
        )
    
    def _poisson_warm_start_with_joint_offsets(
        self, 
        Y: np.ndarray, 
        beta: np.ndarray, 
        intercept_offset: np.ndarray | None,
        covariate_offset: np.ndarray | None,
    ) -> np.ndarray:
        """Poisson warm start with pre-computed intercept and covariate offsets."""
        X = self.design
        n_samples, n_genes = Y.shape
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        eta = np.empty((n_samples, n_genes), dtype=np.float64)
        mu = np.empty_like(eta)
        z = np.empty_like(eta)
        working_response = np.empty_like(eta)
        
        for _ in range(self.poisson_init_iter):
            np.dot(X, beta, out=eta)
            eta += offset_col
            if intercept_offset is not None:
                eta += intercept_offset[None, :]
            if covariate_offset is not None:
                eta += covariate_offset
            np.clip(eta, log_min_mu, 20.0, out=eta)
            np.exp(eta, out=mu)
            np.maximum(mu, self.min_mu, out=mu)
            
            # Poisson working response
            z[:] = eta + (Y - mu) / np.maximum(mu, self.min_mu)
            np.subtract(z, offset_col, out=working_response)
            if intercept_offset is not None:
                working_response -= intercept_offset[None, :]
            if covariate_offset is not None:
                working_response -= covariate_offset
            
            # Solve with Poisson weights (= mu)
            beta = self._weighted_least_squares_batch(mu, working_response)
        
        return beta
    
    def _poisson_warm_start_with_offset(
        self, Y: np.ndarray, beta: np.ndarray, covariate_offset: np.ndarray
    ) -> np.ndarray:
        """Poisson warm start with pre-computed covariate offset."""
        X = self.design
        n_samples, n_genes = Y.shape
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        eta = np.empty((n_samples, n_genes), dtype=np.float64)
        mu = np.empty_like(eta)
        z = np.empty_like(eta)
        working_response = np.empty_like(eta)
        
        for _ in range(self.poisson_init_iter):
            np.dot(X, beta, out=eta)
            eta += offset_col
            eta += covariate_offset  # Include covariate contribution
            np.clip(eta, log_min_mu, 20.0, out=eta)
            np.exp(eta, out=mu)
            np.maximum(mu, self.min_mu, out=mu)
            
            np.subtract(Y, mu, out=z)
            np.divide(z, np.maximum(mu, self.min_mu), out=z)
            z += eta
            np.subtract(z, offset_col, out=working_response)
            working_response -= covariate_offset  # Remove covariate offset
            
            beta_new = self._weighted_least_squares_batch(mu, working_response)
            if np.max(np.abs(beta_new - beta)) < self.tol:
                return beta_new
            beta = beta_new
        return beta
    
    def _poisson_warm_start_batch(
        self, Y: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """Vectorized Poisson warm start for all genes."""
        X = self.design
        n_samples, n_genes = Y.shape
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        # Pre-allocate work arrays
        eta = np.empty((n_samples, n_genes), dtype=np.float64)
        mu = np.empty_like(eta)
        z = np.empty_like(eta)
        working_response = np.empty_like(eta)
        
        for _ in range(self.poisson_init_iter):
            np.dot(X, beta, out=eta)
            eta += offset_col
            np.clip(eta, log_min_mu, 20.0, out=eta)
            np.exp(eta, out=mu)
            np.maximum(mu, self.min_mu, out=mu)
            # Poisson weights = mu
            np.subtract(Y, mu, out=z)
            np.divide(z, np.maximum(mu, self.min_mu), out=z)
            z += eta
            np.subtract(z, offset_col, out=working_response)
            beta_new = self._weighted_least_squares_batch(mu, working_response)
            if np.max(np.abs(beta_new - beta)) < self.tol:
                return beta_new
            beta = beta_new
        return beta
    
    def _weighted_least_squares_batch(
        self, weights: np.ndarray, y_working: np.ndarray
    ) -> np.ndarray:
        """Solve WLS for all genes simultaneously using vectorized operations.
        
        Parameters
        ----------
        weights
            Weight matrix of shape (n_samples, n_genes).
        y_working
            Working response matrix of shape (n_samples, n_genes).
            
        Returns
        -------
        np.ndarray
            Coefficient matrix of shape (n_features, n_genes).
        """
        X = self.design  # (n_samples, n_features)
        n_samples, n_genes = weights.shape
        n_features = self.n_features
        
        # Clip weights for numerical stability
        W = np.clip(weights, self.min_mu, None)  # (n_samples, n_genes)
        
        # Efficient X^T W X computation using blocked approach
        # X^T W X [g,i,j] = sum_k X[k,i] * W[k,g] * X[k,j]
        # 
        # For 2x2 design matrix, we can compute the 4 elements directly:
        # (0,0): sum_k X[k,0]^2 * W[k,g] = sum_k W[k,g] (since X[:,0] = 1)
        # (0,1) = (1,0): sum_k X[k,0] * X[k,1] * W[k,g] = sum_k X[k,1] * W[k,g]
        # (1,1): sum_k X[k,1]^2 * W[k,g]
        
        if n_features == 2:
            # Fast path for common 2-feature design
            X1 = X[:, 1]  # (n_samples,) - the perturbation indicator
            
            # XtWX elements (vectorized over genes)
            xtwx_00 = np.sum(W, axis=0)  # (n_genes,)
            xtwx_01 = X1 @ W  # (n_genes,)
            xtwx_11 = (X1[:, None] ** 2 * W).sum(axis=0)  # (n_genes,)
            
            # X^T W z elements
            Wz = W * y_working  # (n_samples, n_genes)
            xtwz_0 = np.sum(Wz, axis=0)  # (n_genes,)
            xtwz_1 = X1 @ Wz  # (n_genes,)
            
            # Add ridge penalty
            if self.ridge_penalty:
                xtwx_00 = xtwx_00 + self.ridge_penalty
                xtwx_11 = xtwx_11 + self.ridge_penalty
            
            # Solve 2x2 systems analytically using Cramer's rule
            det = xtwx_00 * xtwx_11 - xtwx_01 ** 2
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)  # Avoid division by zero
            
            beta0 = (xtwx_11 * xtwz_0 - xtwx_01 * xtwz_1) / det
            beta1 = (xtwx_00 * xtwz_1 - xtwx_01 * xtwz_0) / det
            
            beta = np.vstack([beta0, beta1])  # (n_features, n_genes)
            return beta
        else:
            # General case using einsum
            xtwx = np.einsum('ki,kg,kj->gij', X, W, X, optimize=True)
            
            # Add ridge penalty to diagonal
            if self.ridge_penalty:
                ridge = self.ridge_penalty * np.eye(n_features, dtype=np.float64)
                xtwx = xtwx + ridge[None, :, :]
            
            # Compute X^T W z for all genes: (n_genes, n_features)
            Wz = W * y_working  # (n_samples, n_genes)
            xtwz = np.einsum('ki,kg->gi', X, Wz, optimize=True)
            
            # Solve all systems at once using batched solve
            # Need to add dimension for broadcasting: (n_genes, n_features, 1)
            try:
                beta = np.linalg.solve(xtwx, xtwz[:, :, None])[:, :, 0]  # (n_genes, n_features)
            except np.linalg.LinAlgError:
                # Fallback to per-gene solve for singular matrices
                beta = np.zeros((n_genes, n_features), dtype=np.float64)
                for g in range(n_genes):
                    try:
                        beta[g] = np.linalg.solve(xtwx[g], xtwz[g])
                    except np.linalg.LinAlgError:
                        beta[g] = np.linalg.lstsq(xtwx[g], xtwz[g], rcond=None)[0]
            
            return beta.T  # (n_features, n_genes)
    
    def _compute_se_batch(self, weights: np.ndarray) -> np.ndarray:
        """Compute standard errors for all genes using vectorized operations."""
        X = self.design  # (n_samples, n_features)
        n_samples, n_genes = weights.shape
        n_features = self.n_features
        
        # Clip weights for numerical stability
        W = np.clip(weights, self.min_mu, None)  # (n_samples, n_genes)
        
        if n_features == 2:
            # Fast path for 2-feature design: use analytical inverse of 2x2 matrix
            X1 = X[:, 1]  # (n_samples,) - the perturbation indicator
            
            # XtWX elements (vectorized over genes)
            xtwx_00 = np.sum(W, axis=0)  # (n_genes,)
            xtwx_01 = X1 @ W  # (n_genes,)
            xtwx_11 = (X1[:, None] ** 2 * W).sum(axis=0)  # (n_genes,)
            
            # Add ridge penalty
            if self.ridge_penalty:
                xtwx_00 = xtwx_00 + self.ridge_penalty
                xtwx_11 = xtwx_11 + self.ridge_penalty
            
            # 2x2 matrix inverse diagonal elements:
            # For M = [[a, b], [b, c]], M^-1 = (1/det) * [[c, -b], [-b, a]]
            # So diag(M^-1) = [c/det, a/det]
            det = xtwx_00 * xtwx_11 - xtwx_01 ** 2
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)  # Avoid division by zero
            
            inv_diag_0 = xtwx_11 / det  # Variance of beta_0 (intercept)
            inv_diag_1 = xtwx_00 / det  # Variance of beta_1 (perturbation effect)
            
            se = np.vstack([
                np.sqrt(np.maximum(inv_diag_0, 1e-12)),
                np.sqrt(np.maximum(inv_diag_1, 1e-12))
            ])  # (n_features, n_genes)
            return se
        else:
            # General case
            # Compute X^T W X for all genes: (n_genes, n_features, n_features)
            xtwx = np.einsum('ki,kg,kj->gij', X, W, X, optimize=True)
            
            # Add ridge penalty to diagonal
            if self.ridge_penalty:
                ridge = self.ridge_penalty * np.eye(n_features, dtype=np.float64)
                xtwx = xtwx + ridge[None, :, :]
            
            # Invert all matrices at once and extract diagonal
            se = np.full((n_features, n_genes), np.inf, dtype=np.float64)
            try:
                inv_xtwx = np.linalg.inv(xtwx)  # (n_genes, n_features, n_features)
                # Extract diagonal of each inverse matrix
                diag_inv = np.diagonal(inv_xtwx, axis1=1, axis2=2)  # (n_genes, n_features)
                se = np.sqrt(np.maximum(diag_inv, 1e-12)).T  # (n_features, n_genes)
            except np.linalg.LinAlgError:
                # Fallback to per-gene inversion for singular matrices
                for g in range(n_genes):
                    try:
                        inv_xtwx_g = np.linalg.inv(xtwx[g])
                        se[:, g] = np.sqrt(np.maximum(np.diag(inv_xtwx_g), 1e-12))
                    except np.linalg.LinAlgError:
                        pass
            
            return se
    
    def _compute_deviance_batch(
        self, Y: np.ndarray, mu: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """Compute deviance for all genes using vectorized operations."""
        # Y, mu: (n_samples, n_genes), alpha: (n_genes,)
        mu_safe = np.maximum(mu, 1e-12)
        Y_safe = np.maximum(Y, 1e-12)
        
        # Compute r = 1/alpha for each gene, broadcast to (1, n_genes)
        r = 1.0 / np.maximum(alpha, 1e-10)  # (n_genes,)
        r = r[None, :]  # (1, n_genes) for broadcasting
        
        with np.errstate(divide="ignore", invalid="ignore"):
            # NB deviance: 2 * sum(y * log(y/mu) - (y + r) * log((y + r) / (mu + r)))
            # Handle y=0 case: when y=0, y*log(y/mu) = 0
            term1 = np.where(Y > 0, Y * np.log(Y_safe / mu_safe), 0.0)
            term2 = (Y + r) * np.log((Y + r) / (mu_safe + r))
            
            # Sum over samples (axis 0) for each gene
            deviance = 2.0 * np.nansum(term1 - term2, axis=0)  # (n_genes,)
        
        return deviance
    
    def _refine_dispersion_cox_reid_batch(
        self, Y: np.ndarray, mu: np.ndarray, alpha_init: np.ndarray
    ) -> np.ndarray:
        """Refine dispersion estimates using Cox-Reid for all genes.
        
        Uses vectorized grid search with numba acceleration for speed.
        Precomputes gammaln(Y+1) to avoid redundant computation.
        """
        n_samples, n_genes = Y.shape
        n_features = self.n_features
        
        # Vectorized grid search with smaller grid (10 points)
        log_grid = np.linspace(-3, 2, 10)
        alpha_grid = 10.0 ** log_grid
        n_alpha = len(alpha_grid)
        
        # Pre-compute design matrix quantities
        X = self.design
        
        # Precompute gammaln(Y + 1) - this is expensive and Y doesn't change
        gammaln_Y_plus_1 = gammaln_nb(Y + 1)
        
        # Compute NB log-likelihood for all alpha values using parallelized numba kernel
        ll_grid = _nb_loglik_grid_numba(Y, mu, alpha_grid, gammaln_Y_plus_1)
        
        # Cox-Reid adjustment: -0.5 * log(det(X^T W X)) for each alpha
        # Precompute X1 quantities for 2-feature case
        if n_features == 2:
            X1 = X[:, 1]
            X1_sq = X1 ** 2
        
        for a_idx, a in enumerate(alpha_grid):
            variance = mu + a * (mu ** 2)
            W = (mu ** 2) / np.maximum(variance, self.min_mu)
            
            if n_features == 2:
                # Fast path for 2-feature design: analytical determinant
                xtwx_00 = np.sum(W, axis=0)
                xtwx_01 = X1 @ W
                xtwx_11 = np.sum(X1_sq[:, None] * W, axis=0)
                det = xtwx_00 * xtwx_11 - xtwx_01 ** 2
                log_det = np.log(np.maximum(det, 1e-12))
            else:
                # General case using einsum
                XtWX = np.einsum('ki,kg,kj->gij', X, W, X, optimize=True)
                try:
                    sign, log_det = np.linalg.slogdet(XtWX)
                    log_det = np.where(sign > 0, log_det, 0.0)
                except np.linalg.LinAlgError:
                    log_det = np.zeros(n_genes)
            
            ll_grid[a_idx] -= 0.5 * log_det
        
        # Find best alpha for each gene
        nll_grid = -ll_grid  # (n_alpha, n_genes)
        best_idx = np.argmin(nll_grid, axis=0)  # (n_genes,)
        best_alpha = alpha_grid[best_idx]
        
        # Clip to reasonable range
        alpha = np.clip(best_alpha, 1e-8, 1e3)
        
        return alpha

    def fit_batch_with_control_cache(
        self,
        perturbation_matrix: np.ndarray | sp.csr_matrix,
        perturbation_offset: np.ndarray,
        control_cache: "ControlStatisticsCache",
        *,
        perturbation_indicator: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> NBGLMBatchResult:
        """Fit NB GLM using precomputed control cell statistics.
        
        This method provides significant speedup by reusing control cell
        contributions (XᵀWX, XᵀWz) from a precomputed cache instead of
        redundantly computing them for each perturbation comparison.
        
        The design matrix is [1, perturbation_indicator] where:
        - Control cells have indicator = 0
        - Perturbation cells have indicator = 1
        
        The control contribution to XᵀWX and XᵀWz is taken from the cache,
        and only perturbation cell contributions are computed fresh.
        
        Parameters
        ----------
        perturbation_matrix
            Expression matrix for perturbation cells only, shape (n_pert, n_genes).
        perturbation_offset
            Log size factors for perturbation cells, shape (n_pert,).
        control_cache
            Precomputed control cell statistics from `precompute_control_statistics`.
        perturbation_indicator
            Binary indicator for perturbation cells in the combined design.
            Should be shape (n_control + n_pert,) with 0 for control, 1 for perturbation.
        valid_mask
            Optional boolean mask for genes to fit, shape (n_genes,).
            
        Returns
        -------
        NBGLMBatchResult
            Fitted coefficients and statistics.
        """
        # Densify perturbation matrix if needed
        if sp.issparse(perturbation_matrix):
            Y_pert = np.asarray(perturbation_matrix.toarray(), dtype=np.float64)
        else:
            Y_pert = np.asarray(perturbation_matrix, dtype=np.float64)
        
        n_pert, n_genes = Y_pert.shape
        n_control = control_cache.control_n
        n_total = n_control + n_pert
        
        # Get control data from cache (already dense, no need for .toarray())
        Y_control = control_cache.control_matrix  # Already np.ndarray
        
        # Initialize outputs
        n_features = 2  # intercept + perturbation
        coef = np.zeros((n_genes, n_features), dtype=np.float64)
        se = np.full((n_genes, n_features), np.inf, dtype=np.float64)
        dispersion = np.full(n_genes, np.nan, dtype=np.float64)
        converged = np.zeros(n_genes, dtype=bool)
        n_iter_arr = np.zeros(n_genes, dtype=np.int32)
        deviance = np.full(n_genes, np.nan, dtype=np.float64)
        
        # Determine valid genes
        if valid_mask is None:
            total_counts = Y_control.sum(axis=0) + Y_pert.sum(axis=0)
            valid_mask = total_counts >= self.min_total_count
        
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        
        if n_valid == 0:
            return NBGLMBatchResult(
                coef=coef, se=se, dispersion=dispersion,
                converged=converged, n_iter=n_iter_arr, deviance=deviance
            )
        
        # Work with valid genes only
        Y_control_valid = Y_control[:, valid_mask]
        Y_pert_valid = Y_pert[:, valid_mask]
        
        # Initialize beta from cache: [β₀_cached, 0]
        beta = np.zeros((n_features, n_valid), dtype=np.float64)
        beta[0, :] = control_cache.beta_intercept[valid_mask]
        
        # Use cached control dispersion as starting point
        alpha = control_cache.control_dispersion[valid_mask].copy()
        
        # Precompute offsets
        offset_control = control_cache.control_offset[:, None]
        offset_pert = perturbation_offset[:, None]
        log_min_mu = np.log(self.min_mu)
        
        # Work arrays
        mu_control = np.empty((n_control, n_valid), dtype=np.float64)
        mu_pert = np.empty((n_pert, n_valid), dtype=np.float64)
        W_control = np.empty_like(mu_control)
        W_pert = np.empty_like(mu_pert)
        
        # Convergence tracking
        gene_converged = np.zeros(n_valid, dtype=bool)
        gene_n_iter = np.zeros(n_valid, dtype=np.int32)
        
        for iteration in range(1, self.max_iter + 1):
            beta_intercept = beta[0, :]  # (n_valid,)
            beta_pert = beta[1, :]  # (n_valid,)
            
            # Control cells: eta = β₀ + offset (perturbation indicator = 0)
            eta_control = beta_intercept[None, :] + offset_control
            np.clip(eta_control, log_min_mu, 20.0, out=eta_control)
            np.exp(eta_control, out=mu_control)
            np.maximum(mu_control, self.min_mu, out=mu_control)
            
            # Perturbation cells: eta = β₀ + β₁ + offset (perturbation indicator = 1)
            eta_pert = beta_intercept[None, :] + beta_pert[None, :] + offset_pert
            np.clip(eta_pert, log_min_mu, 20.0, out=eta_pert)
            np.exp(eta_pert, out=mu_pert)
            np.maximum(mu_pert, self.min_mu, out=mu_pert)
            
            # Weights: W = μ² / (μ + α * μ²)
            var_control = mu_control + alpha[None, :] * mu_control * mu_control
            np.divide(mu_control * mu_control, np.maximum(var_control, self.min_mu), out=W_control)
            
            var_pert = mu_pert + alpha[None, :] * mu_pert * mu_pert
            np.divide(mu_pert * mu_pert, np.maximum(var_pert, self.min_mu), out=W_pert)
            
            # Working responses
            z_control = eta_control + (Y_control_valid - mu_control) / np.maximum(mu_control, self.min_mu)
            z_pert = eta_pert + (Y_pert_valid - mu_pert) / np.maximum(mu_pert, self.min_mu)
            
            # Remove offsets for working response
            z_control_centered = z_control - offset_control
            z_pert_centered = z_pert - offset_pert
            
            # Compute XᵀWX and XᵀWz
            # For design [1, p] where p is perturbation indicator:
            # XᵀWX = [[sum_all(W), sum_pert(W)],
            #         [sum_pert(W), sum_pert(W)]]
            # XᵀWz = [sum_all(W*z), sum_pert(W*z)]
            
            W_control_sum = np.sum(W_control, axis=0)  # (n_valid,)
            W_pert_sum = np.sum(W_pert, axis=0)  # (n_valid,)
            
            Wz_control_sum = np.sum(W_control * z_control_centered, axis=0)  # (n_valid,)
            Wz_pert_sum = np.sum(W_pert * z_pert_centered, axis=0)  # (n_valid,)
            
            # XᵀWX elements
            xtwx_00 = W_control_sum + W_pert_sum  # sum over all cells
            xtwx_01 = W_pert_sum  # only perturbation cells contribute
            xtwx_11 = W_pert_sum  # perturbation indicator is 1 for pert cells
            
            # XᵀWz elements
            xtwz_0 = Wz_control_sum + Wz_pert_sum  # all cells
            xtwz_1 = Wz_pert_sum  # only perturbation cells
            
            # Add ridge penalty
            ridge = self.ridge_penalty
            xtwx_00 = xtwx_00 + ridge
            xtwx_11 = xtwx_11 + ridge
            
            # Solve 2x2 system using Cramer's rule
            det = xtwx_00 * xtwx_11 - xtwx_01 ** 2
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            
            beta_new_0 = (xtwx_11 * xtwz_0 - xtwx_01 * xtwz_1) / det
            beta_new_1 = (xtwx_00 * xtwz_1 - xtwx_01 * xtwz_0) / det
            
            beta_new = np.vstack([beta_new_0, beta_new_1])
            
            # Check convergence
            beta_diff = np.max(np.abs(beta_new - beta), axis=0)
            newly_converged = (beta_diff < self.tol) & ~gene_converged
            gene_converged |= newly_converged
            gene_n_iter[~gene_converged] = iteration
            
            beta = beta_new
            
            # Update dispersion (method of moments)
            resid_control = Y_control_valid - mu_control
            resid_pert = Y_pert_valid - mu_pert
            
            numerator = (
                np.sum((resid_control ** 2 - Y_control_valid) / np.maximum(mu_control ** 2, self.min_mu), axis=0)
                + np.sum((resid_pert ** 2 - Y_pert_valid) / np.maximum(mu_pert ** 2, self.min_mu), axis=0)
            )
            dof = max(n_total - n_features, 1)
            alpha_new = np.clip(numerator / dof, 1e-8, 1e6)
            alpha = np.where(np.isfinite(alpha_new), alpha_new, alpha)
            
            if np.all(gene_converged):
                break
        
        # Compute final standard errors using sandwich estimator (PyDESeq2 style)
        # SE = sqrt(c' @ H @ M @ H @ c) where:
        #   M = XᵀWX (unregularized Fisher information)
        #   Mr = M + ridge*I (regularized)
        #   H = inv(Mr)
        #   c = [0, 1] for perturbation effect
        
        # Recompute XᵀWX for final weights
        W_control_sum = np.sum(W_control, axis=0)
        W_pert_sum = np.sum(W_pert, axis=0)
        
        # Unregularized M
        M00 = W_control_sum + W_pert_sum
        M01 = W_pert_sum
        M11 = W_pert_sum
        
        # Regularized Mr = M + ridge*I
        ridge = self.ridge_penalty
        Mr00 = M00 + ridge
        Mr01 = M01
        Mr11 = M11 + ridge
        
        # H = inv(Mr) for 2x2: inv = (1/det) * [[d, -b], [-c, a]]
        det_r = Mr00 * Mr11 - Mr01 * Mr01
        det_r = np.where(np.abs(det_r) < 1e-12, 1e-12, det_r)
        
        H00 = Mr11 / det_r
        H01 = -Mr01 / det_r
        H11 = Mr00 / det_r
        
        # For contrast c = [0, 1]: Hc = [H[0,1], H[1,1]] = [H01, H11]
        Hc0 = H01
        Hc1 = H11
        
        # Sandwich variance: Hc.T @ M @ Hc
        # = Hc0² * M00 + 2 * Hc0 * Hc1 * M01 + Hc1² * M11
        var_pert = Hc0**2 * M00 + 2 * Hc0 * Hc1 * M01 + Hc1**2 * M11
        se_pert = np.sqrt(np.maximum(var_pert, 1e-12))
        
        # For intercept, contrast c = [1, 0]: Hc = [H00, H01]
        var_intercept = H00**2 * M00 + 2 * H00 * H01 * M01 + H01**2 * M11
        se_intercept = np.sqrt(np.maximum(var_intercept, 1e-12))
        
        se_valid = np.vstack([se_intercept, se_pert])  # (n_features, n_valid)
        
        # Store results
        coef[valid_indices] = beta.T
        se[valid_indices] = se_valid.T
        dispersion[valid_indices] = alpha
        converged[valid_indices] = gene_converged
        n_iter_arr[valid_indices] = gene_n_iter
        
        return NBGLMBatchResult(
            coef=coef, se=se, dispersion=dispersion,
            converged=converged, n_iter=n_iter_arr, deviance=deviance
        )
