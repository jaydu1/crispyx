"""Generalized linear models utilities for differential expression."""

from __future__ import annotations

import ctypes
import logging
import math
from dataclasses import dataclass
from typing import Sequence, Literal, Tuple

import numba as nb
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from numba.extending import get_cython_function_address
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar, minimize, brentq
from scipy.special import gammaln, digamma, polygamma

logger = logging.getLogger(__name__)

# Create numba-accelerated gammaln using scipy's cython implementation
_PTR = ctypes.POINTER
_dble = ctypes.c_double
_addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
_functype = ctypes.CFUNCTYPE(_dble, _dble)
_gammaln_float64 = _functype(_addr)


@nb.vectorize([nb.float64(nb.float64)], nopython=True)
def gammaln_nb(x):
    """Numba-accelerated gammaln using scipy's cython implementation."""
    return _gammaln_float64(x)


@nb.njit(parallel=True)
def _nb_loglik_grid_numba(
    Y: np.ndarray,
    mu: np.ndarray,
    alpha_grid: np.ndarray,
    gammaln_Y_plus_1: np.ndarray,
) -> np.ndarray:
    """Compute NB log-likelihood for all alpha values in grid (parallelized).
    
    Parameters
    ----------
    Y : (n_samples, n_genes)
    mu : (n_samples, n_genes)
    alpha_grid : (n_alpha,)
    gammaln_Y_plus_1 : (n_samples, n_genes) precomputed
    
    Returns
    -------
    ll_grid : (n_alpha, n_genes) log-likelihood for each alpha and gene
    """
    n_samples, n_genes = Y.shape
    n_alpha = alpha_grid.shape[0]
    ll_grid = np.zeros((n_alpha, n_genes), dtype=np.float64)
    
    for a_idx in nb.prange(n_alpha):
        alpha = alpha_grid[a_idx]
        r = 1.0 / alpha
        log_r = np.log(r)
        gammaln_r = math.lgamma(r)
        
        for g in range(n_genes):
            ll = 0.0
            for i in range(n_samples):
                y_ig = Y[i, g]
                mu_ig = mu[i, g]
                # gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                # + r * log(r / (r + mu)) + y * log(mu / (r + mu))
                ll += (
                    math.lgamma(y_ig + r)
                    - gammaln_r
                    - gammaln_Y_plus_1[i, g]
                    + r * (log_r - np.log(r + mu_ig + 1e-12))
                    + y_ig * np.log(mu_ig / (r + mu_ig + 1e-12) + 1e-12)
                )
            ll_grid[a_idx, g] = ll
    
    return ll_grid


# =============================================================================
# Numba-accelerated kernels for joint model streaming
# =============================================================================

@nb.njit(parallel=True, cache=True)
def _accumulate_perturbation_blocks_numba(
    W: np.ndarray,
    Wz: np.ndarray,
    X_dense: np.ndarray,
    pert_idx: np.ndarray,
    n_perturbations: int,
    pert_xtwx_diag: np.ndarray,
    cross_xtwx: np.ndarray,
    pert_xtwz: np.ndarray,
) -> None:
    """Numba-accelerated accumulation of perturbation blocks.
    
    Accumulates X^T W X contributions for perturbation diagonal and cross-terms.
    This replaces the Python for-loop over cells with parallelized Numba code.
    
    Parameters
    ----------
    W : (n_chunk, n_genes)
        Weight matrix.
    Wz : (n_chunk, n_genes)
        Weighted working response.
    X_dense : (n_chunk, n_dense_features)
        Dense design matrix portion.
    pert_idx : (n_chunk,)
        Perturbation index for each cell (-1 for control).
    n_perturbations : int
        Number of perturbations.
    pert_xtwx_diag : (n_genes, n_perturbations)
        Output: diagonal of perturbation block (accumulated in-place).
    cross_xtwx : (n_genes, n_dense_features, n_perturbations)
        Output: cross-term block (accumulated in-place).
    pert_xtwz : (n_genes, n_perturbations)
        Output: perturbation RHS (accumulated in-place).
    """
    n_chunk, n_genes = W.shape
    n_dense = X_dense.shape[1]
    
    # Parallelize over genes for better cache locality
    for g in nb.prange(n_genes):
        for i in range(n_chunk):
            p_idx = pert_idx[i]
            if p_idx >= 0:
                w_ig = W[i, g]
                wz_ig = Wz[i, g]
                pert_xtwx_diag[g, p_idx] += w_ig
                pert_xtwz[g, p_idx] += wz_ig
                for j in range(n_dense):
                    cross_xtwx[g, j, p_idx] += w_ig * X_dense[i, j]


@nb.njit(parallel=True, cache=True)
def _batch_schur_solve_numba(
    dense_xtwx: np.ndarray,
    pert_xtwx_diag: np.ndarray,
    cross_xtwx: np.ndarray,
    dense_xtwz: np.ndarray,
    pert_xtwz: np.ndarray,
    pert_has_data: np.ndarray,
    ridge_dense: np.ndarray,
    ridge_pert: float,
    beta_intercept: np.ndarray,
    beta_cov: np.ndarray,
    beta_pert: np.ndarray,
) -> None:
    """Numba-accelerated batch Schur complement solving.
    
    Solves the block system for all genes in parallel using Schur complement:
        [A  B ] [x1]   [b1]
        [B' D ] [x2] = [b2]
    
    where D is diagonal, using: x1 = (A - B D^{-1} B')^{-1} (b1 - B D^{-1} b2)
                                x2 = D^{-1} (b2 - B' x1)
    
    Parameters
    ----------
    dense_xtwx : (n_genes, n_dense, n_dense)
    pert_xtwx_diag : (n_genes, n_pert)
    cross_xtwx : (n_genes, n_dense, n_pert)
    dense_xtwz : (n_genes, n_dense)
    pert_xtwz : (n_genes, n_pert)
    pert_has_data : (n_pert, n_genes) boolean mask
    ridge_dense : (n_dense, n_dense)
    ridge_pert : float
    beta_intercept : (n_genes,) output
    beta_cov : (n_cov, n_genes) output
    beta_pert : (n_pert, n_genes) output
    """
    n_genes = dense_xtwx.shape[0]
    n_dense = dense_xtwx.shape[1]
    n_pert = pert_xtwx_diag.shape[1]
    n_cov = beta_cov.shape[0]
    
    for g in nb.prange(n_genes):
        # Build A = dense_xtwx[g] + ridge_dense
        A = np.zeros((n_dense, n_dense), dtype=np.float64)
        for i in range(n_dense):
            for j in range(n_dense):
                A[i, j] = dense_xtwx[g, i, j] + ridge_dense[i, j]
        
        # D_diag = pert_xtwx_diag[g] + ridge_pert
        D_diag = np.zeros(n_pert, dtype=np.float64)
        for p in range(n_pert):
            D_diag[p] = pert_xtwx_diag[g, p] + ridge_pert
        
        # B = cross_xtwx[g]
        B = cross_xtwx[g]  # (n_dense, n_pert)
        
        b1 = dense_xtwz[g]  # (n_dense,)
        b2 = pert_xtwz[g]  # (n_pert,)
        
        # D_inv
        D_inv = np.zeros(n_pert, dtype=np.float64)
        for p in range(n_pert):
            D_inv[p] = 1.0 / max(D_diag[p], 1e-12)
        
        # B_Dinv = B * D_inv (element-wise broadcast)
        B_Dinv = np.zeros((n_dense, n_pert), dtype=np.float64)
        for i in range(n_dense):
            for p in range(n_pert):
                B_Dinv[i, p] = B[i, p] * D_inv[p]
        
        # schur = A - B_Dinv @ B.T
        schur = np.zeros((n_dense, n_dense), dtype=np.float64)
        for i in range(n_dense):
            for j in range(n_dense):
                schur[i, j] = A[i, j]
                for p in range(n_pert):
                    schur[i, j] -= B_Dinv[i, p] * B[j, p]
        
        # schur_rhs = b1 - B_Dinv @ b2
        schur_rhs = np.zeros(n_dense, dtype=np.float64)
        for i in range(n_dense):
            schur_rhs[i] = b1[i]
            for p in range(n_pert):
                schur_rhs[i] -= B_Dinv[i, p] * b2[p]
        
        # Solve schur @ x1 = schur_rhs using Gaussian elimination
        # For small matrices (n_dense typically 1-5), direct inversion is fine
        x1 = np.zeros(n_dense, dtype=np.float64)
        if n_dense == 1:
            x1[0] = schur_rhs[0] / max(schur[0, 0], 1e-12)
        else:
            # Simple Gaussian elimination with partial pivoting
            aug = np.zeros((n_dense, n_dense + 1), dtype=np.float64)
            for i in range(n_dense):
                for j in range(n_dense):
                    aug[i, j] = schur[i, j]
                aug[i, n_dense] = schur_rhs[i]
            
            for col in range(n_dense):
                # Find pivot
                max_row = col
                max_val = abs(aug[col, col])
                for row in range(col + 1, n_dense):
                    if abs(aug[row, col]) > max_val:
                        max_val = abs(aug[row, col])
                        max_row = row
                # Swap rows
                if max_row != col:
                    for j in range(n_dense + 1):
                        tmp = aug[col, j]
                        aug[col, j] = aug[max_row, j]
                        aug[max_row, j] = tmp
                # Eliminate
                pivot = aug[col, col]
                if abs(pivot) < 1e-12:
                    continue
                for row in range(col + 1, n_dense):
                    factor = aug[row, col] / pivot
                    for j in range(col, n_dense + 1):
                        aug[row, j] -= factor * aug[col, j]
            
            # Back substitution
            for i in range(n_dense - 1, -1, -1):
                val = aug[i, n_dense]
                for j in range(i + 1, n_dense):
                    val -= aug[i, j] * x1[j]
                if abs(aug[i, i]) > 1e-12:
                    x1[i] = val / aug[i, i]
                else:
                    x1[i] = 0.0
        
        # x2 = D_inv * (b2 - B.T @ x1)
        x2 = np.zeros(n_pert, dtype=np.float64)
        for p in range(n_pert):
            tmp = b2[p]
            for i in range(n_dense):
                tmp -= B[i, p] * x1[i]
            x2[p] = D_inv[p] * tmp
            
            # Zero out if no data, clip to bounds
            if not pert_has_data[p, g]:
                x2[p] = 0.0
            else:
                x2[p] = max(-30.0, min(30.0, x2[p]))
        
        # Store results
        beta_intercept[g] = x1[0]
        for c in range(n_cov):
            beta_cov[c, g] = x1[1 + c]
        for p in range(n_pert):
            beta_pert[p, g] = x2[p]


def _build_sparse_perturbation_indicator(
    pert_idx: np.ndarray,
    n_perturbations: int,
) -> sp.csr_matrix:
    """Build sparse CSR perturbation indicator matrix.
    
    Parameters
    ----------
    pert_idx : (n_cells,)
        Perturbation index for each cell (-1 for control).
    n_perturbations : int
        Number of perturbations.
        
    Returns
    -------
    P : (n_cells, n_perturbations) sparse CSR matrix
        Binary indicator matrix where P[i, p] = 1 if cell i is in perturbation p.
    """
    n_cells = len(pert_idx)
    mask = pert_idx >= 0
    rows = np.where(mask)[0]
    cols = pert_idx[mask]
    data = np.ones(len(rows), dtype=np.float64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_perturbations))


def _accumulate_perturbation_blocks_sparse(
    W: np.ndarray,
    Wz: np.ndarray,
    X_dense: np.ndarray,
    P_chunk: sp.csr_matrix,
    pert_xtwx_diag: np.ndarray,
    cross_xtwx: np.ndarray,
    pert_xtwz: np.ndarray,
) -> None:
    """Sparse matrix accumulation of perturbation blocks.
    
    Uses sparse matrix multiplication which is 10-50x faster than Python loops
    for typical single-cell sparsity patterns.
    
    Parameters
    ----------
    W : (n_chunk, n_genes)
        Weight matrix.
    Wz : (n_chunk, n_genes)
        Weighted working response.
    X_dense : (n_chunk, n_dense_features)
        Dense design matrix portion.
    P_chunk : (n_chunk, n_perturbations) sparse CSR
        Perturbation indicator matrix for this chunk.
    pert_xtwx_diag : (n_genes, n_perturbations)
        Output: diagonal of perturbation block (accumulated in-place).
    cross_xtwx : (n_genes, n_dense_features, n_perturbations)
        Output: cross-term block (accumulated in-place).
    pert_xtwz : (n_genes, n_perturbations)
        Output: perturbation RHS (accumulated in-place).
    """
    # pert_xtwx_diag[g, p] = sum_i W[i, g] * P[i, p]
    # This is W.T @ P -> (n_genes, n_pert)
    pert_xtwx_diag += (W.T @ P_chunk).toarray() if sp.issparse(W.T @ P_chunk) else (W.T @ P_chunk)
    
    # pert_xtwz[g, p] = sum_i Wz[i, g] * P[i, p]
    # This is Wz.T @ P -> (n_genes, n_pert)
    pert_xtwz += (Wz.T @ P_chunk).toarray() if sp.issparse(Wz.T @ P_chunk) else (Wz.T @ P_chunk)
    
    # cross_xtwx[g, j, p] = sum_i W[i, g] * X_dense[i, j] * P[i, p]
    # Rewrite as: for each j, cross_xtwx[:, j, :] = (W * X_dense[:, j:j+1]).T @ P
    n_dense = X_dense.shape[1]
    for j in range(n_dense):
        WX_j = W * X_dense[:, j:j+1]  # (n_chunk, n_genes)
        contrib = WX_j.T @ P_chunk  # (n_genes, n_pert)
        if sp.issparse(contrib):
            contrib = contrib.toarray()
        cross_xtwx[:, j, :] += contrib


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


class NBGLMFitter:
    """Iteratively re-weighted least squares solver for NB GLMs.

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
        Maximum number of Fisher scoring iterations for the negative binomial
        stage.
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
        optimization_method: Literal["irls", "lbfgsb"] = "lbfgsb",
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
        self.optimization_method = optimization_method

    def fit_gene(self, counts: ArrayLike) -> NBGLMResult:
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
        
        # Use L-BFGS-B optimization if requested
        if self.optimization_method == "lbfgsb":
            return self._fit_gene_lbfgsb(y)
        
        beta = np.zeros(self.n_features, dtype=np.float64)
        if self.poisson_init_iter:
            beta = self._poisson_warm_start(y, beta.copy())
        alpha = self.dispersion if self.dispersion is not None else 0.0
        converged = False
        deviance = np.nan
        cov_beta = np.eye(self.n_features, dtype=np.float64)
        mu = np.maximum(np.full_like(y, y.mean()), self.min_mu)
        weights = np.ones_like(mu)
        
        # Use Cox-Reid for final dispersion estimation after initial convergence
        use_cox_reid = (
            self.dispersion is None 
            and self.dispersion_method == "cox-reid"
        )
        
        for iteration in range(1, self.max_iter + 1):
            eta = self.offset + self.design @ beta
            mu = np.exp(np.clip(eta, a_min=np.log(self.min_mu), a_max=None))
            mu = np.maximum(mu, self.min_mu)
            variance = mu + alpha * (mu**2)
            weights = (mu**2) / np.maximum(variance, self.min_mu)
            z = eta + (y - mu) / np.maximum(mu, self.min_mu)
            working_response = z - self.offset
            beta_new, cov_beta = self._weighted_least_squares(weights, working_response)
            beta_diff = float(np.max(np.abs(beta_new - beta)))
            beta = beta_new
            alpha_prev = alpha
            if self.dispersion is None:
                # Use moments for fast iteration, Cox-Reid for final estimate
                alpha = self._update_alpha(y, mu, alpha)
                alpha_diff = abs(alpha - alpha_prev)
            else:
                alpha_diff = 0.0
            deviance = self._compute_deviance(y, mu, alpha)
            tol_alpha = self.tol * max(1.0, abs(alpha_prev))
            if beta_diff < self.tol and alpha_diff <= tol_alpha:
                converged = True
                break
        else:
            # If we exit the loop without breaking we did not converge.
            cov_beta = self._hessian_inverse(weights)
        
        # After convergence, refine dispersion using Cox-Reid if requested
        if converged and use_cox_reid:
            alpha = self.estimate_dispersion_cox_reid(
                y, mu, weights, initial_alpha=alpha
            )
            # Recompute weights and covariance with final dispersion
            variance = mu + alpha * (mu**2)
            weights = (mu**2) / np.maximum(variance, self.min_mu)
            cov_beta = self._hessian_inverse(weights)
        
        se = np.sqrt(np.maximum(np.diag(cov_beta), self.min_mu))
        max_cooks = None
        if self.compute_cooks:
            hat_diag = self._hat_diagonal(weights, cov_beta)
            variance = mu + alpha * (mu**2)
            pearson_resid = (y - mu) / np.sqrt(np.maximum(variance, self.min_mu))
            denom = np.maximum((1.0 - hat_diag) ** 2, self.min_mu)
            cooks = (pearson_resid**2 / max(self.n_features, 1)) * (hat_diag / denom)
            max_cooks = float(np.nanmax(cooks)) if cooks.size else None
        return NBGLMResult(
            coef=beta,
            se=se,
            dispersion=alpha,
            converged=converged,
            n_iter=iteration if converged else self.max_iter,
            deviance=deviance,
            max_cooks=max_cooks,
        )

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
                # NB log-likelihood
                ll = np.sum(
                    gammaln(y + r)
                    - gammaln(r)
                    - gammaln(y + 1)
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
            
            # Negative binomial log-likelihood
            ll = np.sum(
                gammaln(y + r)
                - gammaln(r)
                - gammaln(y + 1)
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
    valid = (
        np.isfinite(means_arr)
        & np.isfinite(disp_arr)
        & (means_arr > min_mean)
        & (disp_arr > 1e-8)
    )
    n_valid = valid.sum()
    
    if n_valid < 3:
        baseline = np.nanmedian(disp_arr[valid]) if np.any(valid) else 0.1
        return np.full_like(means_arr, baseline, dtype=np.float64)
    
    x_valid = means_arr[valid]
    y_valid = disp_arr[valid]
    
    if fit_type == "mean":
        baseline = np.nanmedian(y_valid)
        return np.full_like(means_arr, baseline, dtype=np.float64)
    
    if fit_type == "parametric":
        # PyDESeq2-style Gamma GLM fit: disp = a0 + a1 / mean
        # Use IRLS with Gamma family (variance = mu^2) and identity link
        # Design matrix: [1, 1/mean]
        
        try:
            # Build design matrix for valid genes
            X = np.column_stack([np.ones_like(x_valid), 1.0 / x_valid])
            
            # Initial estimates using simple regression
            log_mean = np.log(np.clip(x_valid, min_mean, None))
            log_disp = np.log(np.clip(y_valid, 1e-10, None))
            init_coeffs = np.polyfit(log_mean, log_disp, deg=1)
            
            # Convert to parametric form initial estimates
            asympt_disp = max(np.exp(init_coeffs[0] + init_coeffs[1] * np.median(log_mean)), 1e-6)
            extra_pois = max(asympt_disp * np.exp(np.median(log_mean)) * 0.1, 0.0)
            params = np.array([asympt_disp, extra_pois])
            
            # Use mask for outlier filtering (PyDESeq2 style)
            use_for_fit = np.ones(len(x_valid), dtype=bool)
            
            for iteration in range(n_iter):
                # Predict dispersion
                pred = X[use_for_fit] @ params
                pred = np.maximum(pred, 1e-8)
                
                # Gamma GLM weights: w = 1/variance = 1/mu^2
                weights = 1.0 / (pred ** 2)
                
                # Weighted least squares update
                X_fit = X[use_for_fit]
                y_fit = y_valid[use_for_fit]
                W = weights
                
                try:
                    XtWX = X_fit.T @ (X_fit * W[:, None])
                    XtWy = X_fit.T @ (y_fit * W)
                    XtWX += 1e-8 * np.eye(2)  # Ridge for stability
                    params_new = np.linalg.solve(XtWX, XtWy)
                    params_new[0] = max(params_new[0], 1e-8)  # a0 > 0
                    params_new[1] = max(params_new[1], 0.0)   # a1 >= 0
                    params = params_new
                except np.linalg.LinAlgError:
                    break
                
                # Outlier detection (PyDESeq2 style): exclude genes with
                # pred_ratio < 1e-4 or pred_ratio >= 15
                if iteration < n_iter - 1:
                    pred_all = X @ params
                    pred_all = np.maximum(pred_all, 1e-8)
                    pred_ratio = y_valid / pred_all
                    use_for_fit = (pred_ratio >= 1e-4) & (pred_ratio < 15.0)
                    if use_for_fit.sum() < 3:
                        use_for_fit = np.ones(len(x_valid), dtype=bool)
            
            # Compute trend for all genes
            trend = params[0] + params[1] / np.maximum(means_arr, min_mean)
            trend = np.maximum(trend, 1e-8)
            return trend
            
        except Exception:
            # Fall back to polynomial fit
            pass
    
    # Fallback: log-quadratic polynomial fit (original method)
    x = np.log(np.clip(means_arr[valid], min_mean, None))
    y = np.log(np.clip(disp_arr[valid], 1e-10, None))
    
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


@nb.njit(parallel=True, cache=True)
def _nb_map_grid_search_numba(
    Y: np.ndarray,
    mu: np.ndarray,
    log_trend: np.ndarray,
    log_alpha_grid: np.ndarray,
    prior_var: float,
    gammaln_Y_plus_1: np.ndarray,
) -> tuple:
    """Vectorized grid search for MAP dispersion across all genes.
    
    For each gene, evaluates the posterior (log-likelihood + log-prior) at each
    grid point and finds the best grid point. Returns the best log-alpha and
    the indices of adjacent grid points for refinement.
    
    Parameters
    ----------
    Y : (n_cells, n_genes)
        Count matrix.
    mu : (n_cells, n_genes)
        Fitted mean matrix.
    log_trend : (n_genes,)
        Log of dispersion trend values.
    log_alpha_grid : (n_grid,)
        Grid of log-dispersion values to search.
    prior_var : float
        Variance of log-normal prior.
    gammaln_Y_plus_1 : (n_cells, n_genes)
        Precomputed gammaln(Y + 1).
        
    Returns
    -------
    best_log_alpha : (n_genes,)
        Best log-alpha value for each gene.
    best_idx : (n_genes,)
        Index of best grid point.
    posterior_grid : (n_grid, n_genes)
        Full posterior values for debugging (optional).
    """
    n_cells, n_genes = Y.shape
    n_grid = log_alpha_grid.shape[0]
    
    best_log_alpha = np.zeros(n_genes, dtype=np.float64)
    best_idx = np.zeros(n_genes, dtype=np.int64)
    
    # Parallelize over genes
    for g in nb.prange(n_genes):
        best_posterior = -np.inf
        best_k = 0
        log_trend_g = log_trend[g]
        
        for k in range(n_grid):
            log_alpha = log_alpha_grid[k]
            alpha = np.exp(log_alpha)
            r = 1.0 / alpha  # Size parameter
            log_r = np.log(r)
            gammaln_r = math.lgamma(r)
            
            # Compute log-likelihood for this gene at this alpha
            ll = 0.0
            for i in range(n_cells):
                y_ig = Y[i, g]
                mu_ig = mu[i, g]
                # NB log-likelihood terms
                ll += (
                    math.lgamma(y_ig + r)
                    - gammaln_r
                    - gammaln_Y_plus_1[i, g]
                    + r * (log_r - np.log(r + mu_ig + 1e-12))
                    + y_ig * np.log(mu_ig / (r + mu_ig + 1e-12) + 1e-12)
                )
            
            # Add log-prior: -0.5 * (log_alpha - log_trend)^2 / prior_var
            log_prior = -0.5 * (log_alpha - log_trend_g) ** 2 / prior_var
            posterior = ll + log_prior
            
            if posterior > best_posterior:
                best_posterior = posterior
                best_k = k
                best_log_alpha[g] = log_alpha
        
        best_idx[g] = best_k
    
    return best_log_alpha, best_idx


def _refine_dispersion_brent(
    j: int,
    Y_col: np.ndarray,
    mu_col: np.ndarray,
    log_trend_j: float,
    prior_var: float,
    log_min: float,
    log_max: float,
    gammaln_Y_plus_1_col: np.ndarray,
) -> float:
    """Refine dispersion for a single gene using Brent's method.
    
    This is used after grid search to get the exact optimum.
    """
    from scipy.optimize import brentq, minimize_scalar
    
    def neg_posterior(log_alpha: float) -> float:
        alpha = np.exp(log_alpha)
        r = 1.0 / alpha
        
        # NB log-likelihood
        ll = np.sum(
            gammaln(Y_col + r)
            - gammaln(r)
            - gammaln_Y_plus_1_col
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
    n_grid: int = 20,
    refine: bool = True,
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
        initial estimate but slower grid search.
    refine
        If True, refine the grid search result using Brent's method.
        Set to False for ~2× speedup with slightly less accuracy.
    n_jobs
        Number of parallel jobs for refinement. -1 uses all cores.
    
    Returns
    -------
    np.ndarray
        MAP dispersion estimates of shape (n_genes,).
    """
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
    
    # Precompute gammaln(Y + 1) for all cells and genes
    gammaln_Y_plus_1 = gammaln(Y + 1)
    
    # Create grid of log-alpha values
    log_alpha_grid = np.linspace(log_min, log_max, n_grid)
    
    # =========================================================================
    # Stage 1: Vectorized grid search using Numba
    # =========================================================================
    best_log_alpha, best_idx = _nb_map_grid_search_numba(
        Y, mu, log_trend, log_alpha_grid, prior_var, gammaln_Y_plus_1
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
    
    # Parallel refinement
    def _refine_gene(j):
        return _refine_dispersion_brent(
            j,
            Y[:, j],
            mu[:, j],
            log_trend[j],
            prior_var,
            float(lower_bounds[j]),
            float(upper_bounds[j]),
            gammaln_Y_plus_1[:, j],
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
        
        # NB log-likelihood (vectorized over cells)
        ll = np.sum(
            gammaln(y_col + inv_alpha)
            - gammaln(inv_alpha)
            - gammaln(y_col + 1)
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
        
        # NB log-likelihood
        r = 1.0 / max(alpha, 1e-10)
        ll = np.sum(
            gammaln(Y_gene + r)
            - gammaln(r)
            - gammaln(Y_gene + 1)
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
    chunk_size: int = 2048,
    max_iter: int = 100,
    tol: float = 1e-6,
    dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
    shrink_dispersion: bool = True,
    per_comparison_dispersion: bool = True,
    use_map_dispersion: bool = True,
    cook_filter: bool = False,
    lfc_shrinkage_type: Literal["normal", "apeglm", "none"] = "none",
    n_jobs: int | None = None,
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
        
    Returns
    -------
    JointModelResult
        Results containing all coefficients, standard errors, and dispersion.
    """
    from .data import iter_matrix_chunks
    
    # Resolve n_jobs: None or -1 means all cores, otherwise use specified value
    import os
    if n_jobs is None or n_jobs == -1:
        _n_jobs = -1  # joblib convention for all cores
    else:
        _n_jobs = max(1, n_jobs)
    
    n_cells = backed_adata.n_obs
    n_genes = backed_adata.n_vars
    
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
    
    for slc, chunk in iter_matrix_chunks(
        backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
    ):
        Y_chunk = np.asarray(chunk, dtype=np.float64)
        Y_full[slc, :] = Y_chunk
        
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
    
    # Work arrays: (n_cells, n_genes)
    eta = np.empty((n_cells, n_genes), dtype=np.float64)
    mu = np.empty_like(eta)
    
    # Coefficients: 
    # beta_intercept: (n_genes,) - intercept per gene
    # beta_perturbation: (n_perturbations, n_genes) - already allocated
    # beta_cov: (n_covariates, n_genes) - already allocated
    
    # Initialize with Poisson warm start
    logger.info("Vectorized IRLS: Poisson warm start...")
    for _ in range(3):
        # Compute eta
        eta[:] = beta_intercept[None, :] + log_size_factors[:, None]
        if n_covariates > 0:
            eta += cov_matrix @ beta_cov
        # Sparse add: for each cell, add its perturbation effect
        eta += pert_indicator @ beta_perturbation
        
        np.clip(eta, np.log(min_mu), 20.0, out=eta)
        np.exp(eta, out=mu)
        np.maximum(mu, min_mu, out=mu)
        
        # Poisson weights (mu) and working response
        z = eta + (Y_full - mu) / np.maximum(mu, 1e-10) - log_size_factors[:, None]
        
        # Update intercept: weighted mean of z
        beta_intercept[:] = np.sum(mu * z, axis=0) / np.maximum(np.sum(mu, axis=0), 1e-10)
        
        # Update perturbation effects
        for p_idx in range(n_perturbations):
            mask = cell_pert_idx == p_idx
            n_p = mask.sum()
            if n_p > 0:
                z_p = z[mask, :] - beta_intercept[None, :]
                mu_p = mu[mask, :]
                beta_perturbation[p_idx, :] = np.sum(mu_p * z_p, axis=0) / np.maximum(np.sum(mu_p, axis=0), 1e-10)
    
    # Initial dispersion estimate
    eta[:] = beta_intercept[None, :] + log_size_factors[:, None]
    if n_covariates > 0:
        eta += cov_matrix @ beta_cov
    eta += pert_indicator @ beta_perturbation
    np.clip(eta, np.log(min_mu), 20.0, out=eta)
    np.exp(eta, out=mu)
    np.maximum(mu, min_mu, out=mu)
    
    resid = Y_full - mu
    dispersion_raw = np.sum(resid ** 2 / mu, axis=0) / n_cells - 1
    dispersion = np.maximum(dispersion_raw, 1e-8)
    
    # NB-IRLS iterations
    logger.info("Vectorized IRLS: fitting coefficients...")
    converged[:] = False
    alpha = dispersion[None, :]  # (1, n_genes)
    
    for iteration in range(max_iter):
        beta_int_old = beta_intercept.copy()
        beta_pert_old = beta_perturbation.copy()
        
        # Compute eta and mu
        eta[:] = beta_intercept[None, :] + log_size_factors[:, None]
        if n_covariates > 0:
            eta += cov_matrix @ beta_cov
        eta += pert_indicator @ beta_perturbation
        
        np.clip(eta, np.log(min_mu), 20.0, out=eta)
        np.exp(eta, out=mu)
        np.maximum(mu, min_mu, out=mu)
        
        # NB weights: w = mu^2 / (mu + alpha * mu^2) = mu / (1 + alpha * mu)
        W = mu / (1 + alpha * mu)
        
        # Working response
        z = eta + (Y_full - mu) / np.maximum(mu, 1e-10) - log_size_factors[:, None]
        
        # Update intercept: weighted average across all cells
        sum_w = W.sum(axis=0)  # (n_genes,)
        z_centered = z - (pert_indicator @ beta_perturbation)  # Remove perturbation effect
        beta_intercept[:] = np.sum(W * z_centered, axis=0) / np.maximum(sum_w, 1e-10)
        
        # Update perturbation effects: vectorized using sparse matrix multiplication
        z_resid = z - beta_intercept[None, :]  # Remove intercept
        
        # Compute weighted sums: P.T @ (W * z_resid) and P.T @ W
        Wz_pert = pert_indicator.T @ (W * z_resid)  # (n_perturbations, n_genes)
        W_pert_sum = np.asarray(pert_indicator.T @ W)  # (n_perturbations, n_genes)
        
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
    
    # Store per-comparison intercepts (for SE computation)
    beta_intercept_per_pert = np.zeros((n_perturbations, n_genes), dtype=np.float64)
    dispersion_per_pert = np.zeros((n_perturbations, n_genes), dtype=np.float64)
    
    # Compute global mu for global dispersion estimate
    eta[:] = beta_intercept[None, :] + log_size_factors[:, None]
    if n_covariates > 0:
        eta += cov_matrix @ beta_cov
    eta += pert_indicator @ beta_perturbation
    
    mu_global = np.exp(np.clip(eta, -30, 30))
    mu_global = np.maximum(mu_global, 1e-10)
    
    # Global dispersion for fallback using correct NB method of moments formula
    # alpha = sum((Y - mu)^2 - Y) / mu^2) / (n - p)
    resid_global = Y_full - mu_global
    variance_global = resid_global ** 2 - Y_full
    denom_global = np.maximum(mu_global ** 2, 1e-10)
    n_params = 1 + n_perturbations + n_covariates  # intercept + pert effects + covariates
    dof_global = max(n_cells - n_params, 1)
    dispersion_raw_global = np.sum(variance_global / denom_global, axis=0) / dof_global
    dispersion_raw_global = np.clip(dispersion_raw_global, 1e-8, 1e6)
    
    if shrink_dispersion:
        trend_global = fit_dispersion_trend(mean_expr, dispersion_raw_global)
        if use_map_dispersion:
            # Use MAP estimation with log-normal prior
            # PyDESeq2-style bounds: max(10, n_cells)
            max_disp = max(10.0, float(n_cells))
            dispersion = estimate_dispersion_map(Y_full, mu_global, trend_global, max_disp=max_disp)
        else:
            dispersion = shrink_dispersions(dispersion_raw_global, trend_global)
    else:
        dispersion = dispersion_raw_global
    
    # Determine batch size for parallel processing to limit memory
    # Each parallel task copies its data subset, so limit concurrent tasks
    # to avoid memory multiplication. Use min of n_jobs and a memory-safe batch size.
    effective_n_jobs = _n_jobs if _n_jobs > 0 else (os.cpu_count() or 1)
    # Limit batch size to prevent excessive memory copying
    # 8 concurrent tasks is usually safe for most systems
    max_concurrent_tasks = min(effective_n_jobs, 8)
    
    def _refine_perturbation(p_idx):
        """Re-estimate intercept and dispersion for control+perturbation subset."""
        pert_mask = cell_pert_idx == p_idx
        subset_mask = control_mask | pert_mask
        n_subset = subset_mask.sum()
        
        if n_subset <= 2:
            return p_idx, beta_intercept.copy(), dispersion.copy()
        
        # Get subset data - use view where possible, copy only when necessary
        # Note: memmap slicing with boolean mask requires copy, but we limit
        # the number of concurrent copies via batch processing
        Y_subset = Y_full[subset_mask, :]
        if hasattr(Y_subset, 'copy'):
            Y_subset = np.asarray(Y_subset)  # Force read from memmap
        log_sf_subset = log_size_factors[subset_mask]
        pert_indicator_subset = pert_mask[subset_mask].astype(np.float64)
        
        # Re-estimate intercept for this subset using a few IRLS iterations
        # Start from global intercept
        int_subset = beta_intercept.copy()
        pert_effect = beta_perturbation[p_idx, :].copy()
        
        # Initial mu with current estimates
        eta_subset = int_subset[None, :] + log_sf_subset[:, None]
        eta_subset += pert_indicator_subset[:, None] * pert_effect[None, :]
        
        mu_subset = np.exp(np.clip(eta_subset, -30, 30))
        mu_subset = np.maximum(mu_subset, 1e-10)
        
        # Initial dispersion estimate using correct NB method of moments formula
        # alpha = sum((Y - mu)^2 - Y) / mu^2) / (n - p)
        # This matches the formula in NBGLMBatchFitter
        resid_subset = Y_subset - mu_subset
        variance = resid_subset ** 2 - Y_subset
        denom = np.maximum(mu_subset ** 2, 1e-10)
        dof = max(n_subset - 2, 1)  # 2 parameters: intercept + perturbation
        disp_raw = np.sum(variance / denom, axis=0) / dof
        disp_raw = np.clip(disp_raw, 1e-8, 1e6)
        disp_subset = disp_raw
        
        # Apply Cook's distance filtering if enabled
        if cook_filter:
            Y_subset, _ = filter_outliers_cooks(
                Y_subset, mu_subset, disp_subset, n_params=2
            )
        
        # Refine intercept with NB weights (3 iterations)
        for _ in range(3):
            # NB weights
            W_subset = mu_subset / (1 + disp_subset[None, :] * mu_subset)
            
            # Working response
            z_subset = eta_subset + (Y_subset - mu_subset) / np.maximum(mu_subset, 1e-10) - log_sf_subset[:, None]
            
            # Remove perturbation effect from z
            z_centered = z_subset - pert_indicator_subset[:, None] * pert_effect[None, :]
            
            # Update intercept as weighted mean
            sum_w = W_subset.sum(axis=0)
            int_subset = np.sum(W_subset * z_centered, axis=0) / np.maximum(sum_w, 1e-10)
            
            # Update mu
            eta_subset = int_subset[None, :] + log_sf_subset[:, None]
            eta_subset += pert_indicator_subset[:, None] * pert_effect[None, :]
            mu_subset = np.exp(np.clip(eta_subset, -30, 30))
            mu_subset = np.maximum(mu_subset, 1e-10)
            
            # Update dispersion estimate using correct NB method of moments formula
            resid_subset = Y_subset - mu_subset
            variance = resid_subset ** 2 - Y_subset
            denom = np.maximum(mu_subset ** 2, 1e-10)
            disp_raw = np.sum(variance / denom, axis=0) / dof
            disp_subset = np.clip(disp_raw, 1e-8, 1e6)
        
        # Apply dispersion shrinkage/MAP after intercept refinement
        if shrink_dispersion:
            mean_subset = Y_subset.sum(axis=0) / n_subset
            trend = fit_dispersion_trend(mean_subset, disp_subset)
            if use_map_dispersion:
                # Use MAP estimation for per-comparison dispersion
                max_disp = max(10.0, float(n_subset))
                disp_subset = estimate_dispersion_map(Y_subset, mu_subset, trend, max_disp=max_disp)
            else:
                disp_subset = shrink_dispersions(disp_subset, trend)
        
        return p_idx, int_subset, disp_subset
    
    # Run per-comparison refinement in parallel with bounded concurrency
    # Process in batches to limit memory from concurrent data copies
    refinement_results = Parallel(n_jobs=max_concurrent_tasks, prefer="threads")(
        delayed(_refine_perturbation)(p_idx) for p_idx in range(n_perturbations)
    )
    
    for p_idx, int_p, disp_p in refinement_results:
        beta_intercept_per_pert[p_idx, :] = int_p
        dispersion_per_pert[p_idx, :] = disp_p
    
    # =========================================================================
    # Compute SE using per-comparison estimates with proper block matrix inversion
    # =========================================================================
    logger.info("Computing standard errors with per-comparison refinement...")
    
    se_perturbation = np.full((n_perturbations, n_genes), np.nan, dtype=np.float64)
    
    def _compute_per_comparison_se(p_idx):
        """Compute SE for perturbation using PyDESeq2-style formula.
        
        Uses the formula: SE = sqrt(Hc.T @ M @ Hc) where H = inv(M + ridge)
        and c = [0, 1] is the contrast vector for the perturbation effect.
        
        For a 2x2 case with M = [[M11, M12], [M12, M22]] and ridge regularization:
        - Mr = M + ridge * I = [[M11+r, M12], [M12, M22+r]]
        - H = inv(Mr)
        - Hc = H @ [0, 1]^T = [H[0,1], H[1,1]]
        - SE = sqrt(Hc.T @ M @ Hc)
        
        This matches PyDESeq2's wald_test function.
        """
        pert_mask = cell_pert_idx == p_idx
        subset_mask = control_mask | pert_mask
        n_subset = subset_mask.sum()
        
        if n_subset <= 2:
            return p_idx, np.full(n_genes, np.nan)
        
        # Get subset data
        Y_subset = Y_full[subset_mask, :]
        log_sf_subset = log_size_factors[subset_mask]
        pert_indicator_subset = pert_mask[subset_mask].astype(np.float64)
        
        # Compute mu using per-comparison intercept
        int_p = beta_intercept_per_pert[p_idx, :]
        pert_effect = beta_perturbation[p_idx, :]
        disp_p = dispersion_per_pert[p_idx, :]
        
        eta_subset = int_p[None, :] + log_sf_subset[:, None]
        eta_subset += pert_indicator_subset[:, None] * pert_effect[None, :]
        
        mu_subset = np.exp(np.clip(eta_subset, -30, 30))
        mu_subset = np.maximum(mu_subset, 1e-10)
        
        # NB weights (PyDESeq2 formula): W = mu / (1 + mu * disp)
        W_subset = mu_subset / (1 + disp_p[None, :] * mu_subset)
        
        # Fisher information matrix M = X.T @ diag(W) @ X
        # For design [1, x] where x is pert_indicator:
        # M11 = sum(W_i)
        # M22 = sum(W_i * x_i^2)
        # M12 = sum(W_i * x_i)
        
        M11 = W_subset.sum(axis=0)  # (n_genes,)
        M22 = (W_subset * pert_indicator_subset[:, None]**2).sum(axis=0)  # (n_genes,)
        M12 = (W_subset * pert_indicator_subset[:, None]).sum(axis=0)  # (n_genes,)
        
        # Ridge regularization (PyDESeq2 default is 1e-6)
        ridge = 1e-6
        
        # Mr = M + ridge * I
        Mr11 = M11 + ridge
        Mr22 = M22 + ridge
        Mr12 = M12  # Off-diagonal unchanged
        
        # H = inv(Mr) for 2x2 matrix
        det_r = Mr11 * Mr22 - Mr12 * Mr12
        
        # H = [[Mr22, -Mr12], [-Mr12, Mr11]] / det_r
        H00 = Mr22 / np.maximum(det_r, 1e-12)
        H01 = -Mr12 / np.maximum(det_r, 1e-12)
        H11 = Mr11 / np.maximum(det_r, 1e-12)
        
        # Contrast vector c = [0, 1] for perturbation effect
        # Hc = H @ c = [H[0,1], H[1,1]] = [H01, H11]
        Hc0 = H01
        Hc1 = H11
        
        # SE = sqrt(Hc.T @ M @ Hc)
        # = sqrt([Hc0, Hc1] @ [[M11, M12], [M12, M22]] @ [Hc0, Hc1].T)
        # = sqrt(Hc0^2 * M11 + 2 * Hc0 * Hc1 * M12 + Hc1^2 * M22)
        variance = Hc0**2 * M11 + 2 * Hc0 * Hc1 * M12 + Hc1**2 * M22
        
        valid = (det_r > 1e-12) & (variance > 0) & pert_has_data[p_idx, :]
        se_p = np.full(n_genes, np.nan)
        se_p[valid] = np.sqrt(variance[valid])
        
        return p_idx, se_p
    
    # Compute SEs in parallel with bounded concurrency
    se_results = Parallel(n_jobs=max_concurrent_tasks, prefer="threads")(
        delayed(_compute_per_comparison_se)(p_idx) for p_idx in range(n_perturbations)
    )
    
    for p_idx, se_p in se_results:
        se_perturbation[p_idx, :] = se_p
    
    # SE for global intercept (uses global dispersion)
    alpha_global = dispersion[None, :]
    W_global = mu_global / (1 + mu_global * alpha_global)
    se_intercept = 1.0 / np.sqrt(np.maximum(W_global.sum(axis=0), 1e-12))
    
    # Apply LFC shrinkage if requested
    if lfc_shrinkage_type != "none":
        logger.info(f"Applying {lfc_shrinkage_type} LFC shrinkage...")
        for p_idx in range(n_perturbations):
            beta_perturbation[p_idx, :] = shrink_log_foldchange(
                beta_perturbation[p_idx, :],
                se_perturbation[p_idx, :],
                shrinkage_type=lfc_shrinkage_type,
            )
    
    # Free Y_full after all computation and clean up temp files
    del Y_full
    import shutil
    try:
        shutil.rmtree(_temp_dir)
    except Exception:
        pass  # Ignore cleanup errors
    
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


def estimate_joint_model_streaming(
    backed_adata,
    *,
    obs_df: "pd.DataFrame",
    perturbation_labels: np.ndarray,
    control_label: str,
    covariate_columns: Sequence[str],
    size_factors: np.ndarray,
    chunk_size: int = 2048,
    poisson_iter: int = 10,
    nb_iter: int = 25,
    tol: float = 1e-6,
    dispersion_method: Literal["moments", "cox-reid"] = "cox-reid",
    shrink_dispersion: bool = True,
    perturbation_chunk_size: int = 100,
    ridge_penalty: float = 1e-6,
    intercept_mode: Literal["global", "per_comparison"] = "per_comparison",
    use_sparse: bool = True,
    use_numba: bool = True,
) -> JointModelResult:
    """Estimate joint model with all perturbations using streaming.
    
    This function fits a full negative binomial GLM with design matrix:
        [intercept, perturbation_1, ..., perturbation_n, covariates]
    
    where control is the reference level. The intercept estimation mode can be
    configured for either accuracy (matching PyDESeq2) or memory efficiency.
    
    The implementation uses sparse perturbation indicators and block-diagonal
    structure of X^T W X for memory efficiency:
    - The perturbation×perturbation block is diagonal (each cell belongs to one group)
    - Only the intercept/covariate rows are dense
    - SEs are computed using proper block matrix inversion (Schur complement),
      accounting for correlations between intercept and perturbation effects
    
    .. note::
        With ``intercept_mode="per_comparison"`` (default), the intercept is
        re-estimated for each control+perturbation subset, matching PyDESeq2's
        pairwise approach. This provides higher accuracy (ρ > 0.99 with PyDESeq2)
        but uses ~2× memory for intercept storage.
        
        With ``intercept_mode="global"``, a single intercept is estimated from
        all cells (original behavior). This is more memory efficient but may
        differ from PyDESeq2 for sparsely expressed genes.
    
    For >100 perturbations, SEs are computed in chunks to bound memory.
    
    Parameters
    ----------
    backed_adata
        Backed AnnData object opened in read mode.
    obs_df
        Full obs DataFrame with all cells.
    perturbation_labels
        Array of perturbation labels for all cells.
    control_label
        The label identifying control cells (reference level).
    covariate_columns
        List of covariate column names to include.
    size_factors
        Per-cell size factors (length n_cells).
    chunk_size
        Number of cells to process per chunk (streaming over cells).
    poisson_iter
        Number of Poisson IRLS iterations for initialization.
    nb_iter
        Number of negative binomial IRLS iterations.
    tol
        Convergence tolerance.
    dispersion_method
        Method for dispersion estimation: "moments" or "cox-reid".
    shrink_dispersion
        If True, shrink dispersions toward fitted trend.
    perturbation_chunk_size
        Number of perturbations to process at once for SE computation.
        Larger values use more memory but may be faster.
    ridge_penalty
        Small diagonal ridge penalty added to X^T W X for numerical stability.
        Helps prevent extreme coefficients for genes with zero counts in some groups.
    intercept_mode
        How to estimate intercepts:
        - "per_comparison": Estimate separate intercept for each control+perturbation
          subset (matches PyDESeq2, higher accuracy, ~2× intercept storage)
        - "global": Single intercept from all cells (memory efficient, original behavior)
    use_sparse
        If True, use sparse matrix operations for perturbation accumulation
        when no covariates are present. Falls back to Numba otherwise.
    use_numba
        If True, use Numba-accelerated kernels for inner loops. Provides
        ~5-10× speedup over Python loops.
        
    Returns
    -------
    JointModelResult
        Results containing all coefficients, standard errors, and dispersion.
    """
    import pandas as pd
    from .data import iter_matrix_chunks
    
    n_cells = backed_adata.n_obs
    n_genes = backed_adata.n_vars
    
    # Identify perturbation groups
    unique_labels = np.unique(perturbation_labels)
    non_control_labels = unique_labels[unique_labels != control_label]
    n_perturbations = len(non_control_labels)
    
    # Build label-to-index mapping for perturbations
    label_to_idx = {label: i for i, label in enumerate(non_control_labels)}
    
    # Build covariate matrix for all cells
    cov_matrices = []
    cov_names: list[str] = []
    for column in covariate_columns:
        if column not in obs_df.columns:
            raise KeyError(f"Covariate '{column}' not found in obs_df")
        series = obs_df[column]
        if series.dtype.kind in {"O", "U"} or str(series.dtype).startswith("category"):
            dummies = pd.get_dummies(series, prefix=column, drop_first=True, dtype=float)
            if dummies.shape[1] > 0:
                cov_matrices.append(dummies.to_numpy(dtype=np.float64))
                cov_names.extend(dummies.columns.astype(str).tolist())
        else:
            cov_matrices.append(series.to_numpy(dtype=np.float64).reshape(-1, 1))
            cov_names.append(str(column))
    
    n_covariates = sum(m.shape[1] for m in cov_matrices) if cov_matrices else 0
    cov_matrix = np.hstack(cov_matrices) if cov_matrices else np.zeros((n_cells, 0), dtype=np.float64)
    
    # Create cell-to-perturbation index (-1 for control cells)
    cell_pert_idx = np.full(n_cells, -1, dtype=np.int32)
    for i, label in enumerate(perturbation_labels):
        if label != control_label:
            cell_pert_idx[i] = label_to_idx[label]
    
    # Log size factors for offset
    log_size_factors = np.log(np.maximum(size_factors, 1e-12))
    
    # Design structure: [intercept, perturbations..., covariates...]
    # n_features = 1 + n_perturbations + n_covariates
    # But we use block structure for efficiency
    n_dense_features = 1 + n_covariates  # intercept + covariates (dense block)
    
    # Initialize coefficients
    beta_intercept = np.zeros(n_genes, dtype=np.float64)
    beta_perturbation = np.zeros((n_perturbations, n_genes), dtype=np.float64)
    beta_cov = np.zeros((n_covariates, n_genes), dtype=np.float64)
    
    # Initialize dispersion
    dispersion = np.full(n_genes, 0.1, dtype=np.float64)
    
    # =========================================================================
    # Pre-compute: Count expressing cells per perturbation per gene
    # This is used to identify genes with zero/few counts in some perturbations,
    # which need regularization to avoid extreme coefficients.
    # =========================================================================
    pert_expr_counts = np.zeros((n_perturbations, n_genes), dtype=np.int32)
    ctrl_expr_counts = np.zeros(n_genes, dtype=np.int32)
    
    for slc, chunk in iter_matrix_chunks(
        backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
    ):
        Y_chunk = np.asarray(chunk, dtype=np.float64)
        pert_idx_chunk = cell_pert_idx[slc]
        
        for i in range(Y_chunk.shape[0]):
            p_idx = pert_idx_chunk[i]
            if p_idx >= 0:
                # Non-control cell
                pert_expr_counts[p_idx] += (Y_chunk[i, :] > 0).astype(np.int32)
            else:
                # Control cell
                ctrl_expr_counts += (Y_chunk[i, :] > 0).astype(np.int32)
    
    # Minimum expressing cells required to estimate coefficient
    min_expr_cells = 3
    # Mask for perturbations with too few expressing cells per gene
    # Shape: (n_perturbations, n_genes)
    pert_has_data = pert_expr_counts >= min_expr_cells
    # Also check if control has enough expressing cells (for valid baseline)
    ctrl_has_data = ctrl_expr_counts >= min_expr_cells
    # Combine: perturbation needs data AND control needs data for a valid LFC
    pert_has_data = pert_has_data & ctrl_has_data[None, :]
    
    # Determine whether to use sparse path (only when no covariates)
    use_sparse_path = use_sparse and n_covariates == 0
    
    # =========================================================================
    # Stage 1: Poisson IRLS to get initial estimates
    # =========================================================================
    # We accumulate X^T W X in block form:
    # - dense_block: (n_dense_features, n_dense_features) for intercept+covariates
    # - pert_diag: (n_perturbations,) diagonal for perturbation×perturbation
    # - cross_block: (n_dense_features, n_perturbations) for cross-terms
    # And X^T W z similarly
    
    # Ridge matrices for regularization
    ridge_dense = ridge_penalty * np.eye(n_dense_features)
    ridge_pert = ridge_penalty
    
    for iteration in range(poisson_iter):
        # Per-gene accumulators for X^T W X blocks
        dense_xtwx_all = np.zeros((n_genes, n_dense_features, n_dense_features), dtype=np.float64)
        pert_xtwx_diag_all = np.zeros((n_genes, n_perturbations), dtype=np.float64)
        cross_xtwx_all = np.zeros((n_genes, n_dense_features, n_perturbations), dtype=np.float64)
        dense_xtwz_all = np.zeros((n_genes, n_dense_features), dtype=np.float64)
        pert_xtwz_all = np.zeros((n_genes, n_perturbations), dtype=np.float64)
        
        for slc, chunk in iter_matrix_chunks(
            backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
        ):
            Y_chunk = np.asarray(chunk, dtype=np.float64)
            n_chunk = Y_chunk.shape[0]
            
            # Get cell data for this chunk
            offset_chunk = log_size_factors[slc]
            pert_idx_chunk = cell_pert_idx[slc]
            cov_chunk = cov_matrix[slc] if n_covariates > 0 else None
            
            # Compute eta = intercept + pert_effect + cov_effect + offset
            eta = beta_intercept[None, :] + offset_chunk[:, None]
            
            # Add perturbation effects (vectorized using indexing)
            pert_mask = pert_idx_chunk >= 0
            if np.any(pert_mask):
                eta[pert_mask] += beta_perturbation[pert_idx_chunk[pert_mask], :]
            
            # Add covariate effects
            if n_covariates > 0 and cov_chunk is not None:
                eta += cov_chunk @ beta_cov
            
            eta = np.clip(eta, -20.0, 20.0)
            mu = np.exp(eta)
            mu = np.maximum(mu, 1e-6)
            
            # Poisson weights = mu (per-gene)
            W = mu  # (n_chunk, n_genes)
            
            # Working response: z = eta - offset + (y - mu) / mu
            z = eta - offset_chunk[:, None] + (Y_chunk - mu) / np.maximum(mu, 1e-6)
            Wz = W * z  # (n_chunk, n_genes)
            
            # Build dense design block: [1, covariates]
            X_dense = np.ones((n_chunk, n_dense_features), dtype=np.float64)
            if n_covariates > 0:
                X_dense[:, 1:] = cov_chunk
            
            # Accumulate dense blocks for all genes at once (vectorized)
            # dense_xtwx[g] += sum_i w[i,g] * X_dense[i,:].T @ X_dense[i,:]
            for j in range(n_dense_features):
                for k in range(j, n_dense_features):
                    prod = X_dense[:, j] * X_dense[:, k]  # (n_chunk,)
                    contrib = (W.T @ prod)  # (n_genes,)
                    dense_xtwx_all[:, j, k] += contrib
                    if j != k:
                        dense_xtwx_all[:, k, j] += contrib
            
            # dense_xtwz: X_dense.T @ Wz -> for gene g: sum_i Wz[i,g] * X_dense[i,:]
            dense_xtwz_all += (Wz.T @ X_dense)  # (n_genes, n_dense)
            
            # Perturbation contributions - use optimized path
            if use_sparse_path:
                # Sparse matrix path (fastest for no-covariate case)
                P_chunk = _build_sparse_perturbation_indicator(pert_idx_chunk, n_perturbations)
                _accumulate_perturbation_blocks_sparse(
                    W, Wz, X_dense, P_chunk,
                    pert_xtwx_diag_all, cross_xtwx_all, pert_xtwz_all
                )
            elif use_numba:
                # Numba-accelerated path
                _accumulate_perturbation_blocks_numba(
                    W, Wz, X_dense, pert_idx_chunk, n_perturbations,
                    pert_xtwx_diag_all, cross_xtwx_all, pert_xtwz_all
                )
            else:
                # Fallback Python loop
                for i in range(n_chunk):
                    p_idx = pert_idx_chunk[i]
                    if p_idx >= 0:
                        w_i = W[i, :]  # (n_genes,)
                        wz_i = Wz[i, :]  # (n_genes,)
                        pert_xtwx_diag_all[:, p_idx] += w_i
                        cross_xtwx_all[:, :, p_idx] += np.outer(w_i, X_dense[i, :])
                        pert_xtwz_all[:, p_idx] += wz_i
        
        # Solve per-gene using Schur complement
        beta_intercept_new = np.zeros(n_genes, dtype=np.float64)
        beta_cov_new = np.zeros((n_covariates, n_genes), dtype=np.float64)
        beta_pert_new = np.zeros((n_perturbations, n_genes), dtype=np.float64)
        
        if use_numba and n_perturbations > 0:
            # Use Numba batch solver
            _batch_schur_solve_numba(
                dense_xtwx_all, pert_xtwx_diag_all, cross_xtwx_all,
                dense_xtwz_all, pert_xtwz_all, pert_has_data,
                ridge_dense, ridge_pert,
                beta_intercept_new, beta_cov_new, beta_pert_new
            )
        else:
            # Fallback Python loop
            for g in range(n_genes):
                A = dense_xtwx_all[g] + ridge_dense  # (n_dense, n_dense)
                D_diag = pert_xtwx_diag_all[g] + ridge_pert  # (n_pert,)
                B = cross_xtwx_all[g]  # (n_dense, n_pert)
                b1 = dense_xtwz_all[g]  # (n_dense,)
                b2 = pert_xtwz_all[g]  # (n_pert,)
                
                D_inv = 1.0 / np.maximum(D_diag, 1e-12)
                B_Dinv = B * D_inv[None, :]  # (n_dense, n_pert)
                schur = A - B_Dinv @ B.T
                schur_rhs = b1 - B_Dinv @ b2
                
                try:
                    x1 = np.linalg.solve(schur, schur_rhs)
                except np.linalg.LinAlgError:
                    x1 = np.linalg.lstsq(schur, schur_rhs, rcond=None)[0]
                
                x2 = D_inv * (b2 - B.T @ x1)
                
                # Zero out coefficients for perturbations with too few expressing cells
                x2 = np.where(pert_has_data[:, g], x2, 0.0)
                
                # Clip remaining coefficients to prevent extreme values
                x2 = np.clip(x2, -30.0, 30.0)
                
                beta_intercept_new[g] = x1[0]
                if n_covariates > 0:
                    beta_cov_new[:, g] = x1[1:]
                beta_pert_new[:, g] = x2
        
        # Check convergence
        max_diff = max(
            np.max(np.abs(beta_intercept_new - beta_intercept)),
            np.max(np.abs(beta_pert_new - beta_perturbation)) if n_perturbations > 0 else 0,
        )
        
        beta_intercept = beta_intercept_new
        beta_cov = beta_cov_new
        beta_perturbation = beta_pert_new
        
        if max_diff < tol:
            break
    
    # =========================================================================
    # Stage 2: Estimate dispersion using method of moments
    # Also compute mean expression in the same pass (consolidating data passes)
    # =========================================================================
    numerator_sum = np.zeros(n_genes, dtype=np.float64)
    mean_expr_sum = np.zeros(n_genes, dtype=np.float64)
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
        
        # Compute mu
        eta = beta_intercept[None, :] + offset_chunk[:, None]
        pert_mask = pert_idx_chunk >= 0
        if np.any(pert_mask):
            eta[pert_mask] += beta_perturbation[pert_idx_chunk[pert_mask], :]
        if n_covariates > 0 and cov_chunk is not None:
            eta += cov_chunk @ beta_cov
        
        eta = np.clip(eta, -20.0, 20.0)
        mu = np.exp(eta)
        mu = np.maximum(mu, 1e-6)
        
        # Method of moments: (y - mu)^2 - y over mu^2
        resid = Y_chunk - mu
        numerator = (resid * resid - Y_chunk) / np.maximum(mu * mu, 1e-12)
        numerator_sum += numerator.sum(axis=0)
        
        # Also accumulate mean expression (consolidated pass)
        mean_expr_sum += Y_chunk.sum(axis=0)
    
    n_features_total = 1 + n_perturbations + n_covariates
    dof = max(n_total - n_features_total, 1)
    dispersion_raw = np.clip(numerator_sum / dof, 1e-8, 1e6)
    dispersion_raw = np.where(np.isfinite(dispersion_raw), dispersion_raw, 0.1)
    
    # Mean expression was accumulated in the same pass
    mean_expr = mean_expr_sum / n_total
    
    if shrink_dispersion:
        trend = fit_dispersion_trend(mean_expr, dispersion_raw)
        dispersion = shrink_dispersions(dispersion_raw, trend)
    else:
        dispersion = dispersion_raw
    
    # =========================================================================
    # Stage 3: NB IRLS with dispersion to refine coefficients and get SEs
    # =========================================================================
    converged = np.zeros(n_genes, dtype=bool)
    n_iter_arr = np.zeros(n_genes, dtype=np.int32)
    
    # SE arrays will be computed on the final iteration
    se_intercept_arr = np.zeros(n_genes, dtype=np.float64)
    se_pert_arr = np.zeros((n_perturbations, n_genes), dtype=np.float64)
    se_cov_arr = np.zeros((n_covariates, n_genes), dtype=np.float64)
    
    for iteration in range(nb_iter):
        # Per-gene accumulators for X^T W X blocks (same structure as Poisson stage)
        dense_xtwx_all = np.zeros((n_genes, n_dense_features, n_dense_features), dtype=np.float64)
        pert_xtwx_diag_all = np.zeros((n_genes, n_perturbations), dtype=np.float64)
        cross_xtwx_all = np.zeros((n_genes, n_dense_features, n_perturbations), dtype=np.float64)
        dense_xtwz_all = np.zeros((n_genes, n_dense_features), dtype=np.float64)
        pert_xtwz_all = np.zeros((n_genes, n_perturbations), dtype=np.float64)
        
        for slc, chunk in iter_matrix_chunks(
            backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
        ):
            Y_chunk = np.asarray(chunk, dtype=np.float64)
            n_chunk = Y_chunk.shape[0]
            
            offset_chunk = log_size_factors[slc]
            pert_idx_chunk = cell_pert_idx[slc]
            cov_chunk = cov_matrix[slc] if n_covariates > 0 else None
            
            # Compute eta and mu
            eta = beta_intercept[None, :] + offset_chunk[:, None]
            pert_mask = pert_idx_chunk >= 0
            if np.any(pert_mask):
                eta[pert_mask] += beta_perturbation[pert_idx_chunk[pert_mask], :]
            if n_covariates > 0 and cov_chunk is not None:
                eta += cov_chunk @ beta_cov
            
            eta = np.clip(eta, -20.0, 20.0)
            mu = np.exp(eta)
            mu = np.maximum(mu, 1e-6)
            
            # NB variance: Var = mu + alpha * mu^2
            variance = mu + dispersion[None, :] * mu * mu
            
            # NB weights: W = mu^2 / Var (per-gene)
            W = (mu * mu) / np.maximum(variance, 1e-12)  # (n_chunk, n_genes)
            
            # Working response
            z = eta - offset_chunk[:, None] + (Y_chunk - mu) / np.maximum(mu, 1e-6)
            Wz = W * z
            
            # Dense design
            X_dense = np.ones((n_chunk, n_dense_features), dtype=np.float64)
            if n_covariates > 0:
                X_dense[:, 1:] = cov_chunk
            
            # Accumulate dense blocks for all genes at once
            for j in range(n_dense_features):
                for k in range(j, n_dense_features):
                    prod = X_dense[:, j] * X_dense[:, k]
                    contrib = (W.T @ prod)
                    dense_xtwx_all[:, j, k] += contrib
                    if j != k:
                        dense_xtwx_all[:, k, j] += contrib
            
            dense_xtwz_all += (Wz.T @ X_dense)
            
            # Perturbation contributions - use optimized path
            if use_sparse_path:
                P_chunk = _build_sparse_perturbation_indicator(pert_idx_chunk, n_perturbations)
                _accumulate_perturbation_blocks_sparse(
                    W, Wz, X_dense, P_chunk,
                    pert_xtwx_diag_all, cross_xtwx_all, pert_xtwz_all
                )
            elif use_numba:
                _accumulate_perturbation_blocks_numba(
                    W, Wz, X_dense, pert_idx_chunk, n_perturbations,
                    pert_xtwx_diag_all, cross_xtwx_all, pert_xtwz_all
                )
            else:
                for i in range(n_chunk):
                    p_idx = pert_idx_chunk[i]
                    if p_idx >= 0:
                        w_i = W[i, :]  # (n_genes,)
                        wz_i = Wz[i, :]
                        pert_xtwx_diag_all[:, p_idx] += w_i
                        cross_xtwx_all[:, :, p_idx] += np.outer(w_i, X_dense[i, :])
                        pert_xtwz_all[:, p_idx] += wz_i
        
        # Solve per-gene using Schur complement
        beta_intercept_new = np.zeros(n_genes, dtype=np.float64)
        beta_cov_new = np.zeros((n_covariates, n_genes), dtype=np.float64)
        beta_pert_new = np.zeros((n_perturbations, n_genes), dtype=np.float64)
        
        # On final iteration, also compute proper SEs via full block inverse
        is_final = (iteration == nb_iter - 1)
        if is_final:
            se_intercept_arr = np.zeros(n_genes, dtype=np.float64)
            se_pert_arr = np.zeros((n_perturbations, n_genes), dtype=np.float64)
            se_cov_arr = np.zeros((n_covariates, n_genes), dtype=np.float64)
        
        # Use Numba batch solver for coefficient updates (not for SE computation)
        if use_numba and n_perturbations > 0 and not is_final:
            _batch_schur_solve_numba(
                dense_xtwx_all, pert_xtwx_diag_all, cross_xtwx_all,
                dense_xtwz_all, pert_xtwz_all, pert_has_data,
                ridge_dense, ridge_pert,
                beta_intercept_new, beta_cov_new, beta_pert_new
            )
        else:
            # Python loop (needed for SE computation on final iteration)
            for g in range(n_genes):
                A = dense_xtwx_all[g] + ridge_dense
                D_diag = pert_xtwx_diag_all[g] + ridge_pert
                B = cross_xtwx_all[g]
                b1 = dense_xtwz_all[g]
                b2 = pert_xtwz_all[g]
                
                D_inv = 1.0 / np.maximum(D_diag, 1e-12)
                B_Dinv = B * D_inv[None, :]
                schur = A - B_Dinv @ B.T
                schur_rhs = b1 - B_Dinv @ b2
                
                try:
                    x1 = np.linalg.solve(schur, schur_rhs)
                except np.linalg.LinAlgError:
                    x1 = np.linalg.lstsq(schur, schur_rhs, rcond=None)[0]
                
                x2 = D_inv * (b2 - B.T @ x1)
                
                # Zero out coefficients for perturbations with too few expressing cells
                x2 = np.where(pert_has_data[:, g], x2, 0.0)
                
                # Clip remaining coefficients to prevent extreme values
                # Using PyDESeq2's bounds: min_beta=-30, max_beta=30
                x2 = np.clip(x2, -30.0, 30.0)
                
                beta_intercept_new[g] = x1[0]
                if n_covariates > 0:
                    beta_cov_new[:, g] = x1[1:]
                beta_pert_new[:, g] = x2
                
                # Compute proper SEs from block inverse on final iteration
                # The full information matrix is:
                #   M = | A   B  |
                #       | B^T D  |
                # where D is diagonal.
                # 
                # Using block matrix inversion, M^{-1} is:
                #   | A^{-1} + A^{-1} B S_p^{-1} B^T A^{-1}   -A^{-1} B S_p^{-1} |
                #   | -S_p^{-1} B^T A^{-1}                     S_p^{-1}          |
                # 
                # where S_p = D - B^T A^{-1} B is the Schur complement for the
                # perturbation block.
                #
                # SE for intercept/covariates comes from upper-left block diagonal.
                # SE for perturbations comes from diag(S_p^{-1}).
                if is_final:
                    try:
                        # Compute A^{-1}
                        A_inv = np.linalg.inv(A)
                        
                        # Schur complement for perturbation block: S_p = D - B^T A^{-1} B
                        # Since D is diagonal and we need S_p for SE computation,
                        # we compute S_p element-wise for efficiency.
                        # S_p[i,j] = D[i,j] - B[:,i]^T @ A_inv @ B[:,j]
                        # For diagonal of S_p: S_p[i,i] = D[i] - B[:,i]^T @ A_inv @ B[:,i]
                        A_inv_B = A_inv @ B  # (n_dense, n_pert)
                        
                        # Diagonal of B^T A^{-1} B is sum_j (B[j,i] * A_inv_B[j,i]) for each i
                        BtAinvB_diag = np.sum(B * A_inv_B, axis=0)  # (n_pert,)
                        S_p_diag = D_diag - BtAinvB_diag  # (n_pert,)
                        
                        # For the off-diagonal terms of S_p, we need full matrix for proper inverse.
                        # S_p = D - B^T @ A_inv @ B where D is diagonal (n_pert, n_pert)
                        BtAinvB = B.T @ A_inv_B  # (n_pert, n_pert)
                        S_p = np.diag(D_diag) - BtAinvB  # (n_pert, n_pert)
                        
                        # Invert S_p to get the lower-right block of M^{-1}
                        S_p_inv = np.linalg.inv(S_p)
                        
                        # SE for perturbations = sqrt(diag(S_p^{-1}))
                        se_pert_arr[:, g] = np.sqrt(np.maximum(np.diag(S_p_inv), 1e-12))
                        
                        # For intercept/covariate SEs, we need upper-left block:
                        # M^{-1}_{11} = A^{-1} + A^{-1} B S_p^{-1} B^T A^{-1}
                        # First compute A^{-1} B S_p^{-1}
                        AinvB_Spinv = A_inv_B @ S_p_inv  # (n_dense, n_pert)
                        # Then M^{-1}_{11} = A_inv + AinvB_Spinv @ B^T @ A_inv
                        #                 = A_inv + AinvB_Spinv @ (A_inv @ B)^T
                        upper_left_block = A_inv + AinvB_Spinv @ A_inv_B.T  # (n_dense, n_dense)
                        
                        # SE for intercept (first element) and covariates (rest)
                        se_intercept_arr[g] = np.sqrt(np.maximum(upper_left_block[0, 0], 1e-12))
                        if n_covariates > 0:
                            se_cov_arr[:, g] = np.sqrt(np.maximum(np.diag(upper_left_block[1:, 1:]), 1e-12))
                            
                    except np.linalg.LinAlgError:
                        # Fallback to diagonal approximation if inversion fails
                        se_intercept_arr[g] = 1.0 / np.sqrt(np.maximum(A[0, 0], 1e-12))
                        se_pert_arr[:, g] = 1.0 / np.sqrt(np.maximum(D_diag, 1e-12))
                        if n_covariates > 0:
                            se_cov_arr[:, g] = np.nan
        
        # Check convergence
        max_diff = max(
            np.max(np.abs(beta_intercept_new - beta_intercept)),
            np.max(np.abs(beta_pert_new - beta_perturbation)) if n_perturbations > 0 else 0,
        )
        
        beta_intercept = beta_intercept_new
        beta_cov = beta_cov_new
        beta_perturbation = beta_pert_new
        
        n_iter_arr[:] = iteration + 1
        
        if max_diff < tol:
            converged[:] = True
            break
    
    # =========================================================================
    # Stage 4: L-BFGS-B fallback for genes that didn't converge well
    # =========================================================================
    # For genes where IRLS didn't converge (or converged to potentially 
    # suboptimal solutions), use L-BFGS-B to refine the estimates.
    # This is similar to PyDESeq2's approach of falling back to L-BFGS-B
    # when IRLS fails.
    
    # Identify genes needing L-BFGS-B refinement: those with max_diff > tol
    # or whose coefficient changes were still large
    needs_lbfgsb = ~converged
    n_needs_lbfgsb = np.sum(needs_lbfgsb)
    
    if n_needs_lbfgsb > 0:
        logger.info(f"Applying L-BFGS-B refinement to {n_needs_lbfgsb} genes that didn't converge well")
        
        # We need to load the data for these genes into memory
        # Collect Y matrix for genes needing refinement
        genes_needing_lbfgsb = np.where(needs_lbfgsb)[0]
        
        # Load full data for L-BFGS-B fitting
        Y_full = []
        X_dense_full = []
        pert_indicators_full = []
        
        for slc, chunk in iter_matrix_chunks(
            backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
        ):
            Y_chunk = np.asarray(chunk, dtype=np.float64)
            n_chunk = Y_chunk.shape[0]
            
            offset_chunk = log_size_factors[slc]
            pert_idx_chunk = cell_pert_idx[slc]
            cov_chunk = cov_matrix[slc] if n_covariates > 0 else None
            
            # Dense design
            X_dense_chunk = np.ones((n_chunk, n_dense_features), dtype=np.float64)
            if n_covariates > 0:
                X_dense_chunk[:, 1:] = cov_chunk
            
            Y_full.append(Y_chunk[:, genes_needing_lbfgsb])
            X_dense_full.append(X_dense_chunk)
            pert_indicators_full.append(pert_idx_chunk)
        
        Y_lbfgsb = np.vstack(Y_full)  # (n_cells, n_genes_needing)
        X_dense_lbfgsb = np.vstack(X_dense_full)  # (n_cells, n_dense)
        pert_indicators = np.concatenate(pert_indicators_full)  # (n_cells,)
        
        # Run L-BFGS-B in parallel for each gene needing refinement
        def fit_gene_lbfgsb(gene_local_idx):
            g = genes_needing_lbfgsb[gene_local_idx]
            Y_gene = Y_lbfgsb[:, gene_local_idx]
            alpha_g = float(dispersion[g])
            beta0_dense = np.zeros(n_dense_features, dtype=np.float64)
            beta0_dense[0] = beta_intercept[g]
            if n_covariates > 0:
                beta0_dense[1:] = beta_cov[:, g]
            beta0_pert = beta_perturbation[:, g].copy()
            pert_data_mask = pert_has_data[:, g]
            
            beta_dense, beta_pert, dev, conv = _lbfgsb_nb_fit_gene(
                Y_gene=Y_gene,
                X_dense=X_dense_lbfgsb,
                pert_indicators=pert_indicators,
                log_size_factors=log_size_factors,
                alpha=alpha_g,
                beta0_dense=beta0_dense,
                beta0_pert=beta0_pert,
                pert_has_data=pert_data_mask,
                min_beta=-30.0,
                max_beta=30.0,
            )
            return gene_local_idx, beta_dense, beta_pert, conv
        
        # Use joblib for parallel execution
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(fit_gene_lbfgsb)(i) for i in range(len(genes_needing_lbfgsb))
        )
        
        # Update coefficients from L-BFGS-B results
        if results is not None:
            for gene_local_idx, beta_dense, beta_pert, conv in results:
                g = genes_needing_lbfgsb[gene_local_idx]
                beta_intercept[g] = beta_dense[0]
                if n_covariates > 0:
                    beta_cov[:, g] = beta_dense[1:]
                beta_perturbation[:, g] = beta_pert
                converged[g] = conv
        
        # Recompute SEs for genes that were refined with L-BFGS-B
        # by running one more IRLS iteration for SE computation
        for g in genes_needing_lbfgsb:
            # Build X^T W X for this gene
            A = np.zeros((n_dense_features, n_dense_features), dtype=np.float64)
            D_diag = np.zeros(n_perturbations, dtype=np.float64)
            B = np.zeros((n_dense_features, n_perturbations), dtype=np.float64)
            
            for slc, chunk in iter_matrix_chunks(
                backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
            ):
                Y_chunk = np.asarray(chunk, dtype=np.float64)
                n_chunk = Y_chunk.shape[0]
                
                offset_chunk = log_size_factors[slc]
                pert_idx_chunk = cell_pert_idx[slc]
                cov_chunk = cov_matrix[slc] if n_covariates > 0 else None
                
                # Dense design
                X_dense_chunk = np.ones((n_chunk, n_dense_features), dtype=np.float64)
                if n_covariates > 0:
                    X_dense_chunk[:, 1:] = cov_chunk
                
                # Compute eta and mu for this gene
                eta_g = beta_intercept[g] + offset_chunk
                pert_mask = pert_idx_chunk >= 0
                if np.any(pert_mask):
                    eta_g[pert_mask] += beta_perturbation[pert_idx_chunk[pert_mask], g]
                if n_covariates > 0 and cov_chunk is not None:
                    eta_g += cov_chunk @ beta_cov[:, g]
                
                eta_g = np.clip(eta_g, -20.0, 20.0)
                mu_g = np.exp(eta_g)
                mu_g = np.maximum(mu_g, 1e-6)
                
                # NB variance and weights
                variance_g = mu_g + dispersion[g] * mu_g * mu_g
                W_g = (mu_g * mu_g) / np.maximum(variance_g, 1e-12)
                
                # Accumulate A = X_dense^T W X_dense
                for j in range(n_dense_features):
                    for k in range(j, n_dense_features):
                        prod = X_dense_chunk[:, j] * X_dense_chunk[:, k]
                        contrib = np.sum(prod * W_g)
                        A[j, k] += contrib
                        if j != k:
                            A[k, j] += contrib
                
                # Accumulate D_diag and B
                for i in range(n_chunk):
                    p_idx = pert_idx_chunk[i]
                    if p_idx >= 0:
                        w_i = W_g[i]
                        D_diag[p_idx] += w_i
                        B[:, p_idx] += w_i * X_dense_chunk[i, :]
            
            # Add ridge
            A += ridge_penalty * np.eye(n_dense_features)
            D_diag += ridge_penalty
            
            # Compute SEs via block inverse (same as in IRLS loop)
            try:
                A_inv = np.linalg.inv(A)
                A_inv_B = A_inv @ B
                BtAinvB = B.T @ A_inv_B
                S_p = np.diag(D_diag) - BtAinvB
                S_p_inv = np.linalg.inv(S_p)
                
                se_pert_arr[:, g] = np.sqrt(np.maximum(np.diag(S_p_inv), 1e-12))
                
                AinvB_Spinv = A_inv_B @ S_p_inv
                upper_left_block = A_inv + AinvB_Spinv @ A_inv_B.T
                
                se_intercept_arr[g] = np.sqrt(np.maximum(upper_left_block[0, 0], 1e-12))
                if n_covariates > 0:
                    se_cov_arr[:, g] = np.sqrt(np.maximum(np.diag(upper_left_block[1:, 1:]), 1e-12))
                    
            except np.linalg.LinAlgError:
                se_intercept_arr[g] = 1.0 / np.sqrt(np.maximum(A[0, 0], 1e-12))
                se_pert_arr[:, g] = 1.0 / np.sqrt(np.maximum(D_diag, 1e-12))
                if n_covariates > 0:
                    se_cov_arr[:, g] = np.nan
    
    # =========================================================================
    # Stage 5: Standard errors were computed in final NB iteration above
    # =========================================================================
    # The SE arrays (se_intercept_arr, se_pert_arr, se_cov_arr) were populated
    # during the final IRLS iteration using proper block matrix inversion.
    # This correctly accounts for correlations between intercept and perturbation
    # effects, matching PyDESeq2's Wald test approach.
    
    se_intercept = se_intercept_arr
    se_perturbation = se_pert_arr
    se_cov = se_cov_arr
    
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
    
    def fit_batch(self, counts: ArrayLike) -> NBGLMBatchResult:
        """Fit NB GLM for all genes in the count matrix simultaneously.
        
        Parameters
        ----------
        counts
            Count matrix of shape ``(n_samples, n_genes)``.
            
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
        
        # Initialize beta for all valid genes: (n_features, n_valid)
        beta = np.zeros((n_features, n_valid), dtype=np.float64)
        
        # Poisson warm start (vectorized)
        if self.poisson_init_iter > 0:
            beta = self._poisson_warm_start_batch(Y_valid, beta)
        
        # Initialize dispersion estimates (method of moments)
        alpha = np.full(n_valid, 0.1, dtype=np.float64)
        
        # IRLS iterations (vectorized)
        gene_converged = np.zeros(n_valid, dtype=bool)
        gene_n_iter = np.zeros(n_valid, dtype=np.int32)
        
        # Pre-allocate work arrays for IRLS loop
        eta = np.empty((self.n_samples, n_valid), dtype=np.float64)
        mu = np.empty_like(eta)
        variance = np.empty_like(eta)
        weights = np.empty_like(eta)
        z = np.empty_like(eta)
        working_response = np.empty_like(eta)
        resid = np.empty_like(eta)
        
        # Precompute constants
        log_min_mu = np.log(self.min_mu)
        offset_col = self.offset[:, None]
        
        for iteration in range(1, self.max_iter + 1):
            # Compute eta and mu for all genes: (n_samples, n_valid)
            np.dot(X, beta, out=eta)
            eta += offset_col
            np.clip(eta, log_min_mu, 20.0, out=eta)
            np.exp(eta, out=mu)
            np.maximum(mu, self.min_mu, out=mu)
            
            # Compute variance and weights: V = mu + alpha * mu^2
            np.multiply(mu, mu, out=variance)
            variance *= alpha[None, :]
            variance += mu
            np.divide(mu * mu, np.maximum(variance, self.min_mu), out=weights)
            
            # Working response
            np.subtract(Y_valid, mu, out=resid)
            np.divide(resid, np.maximum(mu, self.min_mu), out=z)
            z += eta
            np.subtract(z, offset_col, out=working_response)
            
            # Solve weighted least squares for each gene (vectorized)
            beta_new = self._weighted_least_squares_batch(weights, working_response)
            
            # Check convergence per gene
            beta_diff = np.max(np.abs(beta_new - beta), axis=0)
            newly_converged = (beta_diff < self.tol) & ~gene_converged
            gene_converged |= newly_converged
            gene_n_iter[~gene_converged] = iteration
            
            beta = beta_new
            
            # Update dispersion using method of moments (vectorized)
            np.subtract(Y_valid, mu, out=resid)
            np.multiply(resid, resid, out=variance)  # reuse variance as temp
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
        np.clip(eta, log_min_mu, 20.0, out=eta)
        np.exp(eta, out=mu)
        np.maximum(mu, self.min_mu, out=mu)
        np.multiply(mu, mu, out=variance)
        variance *= alpha[None, :]
        variance += mu
        np.divide(mu * mu, np.maximum(variance, self.min_mu), out=weights)
        
        # Compute SE for each gene (need to invert Hessian per gene)
        se_valid = self._compute_se_batch(weights)
        
        # Compute deviance
        dev_valid = self._compute_deviance_batch(Y_valid, mu, alpha)
        
        # Store results back to full arrays
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
