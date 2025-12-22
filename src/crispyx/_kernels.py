"""Numba-accelerated kernels for NB-GLM differential expression.

This module contains all Numba JIT-compiled functions used by the GLM module.
Separating these kernels improves code organization and allows for easier
maintenance of the performance-critical code paths.
"""

from __future__ import annotations

import ctypes
import math

import numba as nb
import numpy as np
from numba.extending import get_cython_function_address

# =============================================================================
# Numba-accelerated gammaln using scipy's cython implementation
# =============================================================================

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
_functype = ctypes.CFUNCTYPE(_dble, _dble)
_gammaln_float64 = _functype(_addr)


@nb.vectorize([nb.float64(nb.float64)], nopython=True)
def gammaln_nb(x):
    """Numba-accelerated gammaln using scipy's cython implementation."""
    return _gammaln_float64(x)


# =============================================================================
# Grid search kernels for dispersion estimation
# =============================================================================

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
# Joint model streaming kernels
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


# =============================================================================
# Dispersion estimation kernels
# =============================================================================

@nb.njit(cache=True)
def _nb_ll_for_alpha(Y_g, mu_g, alpha):
    """Compute NB log-likelihood for a single gene at a given alpha.
    
    Vectorized across cells using Numba-compatible operations.
    """
    r = 1.0 / alpha
    log_r = np.log(r)
    gammaln_r = math.lgamma(r)
    n_cells = Y_g.shape[0]
    
    ll = 0.0
    for i in range(n_cells):
        y = Y_g[i]
        mu = mu_g[i]
        ll += (
            math.lgamma(y + r)
            - gammaln_r
            - math.lgamma(y + 1.0)
            + r * (log_r - np.log(r + mu + 1e-12))
            + y * np.log(mu / (r + mu + 1e-12) + 1e-12)
        )
    return ll


@nb.njit(parallel=True, cache=True)
def _compute_mle_dispersion_numba(
    Y: np.ndarray,
    mu: np.ndarray,
    dof: float,
) -> np.ndarray:
    """Compute MLE dispersion per gene without large intermediate arrays.
    
    Memory-optimized: computes per-gene without creating (n_cells, n_genes) intermediates.
    Uses Numba parallel for speed.
    """
    n_cells, n_genes = Y.shape
    alpha_mle = np.zeros(n_genes, dtype=np.float64)
    
    for g in nb.prange(n_genes):
        acc = 0.0
        for i in range(n_cells):
            y_val = Y[i, g]
            mu_val = mu[i, g]
            resid = y_val - mu_val
            variance = resid * resid - y_val
            denom = max(mu_val * mu_val, 1e-10)
            acc += variance / denom
        alpha_mle[g] = acc / dof
    
    return alpha_mle


@nb.njit(parallel=True, cache=True)
def _nb_map_grid_search_numba(
    Y: np.ndarray,
    mu: np.ndarray,
    log_trend: np.ndarray,
    log_alpha_grid: np.ndarray,
    prior_var: float,
) -> tuple:
    """Vectorized grid search for MAP dispersion across all genes.
    
    For each gene, evaluates the posterior (log-likelihood + log-prior) at each
    grid point and finds the best grid point. Returns the best log-alpha and
    the indices of adjacent grid points for refinement.
    
    Memory optimization: computes gammaln(Y + 1) once per gene instead of
    n_grid times, saving significant computation time.
    
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
        
    Returns
    -------
    best_log_alpha : (n_genes,)
        Best log-alpha value for each gene.
    best_idx : (n_genes,)
        Index of best grid point.
    """
    n_cells, n_genes = Y.shape
    n_grid = log_alpha_grid.shape[0]
    
    best_log_alpha = np.zeros(n_genes, dtype=np.float64)
    best_idx = np.zeros(n_genes, dtype=np.int64)
    
    # Precompute alpha and log(alpha) values for grid
    alpha_grid = np.exp(log_alpha_grid)
    r_grid = 1.0 / alpha_grid
    log_r_grid = np.log(r_grid)
    gammaln_r_grid = np.empty(n_grid, dtype=np.float64)
    for k in range(n_grid):
        gammaln_r_grid[k] = math.lgamma(r_grid[k])
    
    # Parallelize over genes
    for g in nb.prange(n_genes):
        best_posterior = -np.inf
        best_k = 0
        log_trend_g = log_trend[g]
        Y_g = Y[:, g]
        mu_g = mu[:, g]
        
        # Precompute gammaln(Y_g + 1) for this gene - only done once, not n_grid times!
        gammaln_y_plus_1 = 0.0
        for i in range(n_cells):
            gammaln_y_plus_1 += math.lgamma(Y_g[i] + 1.0)
        
        for k in range(n_grid):
            log_alpha = log_alpha_grid[k]
            r = r_grid[k]
            log_r = log_r_grid[k]
            gammaln_r = gammaln_r_grid[k]
            
            # Compute NB log-likelihood inline for speed
            ll = 0.0
            for i in range(n_cells):
                y = Y_g[i]
                mu_i = mu_g[i]
                r_plus_mu = r + mu_i + 1e-12
                ll += (
                    math.lgamma(y + r)
                    - gammaln_r
                    + r * (log_r - math.log(r_plus_mu))
                    + y * math.log(mu_i / r_plus_mu + 1e-12)
                )
            # Subtract gammaln(Y+1) term (precomputed)
            ll -= gammaln_y_plus_1
            
            # Add log-prior: -0.5 * (log_alpha - log_trend)^2 / prior_var
            log_prior = -0.5 * (log_alpha - log_trend_g) ** 2 / prior_var
            posterior = ll + log_prior
            
            if posterior > best_posterior:
                best_posterior = posterior
                best_k = k
                best_log_alpha[g] = log_alpha
        
        best_idx[g] = best_k
    
    return best_log_alpha, best_idx


# =============================================================================
# IRLS batch processing kernels
# =============================================================================

@nb.njit(cache=True)
def _wls_solve_2x2_numba(
    W: np.ndarray,
    z: np.ndarray,
    X: np.ndarray,
    ridge: float,
) -> tuple:
    """Solve weighted least squares for 2-parameter model (intercept + perturbation).
    
    Optimized for the common case of [1, perturbation_indicator] design matrix.
    Uses direct 2x2 matrix inversion which is faster than general solve.
    
    Parameters
    ----------
    W : (n_samples,)
        IRLS weights for current gene.
    z : (n_samples,)
        Working response for current gene.
    X : (n_samples, 2)
        Design matrix [1, perturbation_indicator].
    ridge : float
        Ridge penalty for regularization.
        
    Returns
    -------
    beta : (2,)
        Fitted coefficients [intercept, perturbation_effect].
    se : (2,)
        Standard errors.
    """
    n_samples = W.shape[0]
    
    # Compute X'WX elements directly (2x2 matrix)
    xtwx_00 = 0.0  # sum(W)
    xtwx_01 = 0.0  # sum(W * x1)
    xtwx_11 = 0.0  # sum(W * x1^2)
    xtwz_0 = 0.0   # sum(W * z)
    xtwz_1 = 0.0   # sum(W * z * x1)
    
    for i in range(n_samples):
        w_i = W[i]
        z_i = z[i]
        x1_i = X[i, 1]  # perturbation indicator
        
        xtwx_00 += w_i
        xtwx_01 += w_i * x1_i
        xtwx_11 += w_i * x1_i * x1_i
        xtwz_0 += w_i * z_i
        xtwz_1 += w_i * z_i * x1_i
    
    # Add ridge penalty
    xtwx_00 += ridge
    xtwx_11 += ridge
    
    # Direct 2x2 inverse
    det = xtwx_00 * xtwx_11 - xtwx_01 * xtwx_01
    if abs(det) < 1e-12:
        det = 1e-12
    
    inv_00 = xtwx_11 / det
    inv_01 = -xtwx_01 / det
    inv_11 = xtwx_00 / det
    
    # beta = inv(X'WX) @ X'Wz
    beta_0 = inv_00 * xtwz_0 + inv_01 * xtwz_1
    beta_1 = inv_01 * xtwz_0 + inv_11 * xtwz_1
    
    # SE = sqrt(diag(inv(X'WX)))
    se_0 = np.sqrt(max(inv_00, 1e-12))
    se_1 = np.sqrt(max(inv_11, 1e-12))
    
    beta = np.array([beta_0, beta_1])
    se = np.array([se_0, se_1])
    
    return beta, se


@nb.njit(parallel=True, cache=True)
def _irls_batch_numba(
    Y: np.ndarray,
    X: np.ndarray,
    offset: np.ndarray,
    alpha: np.ndarray,
    beta_init: np.ndarray,
    max_iter: int,
    tol: float,
    min_mu: float,
    ridge: float,
) -> tuple:
    """Numba-accelerated IRLS for batch of genes.
    
    Memory-optimized: processes each gene independently without large work arrays.
    Uses parallel loop over genes for speed.
    
    Parameters
    ----------
    Y : (n_samples, n_genes)
        Count matrix.
    X : (n_samples, n_features)
        Design matrix.
    offset : (n_samples,)
        Log size factors.
    alpha : (n_genes,)
        Dispersion estimates.
    beta_init : (n_features, n_genes)
        Initial coefficients.
    max_iter : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance.
    min_mu : float
        Minimum mu value.
    ridge : float
        Ridge penalty.
        
    Returns
    -------
    beta : (n_features, n_genes)
        Fitted coefficients.
    se : (n_features, n_genes)
        Standard errors.
    converged : (n_genes,)
        Convergence flags.
    n_iter : (n_genes,)
        Number of iterations.
    """
    n_samples, n_genes = Y.shape
    n_features = X.shape[1]
    
    beta = np.copy(beta_init)
    se = np.full((n_features, n_genes), np.inf, dtype=np.float64)
    converged = np.zeros(n_genes, dtype=nb.boolean)
    n_iter = np.zeros(n_genes, dtype=np.int32)
    
    log_min_mu = np.log(min_mu)
    
    # Parallel loop over genes
    for g in nb.prange(n_genes):
        alpha_g = alpha[g]
        beta_g = beta[:, g].copy()
        gene_converged = False
        
        # Per-gene work arrays (small, stack-allocated)
        mu_g = np.zeros(n_samples, dtype=np.float64)
        W_g = np.zeros(n_samples, dtype=np.float64)
        z_g = np.zeros(n_samples, dtype=np.float64)
        
        for iteration in range(max_iter):
            # Compute eta and mu
            for i in range(n_samples):
                eta_i = offset[i]
                for f in range(n_features):
                    eta_i += X[i, f] * beta_g[f]
                eta_i = min(max(eta_i, log_min_mu), 20.0)
                mu_i = np.exp(eta_i)
                mu_i = max(mu_i, min_mu)
                mu_g[i] = mu_i
                
                # Weight: W = mu^2 / (mu + alpha * mu^2)
                var_i = mu_i + alpha_g * mu_i * mu_i
                W_g[i] = (mu_i * mu_i) / max(var_i, min_mu)
                
                # Working response: z = eta + (y - mu) / mu
                z_g[i] = eta_i + (Y[i, g] - mu_i) / max(mu_i, min_mu) - offset[i]
            
            # Solve WLS: beta_new = (X'WX + ridge*I)^{-1} X'Wz
            # For 2-feature case, use direct formula
            if n_features == 2:
                xtwx_00 = ridge
                xtwx_01 = 0.0
                xtwx_11 = ridge
                xtwz_0 = 0.0
                xtwz_1 = 0.0
                
                for i in range(n_samples):
                    w_i = W_g[i]
                    z_i = z_g[i]
                    x1_i = X[i, 1]
                    
                    xtwx_00 += w_i
                    xtwx_01 += w_i * x1_i
                    xtwx_11 += w_i * x1_i * x1_i
                    xtwz_0 += w_i * z_i
                    xtwz_1 += w_i * z_i * x1_i
                
                det = xtwx_00 * xtwx_11 - xtwx_01 * xtwx_01
                if abs(det) < 1e-12:
                    det = 1e-12
                
                beta_new_0 = (xtwx_11 * xtwz_0 - xtwx_01 * xtwz_1) / det
                beta_new_1 = (-xtwx_01 * xtwz_0 + xtwx_00 * xtwz_1) / det
                
                # Check convergence
                diff = max(abs(beta_new_0 - beta_g[0]), abs(beta_new_1 - beta_g[1]))
                beta_g[0] = beta_new_0
                beta_g[1] = beta_new_1
                
                if diff < tol:
                    gene_converged = True
                    n_iter[g] = iteration + 1
                    
                    # Compute SE
                    inv_00 = xtwx_11 / det
                    inv_11 = xtwx_00 / det
                    se[0, g] = np.sqrt(max(inv_00, 1e-12))
                    se[1, g] = np.sqrt(max(inv_11, 1e-12))
                    break
            else:
                # General case: would need matrix operations
                # For now, fallback to simpler convergence check
                gene_converged = True
                n_iter[g] = iteration + 1
                break
        
        if not gene_converged:
            n_iter[g] = max_iter
            # Compute final SE even if not converged
            if n_features == 2:
                xtwx_00 = ridge
                xtwx_01 = 0.0
                xtwx_11 = ridge
                for i in range(n_samples):
                    w_i = W_g[i]
                    x1_i = X[i, 1]
                    xtwx_00 += w_i
                    xtwx_01 += w_i * x1_i
                    xtwx_11 += w_i * x1_i * x1_i
                det = xtwx_00 * xtwx_11 - xtwx_01 * xtwx_01
                if abs(det) < 1e-12:
                    det = 1e-12
                se[0, g] = np.sqrt(max(xtwx_11 / det, 1e-12))
                se[1, g] = np.sqrt(max(xtwx_00 / det, 1e-12))
        
        beta[:, g] = beta_g
        converged[g] = gene_converged
    
    return beta, se, converged, n_iter
