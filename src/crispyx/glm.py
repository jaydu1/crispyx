"""Generalized linear models utilities for differential expression."""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Sequence, Literal, Tuple

import numba as nb
import numpy as np
import scipy.sparse as sp
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
        gammaln_r = gammaln_nb(r)
        
        for g in range(n_genes):
            ll = 0.0
            for i in range(n_samples):
                y_ig = Y[i, g]
                mu_ig = mu[i, g]
                # gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                # + r * log(r / (r + mu)) + y * log(mu / (r + mu))
                ll += (
                    gammaln_nb(y_ig + r)
                    - gammaln_r
                    - gammaln_Y_plus_1[i, g]
                    + r * (log_r - np.log(r + mu_ig + 1e-12))
                    + y_ig * np.log(mu_ig / (r + mu_ig + 1e-12) + 1e-12)
                )
            ll_grid[a_idx, g] = ll
    
    return ll_grid


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
) -> JointModelResult:
    """Estimate joint model with all perturbations using streaming.
    
    This function fits a full negative binomial GLM with design matrix:
        [intercept, perturbation_1, ..., perturbation_n, covariates]
    
    where control is the reference level (intercept represents control baseline).
    This matches PyDESeq2's approach of fitting all conditions simultaneously.
    
    The implementation uses sparse perturbation indicators and block-diagonal
    structure of X^T W X for memory efficiency:
    - The perturbation×perturbation block is diagonal (each cell belongs to one group)
    - Only the intercept/covariate rows are dense
    - SEs are computed using block inversion with Schur complement
    
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
    
    # =========================================================================
    # Stage 1: Poisson IRLS to get initial estimates
    # =========================================================================
    # We accumulate X^T W X in block form:
    # - dense_block: (n_dense_features, n_dense_features) for intercept+covariates
    # - pert_diag: (n_perturbations,) diagonal for perturbation×perturbation
    # - cross_block: (n_dense_features, n_perturbations) for cross-terms
    # And X^T W z similarly
    
    for iteration in range(poisson_iter):
        # Per-gene accumulators for X^T W X blocks
        # dense_xtwx[g, :, :] is (n_dense_features, n_dense_features) for gene g
        # But storing this for all genes is expensive, so we accumulate and solve per-gene
        # For Schur complement, we need:
        # - A[g]: dense_xtwx for gene g
        # - D[g]: perturbation diagonal for gene g (n_perturbations,)
        # - B[g]: cross-terms for gene g (n_dense_features, n_perturbations)
        
        # We'll accumulate these per gene
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
            
            # Accumulate per-gene blocks
            # For each gene g: dense_xtwx[g] += X_dense.T @ diag(W[:, g]) @ X_dense
            for g in range(n_genes):
                w_g = W[:, g]  # (n_chunk,)
                wz_g = Wz[:, g]  # (n_chunk,)
                
                # Dense block: X_dense.T @ diag(w_g) @ X_dense
                XtW = X_dense.T * w_g[None, :]  # (n_dense, n_chunk)
                dense_xtwx_all[g] += XtW @ X_dense
                dense_xtwz_all[g] += XtW @ np.ones(n_chunk)  # Actually: (X_dense.T * w_g) @ 1 for intercept
                dense_xtwz_all[g] += X_dense.T @ (w_g * z[:, g])
                
            # Correction: accumulate X^T W z properly
            # dense_xtwz_all[g] should be X_dense.T @ Wz[:, g]
        
        # Recalculate with correct accumulation
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
            
            eta = beta_intercept[None, :] + offset_chunk[:, None]
            pert_mask = pert_idx_chunk >= 0
            if np.any(pert_mask):
                eta[pert_mask] += beta_perturbation[pert_idx_chunk[pert_mask], :]
            if n_covariates > 0 and cov_chunk is not None:
                eta += cov_chunk @ beta_cov
            
            eta = np.clip(eta, -20.0, 20.0)
            mu = np.exp(eta)
            mu = np.maximum(mu, 1e-6)
            
            W = mu
            z = eta - offset_chunk[:, None] + (Y_chunk - mu) / np.maximum(mu, 1e-6)
            Wz = W * z
            
            X_dense = np.ones((n_chunk, n_dense_features), dtype=np.float64)
            if n_covariates > 0:
                X_dense[:, 1:] = cov_chunk
            
            # Accumulate dense blocks for all genes at once (vectorized)
            # dense_xtwx[g] += sum_i w[i,g] * X_dense[i,:].T @ X_dense[i,:]
            # This is equivalent to: for each feature pair (j,k): sum_i w[i,g] * X[i,j] * X[i,k]
            for j in range(n_dense_features):
                for k in range(j, n_dense_features):
                    # (n_chunk,) * (n_chunk,) -> (n_chunk,), then sum weighted by W
                    prod = X_dense[:, j] * X_dense[:, k]  # (n_chunk,)
                    # W[:, :] * prod[:, None] -> (n_chunk, n_genes), sum over axis 0
                    contrib = (W.T @ prod)  # (n_genes,)
                    dense_xtwx_all[:, j, k] += contrib
                    if j != k:
                        dense_xtwx_all[:, k, j] += contrib
            
            # dense_xtwz: X_dense.T @ Wz -> for gene g: sum_i Wz[i,g] * X_dense[i,:]
            dense_xtwz_all += (Wz.T @ X_dense)  # (n_genes, n_dense)
            
            # Perturbation contributions
            # cross_xtwx[g, j, p] = sum_i (w[i,g] * X_dense[i,j] * indicator[i in pert p])
            for i in range(n_chunk):
                p_idx = pert_idx_chunk[i]
                if p_idx >= 0:
                    w_i = W[i, :]  # (n_genes,)
                    wz_i = Wz[i, :]  # (n_genes,)
                    pert_xtwx_diag_all[:, p_idx] += w_i
                    # cross_xtwx_all[g, j, p_idx] += w[i,g] * X_dense[i, j]
                    # This is: (n_genes,) outer (n_dense,) -> (n_genes, n_dense)
                    cross_xtwx_all[:, :, p_idx] += np.outer(w_i, X_dense[i, :])
                    pert_xtwz_all[:, p_idx] += wz_i
        
        # Transpose dense_xtwz to match expected shape
        dense_xtwz_all = dense_xtwz_all  # Already (n_genes, n_dense)
        
        # Solve per-gene using Schur complement
        ridge_dense = ridge_penalty * np.eye(n_dense_features)
        ridge_pert = ridge_penalty
        
        beta_intercept_new = np.zeros(n_genes, dtype=np.float64)
        beta_cov_new = np.zeros((n_covariates, n_genes), dtype=np.float64)
        beta_pert_new = np.zeros((n_perturbations, n_genes), dtype=np.float64)
        
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
            # This prevents extreme values when there's no data to estimate
            x2 = np.where(pert_has_data[:, g], x2, 0.0)
            
            # Clip remaining coefficients to prevent extreme values
            x2 = np.clip(x2, -10.0, 10.0)
            
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
    # =========================================================================
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
    
    n_features_total = 1 + n_perturbations + n_covariates
    dof = max(n_total - n_features_total, 1)
    dispersion_raw = np.clip(numerator_sum / dof, 1e-8, 1e6)
    dispersion_raw = np.where(np.isfinite(dispersion_raw), dispersion_raw, 0.1)
    
    # Compute mean expression for trend fitting
    mean_expr = np.zeros(n_genes, dtype=np.float64)
    for slc, chunk in iter_matrix_chunks(
        backed_adata, axis=0, chunk_size=chunk_size, convert_to_dense=True
    ):
        Y_chunk = np.asarray(chunk, dtype=np.float64)
        mean_expr += Y_chunk.sum(axis=0)
    mean_expr /= n_total
    
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
    
    # Per-gene XtWX accumulators for SE computation (final iteration)
    pert_xtwx_per_gene_final = np.zeros((n_perturbations, n_genes), dtype=np.float64)
    intercept_xtwx_per_gene_final = np.zeros(n_genes, dtype=np.float64)
    
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
            
            # Perturbation contributions
            for i in range(n_chunk):
                p_idx = pert_idx_chunk[i]
                if p_idx >= 0:
                    w_i = W[i, :]  # (n_genes,)
                    wz_i = Wz[i, :]
                    pert_xtwx_diag_all[:, p_idx] += w_i
                    cross_xtwx_all[:, :, p_idx] += np.outer(w_i, X_dense[i, :])
                    pert_xtwz_all[:, p_idx] += wz_i
        
        # Store XtWX for SE computation (on last iteration)
        if iteration == nb_iter - 1 or iteration >= 0:  # Always store for now
            intercept_xtwx_per_gene_final = dense_xtwx_all[:, 0, 0].copy()
            pert_xtwx_per_gene_final = pert_xtwx_diag_all.T.copy()
        
        # Solve per-gene using Schur complement
        ridge_dense = ridge_penalty * np.eye(n_dense_features)
        ridge_pert = ridge_penalty
        
        beta_intercept_new = np.zeros(n_genes, dtype=np.float64)
        beta_cov_new = np.zeros((n_covariates, n_genes), dtype=np.float64)
        beta_pert_new = np.zeros((n_perturbations, n_genes), dtype=np.float64)
        
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
            x2 = np.clip(x2, -10.0, 10.0)
            
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
        
        n_iter_arr[:] = iteration + 1
        
        if max_diff < tol:
            converged[:] = True
            break
    
    # =========================================================================
    # Stage 4: Compute standard errors using per-gene XtWX
    # =========================================================================
    # For perturbations, SE = 1 / sqrt(XtWX_ii) since the block is diagonal
    # For intercept, we use the accumulated per-gene values
    
    se_intercept = 1.0 / np.sqrt(np.maximum(intercept_xtwx_per_gene_final, 1e-12))
    se_perturbation = 1.0 / np.sqrt(np.maximum(pert_xtwx_per_gene_final, 1e-12))
    se_cov = np.zeros((n_covariates, n_genes), dtype=np.float64)
    
    # For covariates, we'd need the full inverse - approximate with diagonal
    # This is a simplification; full SE would require storing more info
    # For now, set to NaN to indicate not computed
    if n_covariates > 0:
        se_cov[:] = np.nan
    
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
