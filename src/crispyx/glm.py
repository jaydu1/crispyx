"""Generalized linear models utilities for differential expression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.linalg import cho_factor, cho_solve


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
        ridge_penalty: float = 1e-8,
        min_mu: float = 1e-8,
        min_total_count: float = 1.0,
        compute_cooks: bool = False,
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
        beta = np.zeros(self.n_features, dtype=np.float64)
        if self.poisson_init_iter:
            beta = self._poisson_warm_start(y, beta.copy())
        alpha = self.dispersion if self.dispersion is not None else 0.0
        converged = False
        deviance = np.nan
        cov_beta = np.eye(self.n_features, dtype=np.float64)
        mu = np.maximum(np.full_like(y, y.mean()), self.min_mu)
        weights = np.ones_like(mu)
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
    means: ArrayLike, dispersions: ArrayLike, *, min_mean: float = 1e-8
) -> np.ndarray:
    """Fit a smooth log-quadratic trend of dispersion versus mean expression."""

    means_arr = np.asarray(means, dtype=np.float64)
    disp_arr = np.asarray(dispersions, dtype=np.float64)
    valid = (
        np.isfinite(means_arr)
        & np.isfinite(disp_arr)
        & (means_arr > min_mean)
        & (disp_arr > 0)
    )
    if valid.sum() < 3:
        baseline = np.nanmedian(disp_arr[valid]) if np.any(valid) else np.nan
        return np.full_like(means_arr, baseline, dtype=np.float64)
    x = np.log(np.clip(means_arr[valid], min_mean, None))
    y = np.log(np.clip(disp_arr[valid], min_mean, None))
    coeffs = np.polyfit(x, y, deg=2)
    trend = np.exp(np.polyval(coeffs, np.log(np.clip(means_arr, min_mean, None))))
    return trend


def shrink_dispersions(raw: ArrayLike, trend: ArrayLike, prior_weight: float | None = None) -> np.ndarray:
    """Empirically shrink dispersions toward a fitted mean-dispersion trend."""

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
    if prior_weight is None:
        prior_var = np.nanmedian((log_raw - log_trend) ** 2)
        prior_weight = 1.0 / np.maximum(prior_var, 1e-8)
    log_post = (log_raw + prior_weight * log_trend) / (1.0 + prior_weight)
    shrunk[mask] = np.exp(log_post)
    return shrunk


def shrink_log_foldchange(
    coef: ArrayLike, se: ArrayLike, prior_var: float | None = None
) -> np.ndarray:
    """Apply simple empirical Bayes shrinkage to log-fold changes."""

    coef_arr = np.asarray(coef, dtype=np.float64)
    se_arr = np.asarray(se, dtype=np.float64)
    shrunk = np.array(coef_arr, copy=True)
    mask = np.isfinite(coef_arr) & np.isfinite(se_arr) & (se_arr > 0)
    if not np.any(mask):
        return shrunk
    if prior_var is None:
        prior_var = np.nanmedian(se_arr[mask] ** 2)
    prior_var = float(np.maximum(prior_var, 1e-12))
    shrink_factor = prior_var / (prior_var + se_arr[mask] ** 2)
    shrunk[mask] = coef_arr[mask] * shrink_factor
    return shrunk
