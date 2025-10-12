"""Subset of :mod:`scanpy.pp` required for the unit tests."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _library_size(matrix):
    if sp.issparse(matrix):
        # ``sum`` on sparse matrices returns a 2-D matrix. Convert to ``ndarray``.
        return np.asarray(matrix.sum(axis=1))
    return np.asarray(matrix.sum(axis=1, keepdims=True))


def normalize_total(adata, *, target_sum: float = 1e4) -> None:
    """Library-size normalisation matching :func:`scanpy.pp.normalize_total`."""

    matrix = adata.X
    library_size = _library_size(matrix)
    if library_size.ndim == 2:
        library_size = library_size.reshape(-1, 1)
    scale = np.divide(
        float(target_sum),
        library_size,
        out=np.zeros_like(library_size, dtype=float),
        where=library_size > 0,
    )

    if sp.issparse(matrix):
        # ``csr_matrix`` supports broadcasting when multiplying by a dense array
        # with shape ``(n, 1)``.
        adata.X = matrix.multiply(scale).tocsr()
    else:
        adata.X = np.asarray(matrix, dtype=float) * scale


def log1p(adata) -> None:
    """Apply ``np.log1p`` to the values of :attr:`adata.X`."""

    matrix = adata.X
    if sp.issparse(matrix):
        matrix = matrix.copy()
        matrix.data = np.log1p(matrix.data)
        adata.X = matrix
    else:
        adata.X = np.log1p(np.asarray(matrix, dtype=float))


__all__ = ["normalize_total", "log1p"]
