"""Helpers for working with AnnData ``.h5ad`` files in a streaming friendly way."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import anndata as ad
import numpy as np
import pandas as pd

ENSEMBL_PREFIXES = ("ENS", "FBgn", "YAL", "YBL", "YCL", "YDL", "YEL", "YFL", "YGL", "YHL", "YIL", "YJL", "YKL", "YLL", "YML", "YNL", "YOL", "YPL", "YQL", "YRL", "YSL", "YTL", "YUL", "YVL", "YWL", "YXL")


def read_backed(path: str | Path) -> ad.AnnData:
    """Open an ``.h5ad`` file in backed mode for low-memory access."""

    return ad.read_h5ad(str(path), backed="r")


def resolve_output_path(
    input_path: str | Path,
    *,
    suffix: str,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
) -> Path:
    """Construct an informative output path for an intermediate ``.h5ad`` file."""

    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir is not None else input_path.parent
    data_name = data_name or input_path.stem
    return output_dir / f"{data_name}_{suffix}.h5ad"


def ensure_gene_symbol_column(
    adata: ad.AnnData | ad._core.anndata.AnnDataMixin,
    gene_name_column: str | None,
) -> pd.Index:
    """Return a vector of gene symbols and verify they look like symbols, not Ensembl IDs."""

    if gene_name_column is None:
        names = adata.var_names
    else:
        if gene_name_column not in adata.var.columns:
            raise KeyError(
                f"Gene name column '{gene_name_column}' was not found in adata.var. Available columns: {list(adata.var.columns)}"
            )
        names = adata.var[gene_name_column].astype(str)
    _validate_gene_symbols(names)
    return pd.Index(names)


def _validate_gene_symbols(names: Sequence[str]) -> None:
    """Perform a basic sanity check that the provided gene identifiers look like symbols."""

    if len(names) == 0:
        raise ValueError("No gene names were provided.")
    names = pd.Index(names).astype(str)
    prefixes = names.str.upper().str.slice(0, 3)
    ensembl_like = prefixes.isin([p[:3] for p in ENSEMBL_PREFIXES]).sum()
    if ensembl_like > len(names) / 2:
        raise ValueError(
            "The majority of provided gene identifiers appear to be Ensembl-style IDs. "
            "Please supply a column containing gene symbols."
        )


def iter_matrix_chunks(
    adata: ad.AnnData | ad._core.anndata.AnnDataMixin,
    *,
    axis: int = 0,
    chunk_size: int = 1024,
) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield chunks of the expression matrix as dense arrays."""

    if axis not in (0, 1):
        raise ValueError("axis must be 0 (rows) or 1 (columns)")
    n_obs, n_vars = adata.n_obs, adata.n_vars
    length = n_obs if axis == 0 else n_vars
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        if axis == 0:
            block = adata.X[start:end]
        else:
            block = adata.X[:, start:end]
        block = _to_dense(block)
        yield slice(start, end), block


def _to_dense(matrix: np.ndarray) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)

