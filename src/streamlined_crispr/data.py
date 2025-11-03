"""Helpers for working with AnnData ``.h5ad`` files in a streaming friendly way."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Sequence

import h5py
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

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
        raw_names = adata.var_names
        logger.info(
            "No gene_name_column provided; using adata.var_names for gene identifiers."
        )
    else:
        if gene_name_column not in adata.var.columns:
            if gene_name_column == "gene_symbols":
                raw_names = adata.var_names
                logger.info(
                    "Column 'gene_symbols' not found in adata.var; "
                    "using adata.var_names for gene identifiers."
                )
            else:
                raise KeyError(
                    f"Gene name column '{gene_name_column}' was not found in adata.var. Available columns: {list(adata.var.columns)}"
                )
        else:
            raw_names = adata.var[gene_name_column]
    names = pd.Index(raw_names).astype(str)
    _validate_gene_symbols(names)
    return names


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


def resolve_control_label(
    labels: Sequence[str],
    control_label: str | None,
    *,
    verbose: bool = True,
) -> str:
    """Return an explicit control label, inferring one when necessary."""

    if control_label is not None:
        return str(control_label)

    index = pd.Index(labels).astype(str)
    if index.empty:
        raise ValueError(
            "Cannot infer control label because no perturbation labels were provided."
        )
    lower = index.str.lower()

    exact_terms = {"ctrl", "control", "nontarget", "non-target", "non_target"}
    substring_terms = ("ctrl", "control", "nontarget", "non-target", "non_target")

    def _select(predicate) -> str | None:
        for label, lowered in zip(index, lower):
            if predicate(lowered):
                return str(label)
        return None

    candidate = _select(lambda text: text in exact_terms)
    if candidate is None:
        candidate = _select(lambda text: any(term in text for term in substring_terms))
    if candidate is None:
        candidate = _select(lambda text: ("non" in text) and ("target" in text))

    if candidate is None:
        raise ValueError(
            "Unable to infer control label automatically. Please provide 'control_label' explicitly."
        )

    if verbose:
        logger.info("Inferred control label '%s' from perturbation labels.", candidate)
    return candidate


def preview_backed(
    path: str | Path,
    *,
    n_obs: int = 5,
    n_vars: int = 5,
) -> ad.AnnData:
    """Open an ``.h5ad`` file in backed mode, print a preview, and return the object."""

    adata_ro = read_backed(path)
    try:
        print(adata_ro)
        if n_obs > 0 and adata_ro.n_obs > 0:
            print("First obs rows:")
            print(adata_ro.obs.head(n_obs))
        if n_vars > 0 and adata_ro.n_vars > 0:
            print("First var rows:")
            print(adata_ro.var.head(n_vars))
    except Exception:
        adata_ro.file.close()
        raise
    return adata_ro


def iter_matrix_chunks(
    adata: ad.AnnData | ad._core.anndata.AnnDataMixin,
    *,
    axis: int = 0,
    chunk_size: int = 1024,
    convert_to_dense: bool = True,
) -> Iterator[tuple[slice, np.ndarray | sp.spmatrix]]:
    """Yield chunks of the expression matrix."""

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
        if convert_to_dense:
            block = _to_dense(block)
        yield slice(start, end), block


def _to_dense(matrix: np.ndarray) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def normalize_total_block(
    block: np.ndarray | sp.spmatrix,
    *,
    library_size: np.ndarray | None = None,
    target_sum: float = 1e4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a library-size normalised dense view of ``block``.

    Parameters
    ----------
    block:
        A slice of the expression matrix with shape ``(n_cells, n_genes)``.
    library_size:
        Optional precomputed library sizes for the cells in ``block``. When
        ``None`` the library size is computed from ``block`` directly.
    target_sum:
        Target total counts per cell after normalisation, matching the default
        used by :func:`scanpy.pp.normalize_total`.

    Returns
    -------
    tuple
        A tuple ``(normalised, library_size)`` where ``normalised`` is a dense
        ``float64`` array containing the normalised counts and ``library_size``
        contains the per-cell library sizes that were used.
    """

    dense = _to_dense(block).astype(np.float64, copy=True)
    if dense.ndim != 2:
        raise ValueError("block must be two-dimensional")

    if library_size is None:
        library_size = dense.sum(axis=1)
    else:
        library_size = np.asarray(library_size, dtype=np.float64)
        if library_size.shape[0] != dense.shape[0]:
            raise ValueError(
                "library_size length does not match the number of cells in block"
            )

    scale = np.divide(
        float(target_sum),
        library_size,
        out=np.zeros_like(library_size, dtype=np.float64),
        where=library_size > 0,
    )
    dense *= scale[:, None]
    return dense, library_size


def _ensure_csr(matrix: np.ndarray | sp.spmatrix, *, dtype: np.dtype | None = None) -> sp.csr_matrix:
    """Convert the provided matrix to CSR format."""

    if sp.isspmatrix_csr(matrix):
        csr = matrix
    elif sp.issparse(matrix):
        csr = matrix.tocsr()
    else:
        csr = sp.csr_matrix(np.asarray(matrix))
    if dtype is not None and csr.dtype != dtype:
        csr = csr.astype(dtype)
    return csr


def write_filtered_subset(
    source_path: str | Path,
    *,
    cell_mask: np.ndarray,
    gene_mask: np.ndarray,
    output_path: str | Path,
    chunk_size: int = 4096,
    var_assignments: dict[str, Sequence] | None = None,
) -> None:
    """Stream a filtered AnnData view to disk without materialising ``X``."""

    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backed = read_backed(source_path)
    try:
        obs = backed.obs.iloc[cell_mask].copy()
        var = backed.var.iloc[gene_mask].copy()
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)
        if var_assignments:
            for key, values in var_assignments.items():
                if len(values) != var.shape[0]:
                    raise ValueError(
                        f"Length mismatch for column '{key}': expected {var.shape[0]}, received {len(values)}"
                    )
                var[key] = np.asarray(values)
    finally:
        backed.file.close()

    n_obs = int(cell_mask.sum())
    n_vars = int(gene_mask.sum())

    gene_indices = np.flatnonzero(gene_mask)

    row_nnz = np.zeros(n_obs, dtype=np.int64)
    total_nnz = 0
    data_dtype: np.dtype | None = None

    backed = read_backed(source_path)
    try:
        row_offset = 0
        for slc, block in iter_matrix_chunks(backed, axis=0, chunk_size=chunk_size, convert_to_dense=False):
            local_mask = cell_mask[slc]
            if not np.any(local_mask):
                continue
            block = block[local_mask]
            if gene_indices.size:
                block = block[:, gene_indices]
            else:
                block = block[:, []]
            csr = _ensure_csr(block)
            counts = np.diff(csr.indptr)
            size = counts.size
            row_nnz[row_offset : row_offset + size] = counts
            total_nnz += int(csr.nnz)
            if data_dtype is None and csr.nnz:
                data_dtype = csr.data.dtype
            row_offset += size
    finally:
        backed.file.close()

    if data_dtype is None:
        data_dtype = np.float32

    placeholder = sp.csr_matrix((n_obs, n_vars), dtype=data_dtype)
    adata = ad.AnnData(placeholder, obs=obs, var=var)
    adata.write(output_path)

    if n_obs == 0 or n_vars == 0:
        with h5py.File(output_path, "r+") as dest:
            if "X" in dest:
                del dest["X"]
            grp = dest.create_group("X")
            grp.attrs["encoding-type"] = np.bytes_("csr_matrix")
            grp.attrs["encoding-version"] = np.bytes_("0.1.0")
            grp.create_dataset("data", shape=(0,), dtype=data_dtype)
            grp.create_dataset("indices", shape=(0,), dtype=np.int32)
            grp.create_dataset("indptr", data=np.zeros(n_obs + 1, dtype=np.int64))
            grp.attrs["shape"] = np.array([n_obs, n_vars], dtype=np.int64)
        return

    indptr = np.zeros(n_obs + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=indptr[1:])

    with h5py.File(output_path, "r+") as dest:
        if "X" in dest:
            del dest["X"]
        grp = dest.create_group("X")
        grp.attrs["encoding-type"] = np.bytes_("csr_matrix")
        grp.attrs["encoding-version"] = np.bytes_("0.1.0")
        data_ds = grp.create_dataset("data", shape=(total_nnz,), dtype=data_dtype, chunks=True)
        indices_ds = grp.create_dataset("indices", shape=(total_nnz,), dtype=np.int32, chunks=True)
        grp.create_dataset("indptr", data=indptr)
        grp.attrs["shape"] = np.array([n_obs, n_vars], dtype=np.int64)

        backed = read_backed(source_path)
        try:
            offset = 0
            for slc, block in iter_matrix_chunks(
                backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
            ):
                local_mask = cell_mask[slc]
                if not np.any(local_mask):
                    continue
                block = block[local_mask]
                if gene_indices.size:
                    block = block[:, gene_indices]
                else:
                    block = block[:, []]
                csr = _ensure_csr(block, dtype=data_dtype)
                nnz = int(csr.nnz)
                if nnz:
                    data_ds[offset : offset + nnz] = csr.data
                    indices_ds[offset : offset + nnz] = csr.indices.astype(np.int32, copy=False)
                    offset += nnz
        finally:
            backed.file.close()


