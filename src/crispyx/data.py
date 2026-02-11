"""Helpers for working with AnnData ``.h5ad`` files in a streaming friendly way."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

from numba import njit, prange

# Numba-accelerated helpers for dense→CSR conversion (60x faster than scipy)
@njit(parallel=True)
def _numba_count_row_nnz(dense: np.ndarray) -> np.ndarray:
    """Count non-zeros per row using parallel numba."""
    n_rows = dense.shape[0]
    row_nnz = np.zeros(n_rows, dtype=np.int64)
    for i in prange(n_rows):
        count = 0
        for j in range(dense.shape[1]):
            if dense[i, j] != 0:
                count += 1
        row_nnz[i] = count
    return row_nnz

@njit(parallel=True)
def _numba_extract_csr_data(
    dense: np.ndarray,
    indptr: np.ndarray,
    data: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Extract CSR data/indices in parallel from dense array."""
    n_rows = dense.shape[0]
    for i in prange(n_rows):
        pos = indptr[i]
        for j in range(dense.shape[1]):
            val = dense[i, j]
            if val != 0:
                data[pos] = val
                indices[pos] = j
                pos += 1

ENSEMBL_PREFIXES = ("ENS", "FBgn", "YAL", "YBL", "YCL", "YDL", "YEL", "YFL", "YGL", "YHL", "YIL", "YJL", "YKL", "YLL", "YML", "YNL", "YOL", "YPL", "YQL", "YRL", "YSL", "YTL", "YUL", "YVL", "YWL", "YXL")


def is_dense_storage(path: str | Path) -> bool:
    """Check if h5ad file stores X matrix as dense array.
    
    Parameters
    ----------
    path
        Path to h5ad file.
        
    Returns
    -------
    bool
        True if X is stored as dense array, False if sparse (CSR/CSC).
    """
    with h5py.File(path, 'r') as f:
        if 'X' not in f:
            return False
        x_obj = f['X']
        if isinstance(x_obj, h5py.Dataset):
            # Dense array stored directly as dataset
            return True
        elif isinstance(x_obj, h5py.Group):
            # Check encoding-type attribute
            encoding = x_obj.attrs.get('encoding-type', b'')
            if isinstance(encoding, bytes):
                encoding = encoding.decode('utf-8')
            return encoding == 'array'
        return False


def get_matrix_storage_format(path: str | Path) -> str:
    """Detect matrix storage format in h5ad file.
    
    Parameters
    ----------
    path
        Path to h5ad file.
        
    Returns
    -------
    str
        One of: 'csr', 'csc', 'dense'
    """
    with h5py.File(path, 'r') as f:
        if 'X' not in f:
            return 'dense'
        x_obj = f['X']
        if isinstance(x_obj, h5py.Dataset):
            # Dense array stored directly as dataset
            return 'dense'
        elif isinstance(x_obj, h5py.Group):
            # Check encoding-type attribute
            encoding = x_obj.attrs.get('encoding-type', b'')
            if isinstance(encoding, bytes):
                encoding = encoding.decode('utf-8')
            if encoding == 'csc_matrix':
                return 'csc'
            elif encoding == 'csr_matrix':
                return 'csr'
            elif encoding == 'array':
                return 'dense'
        return 'dense'


_MISSING = object()


class _LazyFrameAccessor:
    """Provide a read-friendly view over ``obs``/``var`` tables with ``.load``."""

    def __init__(self, parent: "AnnData", attr: str) -> None:
        self._parent = parent
        self._attr = attr
        self._cache: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if self._cache is None:
            loaded = getattr(self._parent.to_memory(), self._attr)
            if isinstance(loaded, pd.DataFrame):
                loaded = loaded.copy()
            else:
                loaded = pd.DataFrame(loaded)
            self._cache = loaded
        return self._cache

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.load().head(n)

    def __len__(self) -> int:
        return len(self.load())

    def __iter__(self):  # pragma: no cover - passthrough for convenience
        return iter(self.load())

    def __getitem__(self, item):  # pragma: no cover - passthrough for convenience
        return self.load().__getitem__(item)

    def __getattr__(self, name: str):  # pragma: no cover - passthrough for convenience
        return getattr(self.load(), name)

    def __repr__(self) -> str:  # pragma: no cover - display preview
        frame = self.head()
        if frame.empty:
            return f"<{self._attr}: empty DataFrame>"
        return f"<{self._attr} preview>\n{frame}"


def _preview_uns_value(value: Any, n: int = 5) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.head(n)
    if isinstance(value, np.ndarray):
        return value[:n]
    if isinstance(value, Mapping):
        preview: dict[Any, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= n:
                preview["…"] = "…"
                break
            preview[key] = _preview_uns_value(item, n)
        return preview
    if isinstance(value, (list, tuple)):
        return type(value)(value[:n])
    return value


class _LazyUnsEntry:
    """Deferred loader for a single ``uns`` key."""

    def __init__(self, parent: "AnnData", key: str) -> None:
        self._parent = parent
        self._key = key
        self._cache: Any = _MISSING

    def load(self) -> Any:
        if self._cache is _MISSING:
            self._cache = self._parent.to_memory().uns[self._key]
        return self._cache

    def preview(self, n: int = 5) -> Any:
        return _preview_uns_value(self.load(), n)

    def __getattr__(self, name: str):  # pragma: no cover - passthrough for convenience
        return getattr(self.load(), name)

    def __getitem__(self, item):  # pragma: no cover - passthrough for convenience
        return self.load()[item]

    def __repr__(self) -> str:  # pragma: no cover - display preview
        return repr(self.preview())


class _LazyUnsMapping(Mapping[str, _LazyUnsEntry]):
    """Mapping-style accessor exposing ``uns`` keys with lazy loading."""

    def __init__(self, parent: "AnnData") -> None:
        self._parent = parent
        self._cache: dict[str, _LazyUnsEntry] = {}

    def _keys(self) -> list[str]:
        try:
            return list(self._parent.backed.uns.keys())
        except AttributeError:
            return []

    def __getitem__(self, key: str) -> _LazyUnsEntry:
        keys = self._keys()
        if key not in keys:
            raise KeyError(key)
        if key not in self._cache:
            self._cache[key] = _LazyUnsEntry(self._parent, key)
        return self._cache[key]

    def __iter__(self):  # pragma: no cover - passthrough for convenience
        return iter(self._keys())

    def __len__(self) -> int:
        return len(self._keys())

    def keys(self):  # pragma: no cover - convenience mirror
        return self._keys()

    def items(self):  # pragma: no cover - convenience mirror
        return [(key, self[key]) for key in self._keys()]

    def __repr__(self) -> str:  # pragma: no cover - display preview
        keys = self._keys()
        if not keys:
            return "<uns: empty>"
        previews = {key: self[key].preview() for key in keys}
        return repr(previews)


class AnnData:
    """Thin wrapper around a backed :class:`anndata.AnnData` handle."""

    def __init__(self, path: str | Path, *, mode: str = "r") -> None:
        self.path = Path(path)
        self._mode = mode
        self._backed: ad.AnnData | None = None
        self._obs_view: _LazyFrameAccessor | None = None
        self._var_view: _LazyFrameAccessor | None = None
        self._uns_view: _LazyUnsMapping | None = None

    @property
    def filename(self) -> str:
        """Return the underlying filename for compatibility with Scanpy."""

        return str(self.path)

    def __fspath__(self) -> str:  # pragma: no cover - filesystem protocol
        return str(self.path)

    def __str__(self) -> str:  # pragma: no cover - helpful when printing paths
        return str(self.path)

    @property
    def backed(self) -> ad.AnnData:
        """Return the lazily opened backed AnnData handle."""

        if self._backed is None:
            self._backed = ad.read_h5ad(str(self.path), backed=self._mode)
        return self._backed

    def close(self) -> None:
        """Close the underlying file handle if it is open."""

        if self._backed is not None:
            try:
                self._backed.file.close()
            finally:
                self._backed = None
        self._obs_view = None
        self._var_view = None
        self._uns_view = None

    def to_memory(self) -> ad.AnnData:
        """Materialise the backed AnnData into memory."""

        return ad.read_h5ad(str(self.path))

    @property
    def obs(self) -> _LazyFrameAccessor:
        if self._obs_view is None:
            self._obs_view = _LazyFrameAccessor(self, "obs")
        return self._obs_view

    @property
    def var(self) -> _LazyFrameAccessor:
        if self._var_view is None:
            self._var_view = _LazyFrameAccessor(self, "var")
        return self._var_view

    @property
    def uns(self) -> _LazyUnsMapping:
        if self._uns_view is None:
            self._uns_view = _LazyUnsMapping(self)
        return self._uns_view

    def __enter__(self) -> "AnnData":
        self.backed  # ensure handle is opened
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __getattr__(self, name: str):  # pragma: no cover - delegation helper
        return getattr(self.backed, name)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"AnnData(path={self.path!s}, mode='{self._mode}')"

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass


def read_backed(path: str | Path) -> ad.AnnData:
    """Open an ``.h5ad`` file in backed mode for low-memory access."""

    return ad.read_h5ad(str(path), backed="r")


def resolve_data_path(
    data: str | Path | "AnnData" | ad.AnnData,
    *,
    require_exists: bool = True,
) -> Path:
    """Resolve the on-disk path for a backed AnnData object or path-like input.
    
    This utility supports flexible input types for crispyx functions, allowing
    users to pass either a file path or an AnnData object.
    
    Parameters
    ----------
    data
        One of:
        - A string or Path to an h5ad file
        - A crispyx.AnnData wrapper (has .path attribute)
        - A backed anndata.AnnData object (has .filename attribute)
    require_exists
        If True (default), verify the resolved path exists.
        
    Returns
    -------
    Path
        The resolved file path.
        
    Raises
    ------
    TypeError
        If data is an in-memory (non-backed) AnnData or unsupported type.
    FileNotFoundError
        If require_exists is True and the path does not exist.
        
    Examples
    --------
    >>> from crispyx.data import resolve_data_path
    >>> path = resolve_data_path("data/counts.h5ad")
    >>> path = resolve_data_path(adata_wrapper)
    >>> path = resolve_data_path(backed_adata)
    """
    if isinstance(data, (str, Path)):
        result = Path(data)
    elif isinstance(data, AnnData):
        result = data.path
    elif isinstance(data, ad.AnnData):
        filename = getattr(data, "filename", None)
        if filename:
            result = Path(filename)
        else:
            raise TypeError(
                "Operations in crispyx expect a backed AnnData object or file path. "
                "The provided AnnData appears to be in-memory (no .filename attribute)."
            )
    else:
        raise TypeError(
            f"Expected a path-like value or backed AnnData; received {type(data)!r}."
        )
    
    if require_exists and not result.exists():
        raise FileNotFoundError(f"Data file not found: {result}")
    
    return result


def resolve_output_path(
    input_path: str | Path,
    *,
    suffix: str,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    module: str = "crispyx",
) -> Path:
    """Construct an informative output path for an intermediate ``.h5ad`` file."""

    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir is not None else input_path.parent
    if data_name:
        # Preserve any existing module prefix supplied by the caller.
        base = data_name
        if module and not base.startswith(f"{module}_"):
            base = f"{module}_{base}"

        # If the provided name does not already encode the suffix, append it to avoid
        # different intermediates overwriting each other when the same ``data_name``
        # is reused across pipeline steps.
        if not base.endswith(f"_{suffix}"):
            base = f"{base}_{suffix}"

        return output_dir / f"{base}.h5ad"

    return output_dir / f"{module}_{suffix}.h5ad"


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


def read_h5ad_ondisk(
    path: str | Path,
    *,
    n_obs: int = 5,
    n_vars: int = 5,
) -> AnnData:
    """Open an ``.h5ad`` file on disk, print a preview, and return a read-only view."""

    adata_ro = AnnData(path)
    backed = adata_ro.backed
    try:
        print(backed)
        if n_obs > 0 and backed.n_obs > 0:
            print("First obs rows:")
            print(backed.obs.head(n_obs))
        if n_vars > 0 and backed.n_vars > 0:
            print("First var rows:")
            print(backed.var.head(n_vars))
    except Exception:
        adata_ro.close()
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
    dtype: np.dtype | type = np.float64,
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
    dtype:
        Data type for the output dense array. Defaults to float64.

    Returns
    -------
    tuple
        A tuple ``(normalised, library_size)`` where ``normalised`` is a dense
        array containing the normalised counts and ``library_size``
        contains the per-cell library sizes that were used.
    """

    dense = _to_dense(block).astype(dtype, copy=True)
    if dense.ndim != 2:
        raise ValueError("block must be two-dimensional")

    if library_size is None:
        library_size = dense.sum(axis=1)
    else:
        library_size = np.asarray(library_size, dtype=dtype)
        if library_size.shape[0] != dense.shape[0]:
            raise ValueError(
                "library_size length does not match the number of cells in block"
            )

    scale = np.divide(
        float(target_sum),
        library_size,
        out=np.zeros_like(library_size, dtype=dtype),
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


def _extract_csr_components_dense(
    block: np.ndarray,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract CSR data, indices, and row_nnz from dense block efficiently.
    
    Uses numba-accelerated parallel extraction when available (60x faster),
    falling back to numpy vectorized operations otherwise.
    
    Parameters
    ----------
    block
        Dense 2D array of shape (n_rows, n_cols).
    dtype
        Target dtype for data array.
        
    Returns
    -------
    tuple
        (data, indices, row_nnz, total_nnz) where data and indices are flattened
        CSR components and row_nnz is counts per row.
    """
    if block.size == 0:
        empty_data = np.array([], dtype=dtype)
        empty_indices = np.array([], dtype=np.int32)
        empty_nnz = np.zeros(block.shape[0], dtype=np.int64)
        return empty_data, empty_indices, empty_nnz, 0
    
    # Ensure C-contiguous for numba
    if not block.flags['C_CONTIGUOUS']:
        block = np.ascontiguousarray(block)
    
    # Use numba-accelerated parallel extraction (60x faster than scipy)
        row_nnz = _numba_count_row_nnz(block)
        indptr = np.zeros(block.shape[0] + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(row_nnz)
        total_nnz = int(indptr[-1])
        
        if total_nnz == 0:
            return np.array([], dtype=dtype), np.array([], dtype=np.int32), row_nnz, 0
        
        data = np.empty(total_nnz, dtype=dtype)
        indices = np.empty(total_nnz, dtype=np.int32)
        _numba_extract_csr_data(block.astype(dtype, copy=False), indptr, data, indices)
        
        return data, indices, row_nnz, total_nnz


def write_filtered_subset(
    source_path: str | Path,
    *,
    cell_mask: np.ndarray,
    gene_mask: np.ndarray,
    output_path: str | Path,
    chunk_size: int = 4096,
    var_assignments: dict[str, Sequence] | None = None,
    row_nnz: np.ndarray | None = None,
    total_nnz: int | None = None,
    data_dtype: np.dtype | None = None,
    chunk_cache: Any = None,
) -> None:
    """Stream a filtered AnnData view to disk without materialising ``X``.
    
    Parameters
    ----------
    source_path
        Path to source h5ad file.
    cell_mask
        Boolean mask for cells to include.
    gene_mask
        Boolean mask for genes to include.
    output_path
        Path for output h5ad file.
    chunk_size
        Number of cells to process per chunk.
    var_assignments
        Optional dict of column assignments for var DataFrame.
    row_nnz
        Optional pre-computed non-zero counts per row. When provided along
        with total_nnz and data_dtype, skips the counting pass.
    total_nnz
        Optional pre-computed total non-zero count.
    data_dtype
        Optional pre-computed data type for the sparse matrix.
    chunk_cache
        Optional _ChunkCache object from qc module. When provided, reads
        CSR data from cache instead of re-reading the source matrix.
    """

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

    # Use pre-computed values if all three are provided, otherwise compute
    need_counting_pass = row_nnz is None or total_nnz is None or data_dtype is None
    
    if need_counting_pass:
        row_nnz = np.zeros(n_obs, dtype=np.int64)
        total_nnz = 0
        data_dtype_local: np.dtype | None = None

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
                if data_dtype_local is None and csr.nnz:
                    data_dtype_local = csr.data.dtype
                row_offset += size
        finally:
            backed.file.close()
        
        if data_dtype_local is None:
            data_dtype_local = np.float32
        data_dtype = data_dtype_local

    placeholder = sp.csr_matrix((n_obs, n_vars), dtype=data_dtype)
    adata = ad.AnnData(placeholder, obs=obs, var=var)
    adata.write(output_path)

    if n_obs == 0 or n_vars == 0:
        with h5py.File(output_path, "r+", libver='latest') as dest:
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

    # Phase 2 I/O Optimization: Use larger HDF5 chunks and write buffering
    # Optimal chunk size: balance between I/O overhead and cache efficiency
    # Target ~1MB chunks for data (assuming float32 = 4 bytes -> ~256K elements)
    hdf5_chunk_size = min(262144, max(8192, total_nnz // 16))  # 8K to 256K elements
    
    # Write buffer size: accumulate data before writing to reduce I/O syscalls
    write_buffer_size = hdf5_chunk_size * 2  # Buffer 2x chunk size before flushing

    with h5py.File(output_path, "r+", libver='latest') as dest:
        if "X" in dest:
            del dest["X"]
        grp = dest.create_group("X")
        grp.attrs["encoding-type"] = np.bytes_("csr_matrix")
        grp.attrs["encoding-version"] = np.bytes_("0.1.0")
        
        # Use explicit chunk sizes for better I/O performance
        data_ds = grp.create_dataset(
            "data", 
            shape=(total_nnz,), 
            dtype=data_dtype, 
            chunks=(hdf5_chunk_size,) if total_nnz >= hdf5_chunk_size else None
        )
        indices_ds = grp.create_dataset(
            "indices", 
            shape=(total_nnz,), 
            dtype=np.int32, 
            chunks=(hdf5_chunk_size,) if total_nnz >= hdf5_chunk_size else None
        )
        grp.create_dataset("indptr", data=indptr)
        grp.attrs["shape"] = np.array([n_obs, n_vars], dtype=np.int64)

        # Stream data with write buffering
        if chunk_cache is not None:
            # Read from cached CSR chunks (avoids re-reading the original matrix)
            offset = 0
            data_buffer = []
            indices_buffer = []
            buffer_nnz = 0
            
            for filtered_data, filtered_indices, n_cells in chunk_cache.iter_filtered_chunks(
                gene_indices, data_dtype
            ):
                nnz = len(filtered_data)
                if nnz:
                    data_buffer.append(filtered_data)
                    indices_buffer.append(filtered_indices)
                    buffer_nnz += nnz
                    
                    # Flush buffer when it exceeds threshold
                    if buffer_nnz >= write_buffer_size:
                        combined_data = np.concatenate(data_buffer)
                        combined_indices = np.concatenate(indices_buffer)
                        data_ds[offset : offset + buffer_nnz] = combined_data
                        indices_ds[offset : offset + buffer_nnz] = combined_indices
                        offset += buffer_nnz
                        data_buffer = []
                        indices_buffer = []
                        buffer_nnz = 0
            
            # Flush remaining buffer
            if buffer_nnz > 0:
                combined_data = np.concatenate(data_buffer)
                combined_indices = np.concatenate(indices_buffer)
                data_ds[offset : offset + buffer_nnz] = combined_data
                indices_ds[offset : offset + buffer_nnz] = combined_indices
        else:
            # Read from source matrix (fallback when cache not available)
            backed = read_backed(source_path)
            try:
                offset = 0
                data_buffer = []
                indices_buffer = []
                buffer_nnz = 0
                
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
                    
                    # Use optimized extraction for dense, scipy for sparse
                    if sp.issparse(block):
                        csr = _ensure_csr(block, dtype=data_dtype)
                        chunk_data = csr.data
                        chunk_indices = csr.indices.astype(np.int32, copy=False)
                        nnz = int(csr.nnz)
                    else:
                        # Dense: use numba-accelerated direct extraction
                        chunk_data, chunk_indices, _row_nnz, nnz = _extract_csr_components_dense(
                            block, data_dtype
                        )
                    
                    if nnz:
                        data_buffer.append(chunk_data)
                        indices_buffer.append(chunk_indices)
                        buffer_nnz += nnz
                        
                        # Flush buffer when it exceeds threshold
                        if buffer_nnz >= write_buffer_size:
                            combined_data = np.concatenate(data_buffer)
                            combined_indices = np.concatenate(indices_buffer)
                            data_ds[offset : offset + buffer_nnz] = combined_data
                            indices_ds[offset : offset + buffer_nnz] = combined_indices
                            offset += buffer_nnz
                            data_buffer = []
                            indices_buffer = []
                            buffer_nnz = 0
                
                # Flush remaining buffer
                if buffer_nnz > 0:
                    combined_data = np.concatenate(data_buffer)
                    combined_indices = np.concatenate(indices_buffer)
                    data_ds[offset : offset + buffer_nnz] = combined_data
                    indices_ds[offset : offset + buffer_nnz] = combined_indices
            finally:
                backed.file.close()


def normalize_total_log1p(
    data: str | Path | "AnnData" | ad.AnnData,
    output_path: str | Path | None = None,
    *,
    normalize: bool = True,
    log1p: bool = True,
    target_sum: float = 1e4,
    chunk_size: int = 4096,
    output_dir: str | Path | None = None,
    data_name: str | None = None,
    verbose: bool = True,
) -> "AnnData":
    """Stream normalize and/or log-transform an h5ad file without loading it fully into memory.
    
    This function processes the source file in chunks, optionally applying:
    1. Total-count normalization (scanpy.pp.normalize_total equivalent)
    2. Log1p transformation (scanpy.pp.log1p equivalent)
    
    The output is written as a sparse CSR matrix. This is the streaming equivalent
    of calling ``scanpy.pp.normalize_total`` followed by ``scanpy.pp.log1p``.
    
    Parameters
    ----------
    data
        Path to source h5ad file, or a backed AnnData object.
    output_path
        Path for output h5ad file. If None, uses output_dir/data_name pattern.
    normalize
        Whether to apply total-count normalization. Default True.
    log1p
        Whether to apply log1p transformation. Default True.
    target_sum
        Target total counts per cell after normalization. Default 1e4.
        Only used if normalize=True.
    chunk_size
        Number of cells to process per chunk. Default 4096.
    output_dir
        Directory for output file. Defaults to input file's directory.
    data_name
        Custom name for output file. If None, uses "normalized" suffix.
    verbose
        Print progress information.
    
    Returns
    -------
    AnnData
        Read-only AnnData wrapper pointing to the output file.
    
    Examples
    --------
    >>> # Full normalization + log1p (default)
    >>> adata_norm = cx.pp.normalize_total_log1p(adata_ro, output_dir=OUTPUT_DIR, data_name="normalized")
    
    >>> # Only log1p (no normalization)
    >>> adata_log = cx.pp.normalize_total_log1p(adata_ro, normalize=False, output_dir=OUTPUT_DIR)
    
    >>> # Only normalization (no log1p)
    >>> adata_norm = cx.pp.normalize_total_log1p(adata_ro, log1p=False, output_dir=OUTPUT_DIR)
    
    >>> # Use explicit output path
    >>> adata_norm = cx.pp.normalize_total_log1p(adata_ro, "results/normalized.h5ad")
    """
    if not normalize and not log1p:
        raise ValueError("At least one of normalize or log1p must be True")
    
    # Resolve input path from various input types
    source_path = resolve_data_path(data, require_exists=True)
    
    # Resolve output path
    if output_path is not None:
        output_path = Path(output_path)
    else:
        # Build suffix based on options
        if normalize and log1p:
            suffix = "normalized_log1p"
        elif normalize:
            suffix = "normalized"
        else:
            suffix = "log1p"
        output_path = resolve_output_path(
            source_path,
            suffix=suffix,
            output_dir=output_dir,
            data_name=data_name,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ops = []
    if normalize:
        ops.append("normalize")
    if log1p:
        ops.append("log1p")
    if verbose:
        print(f"Generating preprocessed dataset (streaming, {'+'.join(ops)}): {output_path}")
    
    # First pass: count non-zeros and get metadata
    backed = read_backed(source_path)
    try:
        n_obs = backed.n_obs
        n_vars = backed.n_vars
        obs = backed.obs.copy()
        var = backed.var.copy()
        obs.index = obs.index.astype(str)
        var.index = var.index.astype(str)
        
        # Count non-zeros per row (after normalization, same sparsity as input)
        row_nnz = np.zeros(n_obs, dtype=np.int64)
        total_nnz = 0
        
        row_offset = 0
        for slc, block in iter_matrix_chunks(
            backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
        ):
            csr = _ensure_csr(block)
            counts = np.diff(csr.indptr)
            row_nnz[row_offset : row_offset + len(counts)] = counts
            total_nnz += int(csr.nnz)
            row_offset += len(counts)
    finally:
        backed.file.close()
    
    if total_nnz == 0:
        # Empty matrix: write placeholder
        placeholder = sp.csr_matrix((n_obs, n_vars), dtype=np.float32)
        adata = ad.AnnData(placeholder, obs=obs, var=var)
        adata.write(output_path)
        return output_path
    
    # Compute indptr
    indptr = np.zeros(n_obs + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=indptr[1:])
    
    # HDF5 chunk sizing
    hdf5_chunk_size = min(262144, max(8192, total_nnz // 16))
    
    # Create output file with placeholder
    placeholder = sp.csr_matrix((n_obs, n_vars), dtype=np.float32)
    adata = ad.AnnData(placeholder, obs=obs, var=var)
    adata.write(output_path)
    
    # Second pass: normalize, log1p, and write
    with h5py.File(output_path, "r+", libver='latest') as dest:
        if "X" in dest:
            del dest["X"]
        grp = dest.create_group("X")
        grp.attrs["encoding-type"] = np.bytes_("csr_matrix")
        grp.attrs["encoding-version"] = np.bytes_("0.1.0")
        
        data_ds = grp.create_dataset(
            "data",
            shape=(total_nnz,),
            dtype=np.float32,
            chunks=(hdf5_chunk_size,) if total_nnz >= hdf5_chunk_size else None,
        )
        indices_ds = grp.create_dataset(
            "indices",
            shape=(total_nnz,),
            dtype=np.int32,
            chunks=(hdf5_chunk_size,) if total_nnz >= hdf5_chunk_size else None,
        )
        grp.create_dataset("indptr", data=indptr)
        grp.attrs["shape"] = np.array([n_obs, n_vars], dtype=np.int64)
        
        backed = read_backed(source_path)
        try:
            offset = 0
            for slc, block in iter_matrix_chunks(
                backed, axis=0, chunk_size=chunk_size, convert_to_dense=False
            ):
                csr = _ensure_csr(block)
                
                # Start with original data
                processed_data = csr.data.astype(np.float32)
                
                # Apply normalization if requested
                if normalize:
                    # Library size per cell
                    lib_sizes = np.asarray(csr.sum(axis=1)).ravel()
                    # Avoid division by zero
                    scale = np.divide(
                        target_sum, lib_sizes,
                        out=np.zeros_like(lib_sizes, dtype=np.float64),
                        where=lib_sizes > 0,
                    )
                    
                    # Apply normalization to data (CSR stores data in row-major order)
                    # For each row i, data[indptr[i]:indptr[i+1]] are the values
                    for i in range(csr.shape[0]):
                        start_idx = csr.indptr[i]
                        end_idx = csr.indptr[i + 1]
                        processed_data[start_idx:end_idx] = (
                            processed_data[start_idx:end_idx].astype(np.float64) * scale[i]
                        ).astype(np.float32)
                
                # Apply log1p if requested
                if log1p:
                    processed_data = np.log1p(processed_data)
                
                # Write to HDF5
                nnz = len(processed_data)
                if nnz:
                    data_ds[offset : offset + nnz] = processed_data
                    indices_ds[offset : offset + nnz] = csr.indices.astype(np.int32)
                    offset += nnz
        finally:
            backed.file.close()
    
    if verbose:
        print(f"  ✓ Preprocessed dataset written: {n_obs} cells × {n_vars} genes")
    
    return AnnData(output_path)


# Alias for backward compatibility
write_normalized_log1p = normalize_total_log1p


def calculate_optimal_chunk_size(
    n_obs: int,
    n_vars: int,
    available_memory_gb: float | None = None,
    safety_factor: float = 8.0,
    min_chunk: int = 512,
    max_chunk: int = 4096,
) -> int:
    """Calculate optimal chunk size based on dataset dimensions and available memory.
    
    Parameters
    ----------
    n_obs
        Number of observations (cells) in the dataset.
    n_vars
        Number of variables (genes) in the dataset.
    available_memory_gb
        Available memory in gigabytes. If None, auto-detects using psutil.
    safety_factor
        Safety multiplier to account for overhead (default 8.0 for backed operations).
    min_chunk
        Minimum chunk size to return (default 512).
    max_chunk
        Maximum chunk size to return (default 4096).
    
    Returns
    -------
    int
        Recommended chunk size, clamped to [min_chunk, max_chunk].
    
    Examples
    --------
    >>> calculate_optimal_chunk_size(100000, 20000, available_memory_gb=32)
    2000
    """
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / 1e9
        except ImportError:
            logger.warning(
                "psutil not installed, using default 16GB for chunk size calculation. "
                "Install with: pip install psutil"
            )
            available_memory_gb = 16.0
    
    # Calculate chunk size based on memory
    # Each chunk uses approximately: chunk_size * n_vars * 8 bytes (float64)
    # Multiply by safety_factor for overhead
    bytes_per_chunk = n_vars * 8 * safety_factor
    max_chunk_from_memory = int((available_memory_gb * 1e9) / bytes_per_chunk)
    
    # Clamp to reasonable range
    chunk_size = max(min_chunk, min(max_chunk, max_chunk_from_memory))
    
    logger.info(
        f"Calculated chunk size: {chunk_size} "
        f"(dataset: {n_obs} cells × {n_vars} genes, "
        f"available memory: {available_memory_gb:.1f}GB)"
    )
    
    return chunk_size


def calculate_optimal_gene_chunk_size(
    n_obs: int,
    n_vars: int,
    n_groups: int | None = None,
    available_memory_gb: float | None = None,
    safety_factor: float = 8.0,
    memory_fraction: float = 0.5,
    min_chunk: int = 32,
    max_chunk: int = 512,
) -> int:
    """Calculate optimal gene chunk size for column-wise operations.
    
    For operations that iterate over genes (columns), such as Wilcoxon tests,
    each chunk loads all cells for a subset of genes. Memory usage is dominated
    by n_obs × chunk_size rather than chunk_size × n_vars.
    
    Enhanced to account for the number of perturbation groups, which significantly
    impacts memory usage due to output array allocation.
    
    Parameters
    ----------
    n_obs
        Number of observations (cells) in the dataset.
    n_vars
        Number of variables (genes) in the dataset.
    n_groups
        Number of perturbation groups. If provided, used to estimate memory
        for output arrays. Large group counts require smaller chunks.
    available_memory_gb
        Available memory in gigabytes. If None, auto-detects using psutil.
    safety_factor
        Safety multiplier to account for overhead (default 8.0).
    memory_fraction
        Fraction of available memory to use (default 0.5). Leave headroom
        for memory-mapped arrays and system overhead.
    min_chunk
        Minimum chunk size to return (default 32).
    max_chunk
        Maximum chunk size to return (default 512).
    
    Returns
    -------
    int
        Recommended gene chunk size, clamped to [min_chunk, max_chunk].
    
    Examples
    --------
    >>> calculate_optimal_gene_chunk_size(100000, 20000, available_memory_gb=32)
    512
    >>> calculate_optimal_gene_chunk_size(4000000, 38000, n_groups=18000, available_memory_gb=128)
    64
    """
    if available_memory_gb is None:
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / 1e9
        except ImportError:
            logger.warning(
                "psutil not installed, using default 16GB for chunk size calculation. "
                "Install with: pip install psutil"
            )
            available_memory_gb = 16.0
    
    # Usable memory = fraction of available (default 50%)
    usable_memory_bytes = available_memory_gb * memory_fraction * 1e9
    
    # Base memory: dense conversion of n_obs × chunk_size (float64)
    base_memory_per_gene = n_obs * 8
    
    # Group memory: output arrays of n_groups × chunk_size (float64) × ~8 arrays
    # (effect, u_stat, pvalue, z_score, lfc, pts, pts_rest, order)
    group_memory_per_gene = (n_groups * 8 * 8) if n_groups else 0
    
    # Total memory per gene with safety factor
    total_memory_per_gene = (base_memory_per_gene + group_memory_per_gene) * safety_factor
    
    # Calculate max chunk from memory
    max_chunk_from_memory = int(usable_memory_bytes / total_memory_per_gene) if total_memory_per_gene > 0 else max_chunk
    
    # Dynamic max_chunk based on n_groups (datasets with many groups need smaller chunks)
    effective_max_chunk = max_chunk
    if n_groups is not None:
        if n_groups > 10000:
            effective_max_chunk = min(effective_max_chunk, 128)
        elif n_groups > 5000:
            effective_max_chunk = min(effective_max_chunk, 256)
        elif n_groups > 2000:
            effective_max_chunk = min(effective_max_chunk, 384)
    
    # Clamp to reasonable range
    chunk_size = max(min_chunk, min(effective_max_chunk, max_chunk_from_memory))
    
    logger.info(
        f"Calculated gene chunk size: {chunk_size} "
        f"(dataset: {n_obs} cells × {n_vars} genes, "
        f"groups: {n_groups or 'unknown'}, "
        f"available memory: {available_memory_gb:.1f}GB)"
    )
    
    return chunk_size


def calculate_adaptive_qc_thresholds(
    adata: ad.AnnData,
    perturbation_column: str,
    mode: str = "conservative",
    sample_size: int = 10000,
    chunk_size: int | None = None,
) -> dict:
    """Calculate adaptive QC thresholds based on data distribution.
    
    Uses percentile-based approach to determine appropriate QC parameters
    that retain most of the data while filtering outliers.
    
    Parameters
    ----------
    adata
        AnnData object (can be backed).
    perturbation_column
        Column in adata.obs containing perturbation labels.
    mode
        'conservative' (10th percentile, retains ~90%) or 
        'aggressive' (5th percentile, retains ~95%).
    sample_size
        Maximum number of cells to sample for gene expression analysis.
    chunk_size
        Optional fixed chunk size to use. If None, calculated adaptively.
    
    Returns
    -------
    dict
        Dictionary with keys: min_genes, min_cells_per_perturbation,
        min_cells_per_gene, chunk_size.
    
    Examples
    --------
    >>> adata = ad.read_h5ad("data.h5ad", backed='r')
    >>> thresholds = calculate_adaptive_qc_thresholds(adata, "perturbation")
    >>> adata.file.close()
    """
    percentile = 10.0 if mode == "conservative" else 5.0
    
    # Analyze perturbation sizes
    if perturbation_column not in adata.obs.columns:
        raise KeyError(
            f"Perturbation column '{perturbation_column}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    pert_counts = adata.obs[perturbation_column].value_counts()
    p_percentile = pert_counts.quantile(percentile / 100.0)
    min_cells_per_pert = int(max(5, min(50, p_percentile)))
    
    # Analyze gene expression (sample if dataset is large)
    n_sample = min(sample_size, adata.n_obs)
    if n_sample < adata.n_obs:
        sample_idx = np.random.choice(adata.n_obs, n_sample, replace=False)
        sample_idx = np.sort(sample_idx)  # Sort for efficient backed access
    else:
        sample_idx = None
    
    # Calculate cells per gene and genes per cell efficiently
    # Use chunked processing to handle backed datasets
    cells_per_gene = np.zeros(adata.n_vars, dtype=np.int64)
    genes_per_cell = np.zeros(n_sample if sample_idx is not None else adata.n_obs, dtype=np.int64)
    
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(adata.n_obs, adata.n_vars)
    cell_idx = 0
    
    for chunk_start in range(0, adata.n_obs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, adata.n_obs)
        
        # Get the chunk indices
        if sample_idx is not None:
            # Get indices that fall in this chunk
            mask = (sample_idx >= chunk_start) & (sample_idx < chunk_end)
            if not mask.any():
                continue
            chunk_indices = sample_idx[mask] - chunk_start
            X_chunk = adata.X[chunk_start:chunk_end][chunk_indices]
        else:
            X_chunk = adata.X[chunk_start:chunk_end]
        
        # Process chunk (works with both sparse and dense)
        if sp.issparse(X_chunk):
            # Count non-zeros per gene (column)
            cells_per_gene += np.asarray(np.diff(X_chunk.tocsc().indptr))
            # Count non-zeros per cell (row)
            chunk_genes_per_cell = np.asarray(np.diff(X_chunk.tocsr().indptr))
        else:
            # Dense matrix
            X_chunk_bool = X_chunk > 0
            cells_per_gene += np.asarray(X_chunk_bool.sum(axis=0)).ravel()
            chunk_genes_per_cell = np.asarray(X_chunk_bool.sum(axis=1)).ravel()
        
        # Store genes per cell for this chunk
        n_cells_in_chunk = chunk_genes_per_cell.shape[0]
        genes_per_cell[cell_idx:cell_idx + n_cells_in_chunk] = chunk_genes_per_cell
        cell_idx += n_cells_in_chunk
    
    # Trim genes_per_cell if we didn't fill it completely
    genes_per_cell = genes_per_cell[:cell_idx]
    
    # Calculate thresholds from the collected statistics
    gene_percentile = np.percentile(cells_per_gene, percentile)
    min_cells_per_gene = int(max(5, min(100, gene_percentile)))
    
    median_genes = int(np.median(genes_per_cell))
    min_genes = max(5, min(50, median_genes // 10))  # 10% of median
    
    # Calculate optimal chunk size if not provided
    if chunk_size is None:
        chunk_size = calculate_optimal_chunk_size(adata.n_obs, adata.n_vars)
    
    thresholds = {
        "min_genes": min_genes,
        "min_cells_per_perturbation": min_cells_per_pert,
        "min_cells_per_gene": min_cells_per_gene,
        "chunk_size": chunk_size,
    }
    
    logger.info(
        f"Adaptive QC thresholds ({mode} mode):\n"
        f"  - min_genes: {min_genes}\n"
        f"  - min_cells_per_perturbation: {min_cells_per_pert} "
        f"({percentile}th percentile: {p_percentile:.1f})\n"
        f"  - min_cells_per_gene: {min_cells_per_gene} "
        f"({percentile}th percentile: {gene_percentile:.1f})\n"
        f"  - chunk_size: {chunk_size}"
    )
    
    return thresholds


def standardize_dataset(
    dataset_path: Path | str,
    perturbation_column: str,
    control_label: str | None,
    gene_name_column: str | None,
    output_dir: Path | str,
    force: bool = False,
) -> Path:
    """Standardize dataset column names and control labels with caching.
    
    Creates a standardized copy of the dataset with:
    - perturbation_column renamed to 'perturbation'
    - control labels standardized to 'control'
    - gene_name_column set as var.index if specified
    
    This function uses a streaming approach to avoid loading the X matrix
    into memory, making it suitable for very large datasets (>1M cells).
    
    Standardized files are cached in {output_dir}/.cache/ and reused
    unless force=True.
    
    Parameters
    ----------
    dataset_path
        Path to original dataset (.h5ad file).
    perturbation_column
        Name of perturbation column in original dataset.
    control_label
        Control label to standardize. If None, auto-detects.
    gene_name_column
        Gene name column to use as var.index. If None, uses existing var.index.
    output_dir
        Directory for cached standardized files.
    force
        If True, regenerate standardized file even if cache exists.
    
    Returns
    -------
    Path
        Path to standardized dataset (either cached or newly created).
    
    Examples
    --------
    >>> standardized_path = standardize_dataset(
    ...     "data/original.h5ad",
    ...     perturbation_column="gene",
    ...     control_label=None,
    ...     gene_name_column="gene_symbols",
    ...     output_dir="results",
    ...     force=False
    ... )
    """
    import datetime
    import shutil
    
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    standardized_path = cache_dir / f"standardized_{dataset_path.stem}.h5ad"
    
    # Check if cached version exists
    if standardized_path.exists() and not force:
        logger.info(f"Using cached standardized dataset: {standardized_path}")
        return standardized_path
    
    logger.info(f"Standardizing dataset: {dataset_path.name}")
    logger.info(f"  - Perturbation column: '{perturbation_column}' → 'perturbation'")
    
    # Track standardization metadata
    metadata = {
        "original_path": str(dataset_path),
        "standardization_timestamp": datetime.datetime.now().isoformat(),
        "column_mappings": {},
        "label_mappings": {},
    }
    
    # Read obs/var metadata only (without loading X into memory)
    adata = ad.read_h5ad(dataset_path, backed='r')
    obs_df = adata.obs.copy()
    var_df = adata.var.copy()
    uns_dict = dict(adata.uns)  # shallow copy of uns
    adata.file.close()
    
    # Standardize perturbation column in obs
    if perturbation_column != "perturbation":
        if perturbation_column not in obs_df.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' not found. "
                f"Available: {list(obs_df.columns)}"
            )
        obs_df.rename(columns={perturbation_column: "perturbation"}, inplace=True)
        metadata["column_mappings"]["perturbation"] = perturbation_column
        logger.info(f"  - Renamed '{perturbation_column}' → 'perturbation'")
    
    # Standardize control label
    labels = obs_df["perturbation"].astype(str).to_numpy()
    detected_control = resolve_control_label(labels, control_label, verbose=False)
    
    if detected_control != "control":
        obs_df["perturbation"] = (
            obs_df["perturbation"]
            .astype(str)
            .replace({detected_control: "control"})
        )
        metadata["label_mappings"][detected_control] = "control"
        logger.info(f"  - Standardized control: '{detected_control}' → 'control'")
    else:
        logger.info(f"  - Control label already standardized: 'control'")
    
    # Standardize gene names in var
    if gene_name_column is not None:
        if gene_name_column in var_df.columns:
            if not (var_df.index == var_df[gene_name_column]).all():
                var_df.index = var_df[gene_name_column].values
                metadata["column_mappings"]["var.index"] = gene_name_column
                logger.info(f"  - Set var.index from '{gene_name_column}'")
        else:
            logger.warning(
                f"Gene column '{gene_name_column}' not found in var. "
                f"Using existing var.index."
            )
    
    # Store metadata
    if "standardization_metadata" not in uns_dict:
        uns_dict["standardization_metadata"] = {}
    uns_dict["standardization_metadata"].update(metadata)
    
    # Copy h5ad file at filesystem level (streaming - no memory load)
    logger.info(f"  - Copying dataset (streaming, no X matrix load)...")
    shutil.copy2(dataset_path, standardized_path)
    
    # Modify obs/var/uns in-place using h5py
    logger.info(f"  - Updating metadata in copied file...")
    with h5py.File(standardized_path, 'r+') as f:
        # Update obs - need to rewrite the obs group
        # Read current obs structure and update
        _update_h5ad_dataframe(f, 'obs', obs_df)
        
        # Update var - need to rewrite the var group
        _update_h5ad_dataframe(f, 'var', var_df)
        
        # Update uns - handle the standardization_metadata key
        _update_h5ad_uns(f, 'uns', uns_dict)
    
    logger.info(f"Saved standardized dataset: {standardized_path}")
    
    return standardized_path


def _update_h5ad_dataframe(h5file: h5py.File, group_name: str, df: pd.DataFrame) -> None:
    """Update obs or var DataFrame in an h5ad file in-place.
    
    This function handles the anndata HDF5 format where DataFrames are stored
    as groups with individual columns as datasets.
    """
    if group_name not in h5file:
        return
    
    grp = h5file[group_name]
    
    # Update the index (stored as _index attribute or separate dataset)
    index_key = grp.attrs.get('_index', '_index')
    if isinstance(index_key, bytes):
        index_key = index_key.decode('utf-8')
    
    if index_key in grp:
        del grp[index_key]
    # Store index as variable-length strings
    index_vals = df.index.astype(str).values
    grp.create_dataset(index_key, data=index_vals.astype('O'), dtype=h5py.special_dtype(vlen=str))
    
    # Update column names (stored as 'column-order' attribute)
    col_names = list(df.columns)
    # HDF5 attributes don't support object dtype; encode strings as bytes
    grp.attrs['column-order'] = np.array([c.encode('utf-8') for c in col_names])
    
    # Update each column
    for col in df.columns:
        if col in grp:
            del grp[col]
        
        col_data = df[col]
        
        # Handle categorical columns
        if hasattr(col_data, 'cat'):
            # Store as categorical (anndata format)
            cat_grp = grp.create_group(col)
            cat_grp.attrs['encoding-type'] = 'categorical'
            cat_grp.attrs['encoding-version'] = '0.2.0'
            cat_grp.attrs['ordered'] = col_data.cat.ordered  # Required for anndata
            
            # Store categories
            categories = col_data.cat.categories.astype(str).values
            cat_grp.create_dataset('categories', data=categories.astype('O'), 
                                   dtype=h5py.special_dtype(vlen=str))
            
            # Store codes
            cat_grp.create_dataset('codes', data=col_data.cat.codes.values)
        elif col_data.dtype == object or col_data.dtype.kind in ('U', 'S'):
            # String column - store as variable-length strings
            str_vals = col_data.astype(str).values
            grp.create_dataset(col, data=str_vals.astype('O'), 
                               dtype=h5py.special_dtype(vlen=str))
        else:
            # Numeric column
            grp.create_dataset(col, data=col_data.values)


def _update_h5ad_uns(h5file: h5py.File, group_name: str, uns_dict: dict) -> None:
    """Update uns dict in an h5ad file in-place.
    
    Only updates the standardization_metadata key to avoid breaking other uns data.
    """
    if group_name not in h5file:
        h5file.create_group(group_name)
    
    grp = h5file[group_name]
    
    # Only update standardization_metadata to be safe
    key = 'standardization_metadata'
    if key in uns_dict:
        if key in grp:
            del grp[key]
        
        # Store as a group with string datasets for each subkey
        meta_grp = grp.create_group(key)
        meta_grp.attrs['encoding-type'] = 'dict'
        meta_grp.attrs['encoding-version'] = '0.1.0'
        
        for subkey, subval in uns_dict[key].items():
            if isinstance(subval, dict):
                # Nested dict - store as JSON string
                import json
                meta_grp.create_dataset(subkey, data=json.dumps(subval),
                                        dtype=h5py.special_dtype(vlen=str))
            elif isinstance(subval, str):
                meta_grp.create_dataset(subkey, data=subval,
                                        dtype=h5py.special_dtype(vlen=str))
            else:
                meta_grp.create_dataset(subkey, data=str(subval),
                                        dtype=h5py.special_dtype(vlen=str))


def needs_sorting_for_nbglm(
    path: str | Path,
    perturbation_column: str = "perturbation",
    *,
    min_cells: int = 360_000,
    min_perturbations: int = 100,
    contiguity_threshold: float = 0.5,
) -> bool:
    """Check if a dataset would benefit from sorting by perturbation for NB-GLM.
    
    Large datasets with scattered cells benefit from having cells sorted
    by perturbation label, as this enables contiguous I/O reads instead of
    random access. This is especially important when the data is stored on
    HDD (rotational disk).
    
    The default thresholds are based on I/O overhead analysis:
    - At ~100 IOPS (typical HDD), 360K cells = 1 hour of random I/O overhead
    - min_perturbations=100 ensures sufficient parallel workload to benefit
    - contiguity_threshold=0.5 catches scattered datasets
    
    Parameters
    ----------
    path
        Path to h5ad file.
    perturbation_column
        Column in obs containing perturbation labels.
    min_cells
        Minimum number of cells for sorting to be recommended.
        Default: 360,000 (~1 hour of random I/O on HDD at 100 IOPS).
    min_perturbations
        Minimum number of perturbations for sorting to be recommended.
    contiguity_threshold
        If average contiguity is below this threshold, sorting is recommended.
        Contiguity is the fraction of a perturbation's cells that are in a
        contiguous block (1.0 = perfectly contiguous, 0.0 = completely scattered).
    
    Returns
    -------
    bool
        True if sorting is recommended, False otherwise.
    """
    backed = read_backed(path)
    try:
        # First check if file is already sorted
        if "sorting_metadata" in backed.uns:
            metadata = backed.uns["sorting_metadata"]
            if metadata.get("sorted_by") == perturbation_column:
                logger.debug(f"Dataset is already sorted by '{perturbation_column}'")
                return False
        
        n_cells = backed.n_obs
        
        # Check cell count threshold
        if n_cells < min_cells:
            logger.debug(f"Dataset has {n_cells:,} cells < {min_cells:,}, sorting not needed")
            return False
        
        # Get perturbation labels
        if perturbation_column not in backed.obs.columns:
            logger.warning(f"Perturbation column '{perturbation_column}' not found")
            return False
        
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        unique_labels = np.unique(labels)
        n_perts = len(unique_labels)
        
        # Check perturbation count threshold
        if n_perts < min_perturbations:
            logger.debug(f"Dataset has {n_perts} perturbations < {min_perturbations}, sorting not needed")
            return False
        
        # Sample perturbations to estimate contiguity
        sample_perts = unique_labels[:min(20, n_perts)]
        total_contiguity = 0.0
        
        for pert in sample_perts:
            indices = np.where(labels == pert)[0]
            if len(indices) < 2:
                total_contiguity += 1.0
                continue
            span = indices.max() - indices.min() + 1
            contiguity = len(indices) / span
            total_contiguity += contiguity
        
        avg_contiguity = total_contiguity / len(sample_perts)
        
        if avg_contiguity >= contiguity_threshold:
            logger.debug(f"Dataset contiguity {avg_contiguity:.1%} >= {contiguity_threshold:.1%}, sorting not needed")
            return False
        
        logger.info(
            f"Dataset would benefit from sorting: {n_cells:,} cells, {n_perts} perturbations, "
            f"contiguity {avg_contiguity:.1%} < {contiguity_threshold:.1%}"
        )
        return True
        
    finally:
        backed.file.close()


def sort_by_perturbation(
    path: str | Path,
    perturbation_column: str = "perturbation",
    control_label: str | None = None,
    *,
    output_path: str | Path | None = None,
    chunk_size: int = 4096,
    force: bool = False,
) -> Path:
    """Sort cells by perturbation label for contiguous I/O access.
    
    Creates a new h5ad file with cells reordered so that all cells from each
    perturbation are contiguous. Control cells are placed first, followed by
    each perturbation group in alphabetical order. This enables efficient
    sequential reads when processing perturbations in parallel.
    
    The function works in streaming mode to handle datasets larger than memory.
    Sorting information is stored in uns['sorting_metadata'].
    
    Parameters
    ----------
    path
        Path to input h5ad file.
    perturbation_column
        Column in obs containing perturbation labels.
    control_label
        Label for control cells. If None, auto-detected from common patterns.
    output_path
        Path for output sorted file. If None, appends '_sorted' to input name.
    chunk_size
        Number of cells to process per chunk during streaming write.
    force
        If True, recreate sorted file even if it already exists.
    
    Returns
    -------
    Path
        Path to the sorted h5ad file.
    
    Examples
    --------
    >>> sorted_path = sort_by_perturbation(
    ...     "data/large_dataset.h5ad",
    ...     perturbation_column="perturbation",
    ...     control_label="control",
    ... )
    >>> # sorted_path is now "data/large_dataset_sorted.h5ad"
    
    Notes
    -----
    The sorted file contains additional metadata in uns['sorting_metadata']:
    - original_path: Path to the original unsorted file
    - sort_order: Array mapping new indices to original indices
    - perturbation_boundaries: Dict mapping perturbation labels to (start, end) indices
    - sorted_by: The column used for sorting
    - timestamp: When the sorting was performed
    """
    import datetime
    
    path = Path(path)
    
    # Determine output path
    if output_path is None:
        output_path = path.parent / f"{path.stem}_sorted.h5ad"
    else:
        output_path = Path(output_path)
    
    # Check if already sorted
    if output_path.exists() and not force:
        # Verify it's properly sorted
        try:
            backed = read_backed(output_path)
            has_metadata = "sorting_metadata" in backed.uns
            backed.file.close()
            if has_metadata:
                logger.info(f"Using existing sorted file: {output_path}")
                return output_path
        except Exception:
            pass  # File exists but invalid, recreate
    
    logger.info(f"Sorting dataset by perturbation: {path}")
    
    # Read metadata
    backed = read_backed(path)
    try:
        n_cells = backed.n_obs
        n_genes = backed.n_vars
        
        # Get perturbation labels
        if perturbation_column not in backed.obs.columns:
            raise KeyError(
                f"Perturbation column '{perturbation_column}' not found. "
                f"Available: {list(backed.obs.columns)}"
            )
        
        labels = backed.obs[perturbation_column].astype(str).to_numpy()
        
        # Detect control label if not specified
        if control_label is None:
            control_label = resolve_control_label(labels, None)
        
        # Create sort order: control first, then alphabetical
        unique_labels = sorted(set(labels) - {control_label})
        label_order = [control_label] + unique_labels
        
        # Create mapping from label to sort priority
        label_priority = {label: i for i, label in enumerate(label_order)}
        
        # Get sort indices (stable sort to preserve original order within groups)
        sort_keys = np.array([label_priority[l] for l in labels])
        sort_indices = np.argsort(sort_keys, kind='stable')
        
        # Compute perturbation boundaries
        sorted_labels = labels[sort_indices]
        boundaries = {}
        current_label = None
        start_idx = 0
        
        for i, label in enumerate(sorted_labels):
            if label != current_label:
                if current_label is not None:
                    boundaries[current_label] = [start_idx, i]  # Use list, not tuple
                current_label = label
                start_idx = i
        if current_label is not None:
            boundaries[current_label] = [start_idx, len(sorted_labels)]  # Use list, not tuple
        
        # Read obs and var
        obs_sorted = backed.obs.iloc[sort_indices].copy()
        var = backed.var.copy()
        
        # Get uns (convert to regular dict for modification)
        uns = dict(backed.uns)
        
    finally:
        backed.file.close()
    
    # Add sorting metadata - convert all to h5ad-compatible types
    # Note: boundaries values are [start, end] lists (tuples not serializable)
    sorting_metadata = {
        "original_path": str(path),
        "sorted_by": perturbation_column,
        "control_label": control_label,
        "timestamp": datetime.datetime.now().isoformat(),
        "n_perturbations": len(label_order),
        "perturbation_boundaries": boundaries,  # Dict[str, List[int]]
    }
    # Don't store full sort_order for large datasets (memory waste)
    if len(sort_indices) < 100000:
        sorting_metadata["sort_order"] = sort_indices.tolist()
    uns["sorting_metadata"] = sorting_metadata
    
    # Create output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  Reordering {n_cells:,} cells into {len(label_order)} contiguous groups...")
    
    # Check if source is dense or sparse
    storage_format = get_matrix_storage_format(path)
    is_dense = storage_format == "dense"
    
    if is_dense:
        # For dense storage: write directly as dense
        _write_sorted_dense(
            source_path=path,
            output_path=output_path,
            sort_indices=sort_indices,
            obs_sorted=obs_sorted,
            var=var,
            uns=uns,
            chunk_size=chunk_size,
        )
    else:
        # For sparse storage: stream as CSR
        _write_sorted_sparse(
            source_path=path,
            output_path=output_path,
            sort_indices=sort_indices,
            obs_sorted=obs_sorted,
            var=var,
            uns=uns,
            chunk_size=chunk_size,
        )
    
    logger.info(f"Saved sorted dataset: {output_path}")
    logger.info(f"  Perturbation groups: {len(label_order)} (control + {len(unique_labels)} perturbations)")
    
    return output_path


def _write_sorted_dense(
    source_path: Path,
    output_path: Path,
    sort_indices: np.ndarray,
    obs_sorted: pd.DataFrame,
    var: pd.DataFrame,
    uns: dict,
    chunk_size: int,
) -> None:
    """Write sorted file for dense input matrix.
    
    Uses chunked reading to avoid loading full matrix into memory.
    Writes h5ad in two passes: first X matrix via h5py, then metadata via anndata.
    """
    backed = read_backed(source_path)
    try:
        n_cells = backed.n_obs
        n_genes = backed.n_vars
        
        # Get dtype from first chunk
        sample = backed.X[:min(100, n_cells), :]
        dtype = sample.dtype
        
        # Create output with chunked writing for memory efficiency
        with h5py.File(output_path, 'w') as f:
            # Create dataset with chunking for efficient access
            X_out = f.create_dataset(
                'X',
                shape=(n_cells, n_genes),
                dtype=dtype,
                chunks=(min(chunk_size, n_cells), n_genes),
            )
            
            # Write in chunks based on output order
            for start in range(0, n_cells, chunk_size):
                end = min(start + chunk_size, n_cells)
                # Get the original indices for this output chunk
                chunk_indices = sort_indices[start:end]
                
                # Read from source (optimize by sorting indices for sequential read)
                read_order = np.argsort(chunk_indices)
                sorted_chunk_indices = chunk_indices[read_order]
                
                # Read data in optimized order
                chunk_data = backed.X[sorted_chunk_indices, :]
                
                # Reorder to match output order
                inverse_order = np.argsort(read_order)
                chunk_data = chunk_data[inverse_order]
                
                X_out[start:end, :] = chunk_data
                
                if (start // chunk_size) % 100 == 0:
                    logger.debug(f"  Written {end:,}/{n_cells:,} cells...")
        
    finally:
        backed.file.close()
    
    # Write metadata using anndata (proper h5ad structure)
    # First create a temp file with correct metadata structure
    temp_path = output_path.with_suffix('.meta.h5ad')
    adata_meta = ad.AnnData(
        X=sp.csr_matrix((len(obs_sorted), len(var)), dtype=np.float32),  # Placeholder
        obs=obs_sorted,
        var=var,
        uns=uns,
    )
    adata_meta.write(temp_path)
    
    # Copy metadata from temp to main file  
    with h5py.File(temp_path, 'r') as src:
        with h5py.File(output_path, 'a') as dst:
            for key in ['obs', 'var', 'uns']:
                if key in src:
                    if key in dst:
                        del dst[key]
                    src.copy(key, dst)
    
    # Cleanup temp file
    temp_path.unlink()



def _write_sorted_sparse(
    source_path: Path,
    output_path: Path,
    sort_indices: np.ndarray,
    obs_sorted: pd.DataFrame,
    var: pd.DataFrame,
    uns: dict,
    chunk_size: int,
) -> None:
    """Write sorted file for sparse input matrix."""
    # For sparse, we need to read and reorder
    # This is memory-intensive for very large datasets
    
    backed = read_backed(source_path)
    try:
        n_cells = backed.n_obs
        
        # Read full sparse matrix (required for reordering)
        logger.info("  Loading sparse matrix for reordering...")
        X_sparse = backed.X[:]
        if sp.issparse(X_sparse):
            X_sparse = sp.csr_matrix(X_sparse)
        else:
            X_sparse = sp.csr_matrix(X_sparse)
    finally:
        backed.file.close()
    
    # Reorder
    X_sorted = X_sparse[sort_indices, :]
    
    # Create AnnData and write
    adata_sorted = ad.AnnData(
        X=X_sorted,
        obs=obs_sorted,
        var=var,
        uns=uns,
    )
    adata_sorted.write(output_path)


def get_perturbation_slice(
    adata_or_path: str | Path | ad.AnnData,
    perturbation_label: str,
    perturbation_column: str = "perturbation",
) -> tuple[slice | None, bool]:
    """Get slice for a perturbation's cells, checking if file is sorted.
    
    If the file is sorted by perturbation, returns a contiguous slice.
    Otherwise, returns None for the slice (caller should use boolean mask).
    
    Parameters
    ----------
    adata_or_path
        Path to h5ad file, or an already-opened AnnData object.
    perturbation_label
        Label of the perturbation to get slice for.
    perturbation_column
        Column containing perturbation labels.
    
    Returns
    -------
    tuple[slice | None, bool]
        (slice object or None, is_sorted flag).
        If is_sorted is True, slice is valid for contiguous access.
        If is_sorted is False, slice is None and caller should use mask.
    """
    # Handle both path and AnnData input
    if isinstance(adata_or_path, ad.AnnData):
        adata = adata_or_path
        should_close = False
    else:
        adata = read_backed(adata_or_path)
        should_close = True
    
    try:
        # Check for sorting metadata
        if "sorting_metadata" in adata.uns:
            metadata = adata.uns["sorting_metadata"]
            if metadata.get("sorted_by") == perturbation_column:
                boundaries = metadata.get("perturbation_boundaries", {})
                if perturbation_label in boundaries:
                    start, end = boundaries[perturbation_label]
                    return slice(start, end), True
        
        return None, False
    finally:
        if should_close:
            adata.file.close()
