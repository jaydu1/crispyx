"""A lightweight subset of the :mod:`scanpy` API for offline testing."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import anndata as ad

from . import pp  # noqa: F401

PathLike = Union[str, Path]


def read_h5ad(filename: PathLike, *, backed: Union[bool, str] = False):
    """Read an AnnData object from disk.

    Parameters
    ----------
    filename:
        Path to the ``.h5ad`` file.
    backed:
        Present for API compatibility with :func:`scanpy.read_h5ad`. Only ``False``
        is supported; passing any other value raises :class:`NotImplementedError`.
    """

    if backed not in (False, None):
        raise NotImplementedError("backed mode is not supported in the test stub")
    return ad.read_h5ad(filename)


__all__ = ["read_h5ad", "pp"]
