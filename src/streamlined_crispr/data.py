"""Utilities for reading and writing lightweight ``.h5ad``-like files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence


@dataclass
class SimpleAnnData:
    path: Path
    obs: List[Dict[str, str]]
    var: List[str]

    @property
    def n_obs(self) -> int:
        return len(self.obs)

    @property
    def n_vars(self) -> int:
        return len(self.var)

    def iter_rows(self, start: int = 0, end: int | None = None) -> Iterator[List[int]]:
        return iter_matrix_rows(self.path, start=start, end=end)


def write_h5ad(
    path: str | Path,
    matrix: Sequence[Sequence[int]],
    obs: Sequence[Dict[str, str]],
    var: Sequence[str],
) -> Path:
    """Write a minimal on-disk representation of expression data."""

    path = Path(path)
    header = {"obs": list(obs), "var": list(var)}
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        for row in matrix:
            handle.write(",".join(str(int(value)) for value in row) + "\n")
    return path


def read_h5ad(path: str | Path) -> SimpleAnnData:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
    return SimpleAnnData(path=path, obs=header["obs"], var=header["var"])


def iter_matrix_rows(
    path: str | Path,
    *,
    start: int = 0,
    end: int | None = None,
) -> Iterator[List[int]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        handle.readline()  # skip header
        for idx, line in enumerate(handle):
            if idx < start:
                continue
            if end is not None and idx >= end:
                break
            line = line.strip()
            if not line:
                continue
            yield [int(token) for token in line.split(",")]
