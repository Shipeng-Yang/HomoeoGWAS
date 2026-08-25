"""Phenotype-independent canonical homoeolog group and edge families."""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MasterGroupFamily:
    subgenomes: tuple[str, ...]
    group_ids: tuple[str, ...]
    genes: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        subgenomes = tuple(self.subgenomes)
        group_ids = tuple(self.group_ids)
        genes = tuple(tuple(row) for row in self.genes)
        object.__setattr__(self, "subgenomes", subgenomes)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "genes", genes)
        if len(subgenomes) < 2 or len(set(subgenomes)) != len(subgenomes):
            raise ValueError("subgenomes must contain at least two unique labels")
        if any(not isinstance(group_id, str) for group_id in group_ids):
            raise ValueError("group_id values must be strings")
        if any(not isinstance(gene, str) for row in genes for gene in row):
            raise ValueError("gene identifiers must be strings")
        if len(group_ids) != len(genes):
            raise ValueError("group_ids and genes must have the same row count")
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("group_id values must be unique")
        if any(len(row) != len(subgenomes) for row in genes):
            raise ValueError("every group must contain exactly one gene per subgenome")


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: str
    direction: str
    sub_x: str
    sub_y: str
    gene_x: str
    gene_y: str
    source_group_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExpandedEdgeFamily:
    edges: tuple[EdgeRecord, ...]
    group_edge_indices: tuple[tuple[int, ...], ...]


def expand_pair_edges(family: MasterGroupFamily) -> ExpandedEdgeFamily:
    """Expand each master group into unique, deterministically ordered pairs."""
    edges: list[EdgeRecord] = []
    group_edge_indices: list[tuple[int, ...]] = []
    by_key: dict[tuple[str, str, str, str], int] = {}
    for group_id, row in zip(family.group_ids, family.genes):
        indices: list[int] = []
        for x, y in itertools.combinations(range(len(family.subgenomes)), 2):
            sub_x, sub_y = family.subgenomes[x], family.subgenomes[y]
            gene_x, gene_y = row[x], row[y]
            key = (sub_x, sub_y, gene_x, gene_y)
            index = by_key.get(key)
            if index is None:
                index = len(edges)
                by_key[key] = index
                edges.append(EdgeRecord(
                    edge_id=f"{sub_x}{sub_y}:{gene_x}:{gene_y}",
                    direction=f"{sub_x}{sub_y}",
                    sub_x=sub_x, sub_y=sub_y, gene_x=gene_x, gene_y=gene_y,
                    source_group_ids=(group_id,),
                ))
            else:
                old = edges[index]
                if group_id not in old.source_group_ids:
                    edges[index] = EdgeRecord(
                        edge_id=old.edge_id, direction=old.direction,
                        sub_x=old.sub_x, sub_y=old.sub_y,
                        gene_x=old.gene_x, gene_y=old.gene_y,
                        source_group_ids=old.source_group_ids + (group_id,),
                    )
            indices.append(index)
        group_edge_indices.append(tuple(indices))
    return ExpandedEdgeFamily(tuple(edges), tuple(group_edge_indices))


def load_master_group_family(
    path: str | Path,
    subgenomes: Iterable[str],
    require_group_id: bool = False,
) -> MasterGroupFamily:
    """Read a TSV/CSV master-group table.

    Gene columns are named ``gene_<subgenome>``.  If no ``group_id`` column is
    present, IDs are derived from the ordered row genes joined by ``|``.
    """
    labels = tuple(subgenomes)
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
        except csv.Error:
            dialect = csv.excel_tab
        reader = csv.DictReader(handle, dialect=dialect)
        fields = tuple(reader.fieldnames or ())
        required = tuple(f"gene_{label}" for label in labels)
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        has_group_id = "group_id" in fields
        if require_group_id and not has_group_id:
            raise ValueError("missing group_id column")
        group_ids: list[str] = []
        genes: list[tuple[str, ...]] = []
        for row_number, row in enumerate(reader, start=2):
            values = tuple((row.get(f"gene_{label}") or "").strip() for label in labels)
            if any(not value for value in values):
                raise ValueError(f"missing gene value on row {row_number}")
            group_id = (row.get("group_id") or "").strip() if has_group_id else "|".join(values)
            if not group_id:
                raise ValueError(f"missing group_id on row {row_number}")
            group_ids.append(group_id)
            genes.append(values)
    return MasterGroupFamily(labels, tuple(group_ids), tuple(genes))
