#!/usr/bin/env python3
"""Prepare (but never execute) the frozen wheat F2143 edge-primary run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from homoeogwas import __version__
from homoeogwas.group_family import (
    MasterGroupFamily,
    expand_pair_edges,
)
from homoeogwas.interact import (
    _validate_snp_mapping,
    preflight_interact,
    validate_interact_config,
)
from homoeogwas.io import plink_path

SUBGENOMES = ("A", "B", "D")
SAMPLE_COL = "sample"
TRAIT = "days_to_emerg_env_adjusted"
FROZEN_GROUP_SHA256 = (
    "4a905dab9aba53b639958aa4a540b7e5494ef61cce35c9c2f6409e83394684dc"
)
FROZEN_PHENOTYPE_SHA256 = (
    "908ebabbd0a8e669d8ab6bee3758c2f91f2ed17d7525bca5f3f11da66534aebc"
)
PRODUCTION_GROUP_COUNT = 2143
PRODUCTION_SAMPLE_COUNT = 827
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 2026
CHECKPOINT_BLOCK_SIZE = 25
MANIFEST_SCHEMA = "homoeogwas-wheat-f2143-edge-pre-run-manifest-v1"
CONFIG_SCHEMA = "homoeogwas-canonical-group-omnib-v1"

GENOTYPE_PREFIXES = {
    "A": "/mnt/nvme/wheat_genomewide/gw_A_flank",
    "B": "/mnt/nvme/wheat_genomewide/gw_B_flank",
    "D": "/mnt/nvme/wheat_genomewide/gw_D_flank",
}
SNP_TO_GENE = {
    "A": "data/processed/wheat/scanT/verified_mapping/snp_to_gene_A.npz",
    "B": "data/processed/wheat/scanT/verified_mapping/snp_to_gene_B.npz",
    "D": "data/processed/wheat/scanT/verified_mapping/snp_to_gene_D.npz",
}


@dataclass(frozen=True)
class PreparedWheatRun:
    groups: str
    phenotype: str
    config: str
    manifest: str
    n_groups: int
    n_unique_edges: int
    n_samples: int


def _sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_write_bytes(path: str | Path, body: bytes) -> None:
    """Durably replace *path* from a same-directory, fsynced temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _verified_source_bytes(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> tuple[Path, bytes, str]:
    source = Path(path).resolve(strict=True)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in expected_sha256.lower())
    ):
        raise ValueError(f"{label} expected SHA-256 must be 64 hexadecimal characters")
    body = source.read_bytes()
    observed = _sha256_bytes(body)
    if observed != expected_sha256.lower():
        raise ValueError(
            f"{label} source SHA-256 mismatch: expected {expected_sha256.lower()}, "
            f"observed {observed} for {source}"
        )
    return source, body, observed


def _read_tsv(body: bytes, label: str) -> csv.DictReader:
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 TSV text") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if not reader.fieldnames:
        raise ValueError(f"{label} is missing a header")
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError(f"{label} contains duplicate column names")
    return reader


def _parse_family(body: bytes, *, production: bool) -> tuple[MasterGroupFamily, str]:
    reader = _read_tsv(body, "groups source")
    fields = tuple(reader.fieldnames or ())
    required = tuple(f"gene_{subgenome}" for subgenome in SUBGENOMES)
    missing = [field for field in required if field not in fields]
    if missing:
        raise ValueError(
            "missing required group columns: " + ", ".join(missing)
        )
    present_ids = [field for field in ("triad_id", "group_id") if field in fields]
    if len(present_ids) != 1:
        raise ValueError(
            "groups source must contain exactly one ID column: triad_id or group_id"
        )
    id_column = present_ids[0]
    if production and id_column != "triad_id":
        raise ValueError("production F2143 groups must retain source ID column triad_id")

    group_ids: list[str] = []
    genes: list[tuple[str, ...]] = []
    seen_ids: set[str] = set()
    seen_groups: set[tuple[str, ...]] = set()
    for row_number, row in enumerate(reader, start=2):
        group_id = row.get(id_column)
        ordered_genes = tuple(row.get(f"gene_{subgenome}") for subgenome in SUBGENOMES)
        values = (group_id, *ordered_genes)
        if any(value is None or not value.strip() for value in values):
            raise ValueError(
                f"blank group identifier or gene on groups row {row_number}"
            )
        if any(value != value.strip() for value in values):
            raise ValueError(
                f"leading/trailing whitespace in group identifiers on row {row_number}"
            )
        assert group_id is not None
        genes_row = tuple(value for value in ordered_genes if value is not None)
        if group_id in seen_ids:
            raise ValueError(f"duplicate group ID {group_id!r} on row {row_number}")
        if genes_row in seen_groups:
            raise ValueError(
                f"duplicate biological group {genes_row!r} on row {row_number}"
            )
        if id_column == "triad_id" and group_id != "|".join(genes_row):
            raise ValueError(
                "triad_id does not equal its ordered A/B/D gene identity on "
                f"row {row_number}; canonical parsing would not preserve it"
            )
        seen_ids.add(group_id)
        seen_groups.add(genes_row)
        group_ids.append(group_id)
        genes.append(genes_row)
    if not group_ids:
        raise ValueError("groups source contains no data rows")
    return MasterGroupFamily(SUBGENOMES, tuple(group_ids), tuple(genes)), id_column


def _parse_sample_ids(body: bytes) -> tuple[str, ...]:
    reader = _read_tsv(body, "phenotype source")
    fields = tuple(reader.fieldnames or ())
    missing = [field for field in (SAMPLE_COL, TRAIT) if field not in fields]
    if missing:
        raise ValueError(
            "phenotype source is missing required columns: " + ", ".join(missing)
        )
    samples: list[str] = []
    seen: set[str] = set()
    missing_tokens = {"na", "nan", "null", "none"}
    for row_number, row in enumerate(reader, start=2):
        sample = row.get(SAMPLE_COL)
        if sample is None or not sample.strip() or sample.strip().lower() in missing_tokens:
            raise ValueError(f"blank or missing sample ID on phenotype row {row_number}")
        if sample != sample.strip():
            raise ValueError(
                f"leading/trailing whitespace in sample ID on phenotype row {row_number}"
            )
        if sample in seen:
            raise ValueError(f"duplicate sample ID {sample!r} on phenotype row {row_number}")
        seen.add(sample)
        samples.append(sample)
    if not samples:
        raise ValueError("phenotype source contains no samples")
    return tuple(samples)


def _joined_sha256(values: Iterable[str]) -> str:
    return _sha256_bytes("\0".join(values).encode("utf-8"))


def _family_identity(family: MasterGroupFamily) -> tuple[Any, dict[str, Any]]:
    expanded = expand_pair_edges(family)
    group_records = [
        "\t".join((group_id, *genes))
        for group_id, genes in zip(family.group_ids, family.genes, strict=True)
    ]
    edge_records = [
        "\t".join((
            edge.edge_id,
            edge.sub_x,
            edge.sub_y,
            edge.gene_x,
            edge.gene_y,
            *edge.source_group_ids,
        ))
        for edge in expanded.edges
    ]
    direction_counts = Counter(edge.direction for edge in expanded.edges)
    identity = {
        "n_groups": len(family.group_ids),
        "expanded_edge_rows": sum(len(indices) for indices in expanded.group_edge_indices),
        "n_unique_edges": len(expanded.edges),
        "direction_counts": {
            direction: int(direction_counts.get(direction, 0))
            for direction in ("AB", "AD", "BD")
        },
        "group_family_sha256": _joined_sha256(group_records),
        "edge_family_sha256": _joined_sha256(edge_records),
        "ordered_group_ids_sha256": _joined_sha256(family.group_ids),
        "ordered_edge_ids_sha256": _joined_sha256(
            edge.edge_id for edge in expanded.edges
        ),
    }
    return expanded, identity


def _read_fam_ids(prefix: str | Path) -> tuple[str, ...]:
    fam = plink_path(prefix, ".fam").resolve(strict=True)
    sample_ids: list[str] = []
    with fam.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) < 2 or not fields[1]:
                raise ValueError(f"invalid PLINK FAM sample ID on {fam}:{row_number}")
            sample_ids.append(str(fields[1]))
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"duplicate sample IDs in PLINK FAM {fam}")
    return tuple(sample_ids)


def _resource_identities(
    phenotype_samples: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, str], dict[str, Any], tuple[str, ...]]:
    genotype: dict[str, str] = {}
    mappings: dict[str, str] = {}
    identities: dict[str, Any] = {"genotype": {}, "snp_to_gene": {}}
    sample_orders: dict[str, tuple[str, ...]] = {}
    for subgenome in SUBGENOMES:
        prefix = str(Path(GENOTYPE_PREFIXES[subgenome]).resolve())
        mapping = str(Path(SNP_TO_GENE[subgenome]).resolve(strict=True))
        for extension in (".bed", ".bim", ".fam"):
            plink_path(prefix, extension).resolve(strict=True)
        _validate_snp_mapping(prefix, mapping, expected_subgenome=subgenome)
        genotype[subgenome] = prefix
        mappings[subgenome] = mapping
        identities["genotype"][subgenome] = {
            "prefix": prefix,
            "bed": _file_identity(plink_path(prefix, ".bed")),
            "bim": _file_identity(plink_path(prefix, ".bim")),
            "fam": _file_identity(plink_path(prefix, ".fam")),
        }
        identities["snp_to_gene"][subgenome] = _file_identity(mapping)
        sample_orders[subgenome] = _read_fam_ids(prefix)

    first = sample_orders[SUBGENOMES[0]]
    for subgenome in SUBGENOMES[1:]:
        if sample_orders[subgenome] != first:
            raise ValueError(
                f"genotype sample order mismatch between A and {subgenome}"
            )
    phenotype_set = set(phenotype_samples)
    analysis_samples = tuple(sample for sample in first if sample in phenotype_set)
    missing = sorted(phenotype_set - set(analysis_samples))
    if missing:
        raise ValueError(
            f"phenotype has {len(missing)} sample IDs absent from genotype "
            f"(e.g. {missing[:3]})"
        )
    return genotype, mappings, identities, analysis_samples


def _build_config(
    *,
    out_dir: Path,
    groups: Path,
    phenotype: Path,
    genotype: dict[str, str],
    mappings: dict[str, str],
) -> dict[str, Any]:
    return {
        "interact": {
            "mode": "group",
            "subgenomes": list(SUBGENOMES),
            "groups": str(groups.resolve()),
            "statistic": "omniB",
            "hypothesis_unit": "edge",
            "subset_order": 2,
            "family_scope": "primary_only",
            "primary_transform": "INT",
            "primary_multiplicity": "bootstrap_minp",
            "genotype": genotype,
            "snp_to_gene": mappings,
            "phenotype": str(phenotype.resolve()),
            "sample_col": SAMPLE_COL,
            "trait": TRAIT,
            "burden": {"cap": 150, "min_snp": 3, "maf_min": 0.01, "n_pc": 3},
            "grm": {
                "method": "grm_from_X",
                "maf_min": 0.01,
                "scope": "all_subgenomes",
            },
            "calibration": {
                "method": "bootstrap",
                "B": BOOTSTRAP_B,
                "seed": BOOTSTRAP_SEED,
                "checkpoint": {
                    "enabled": True,
                    "root": str((out_dir / "checkpoints").resolve()),
                    "block_size": CHECKPOINT_BLOCK_SIZE,
                },
            },
        },
        "outputs": {"out_dir": str(out_dir.resolve()), "full_ranking": True},
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def prepare_wheat_inputs(
    source_groups,
    source_phenotype,
    out_dir,
    expected_group_sha256,
    expected_phenotype_sha256,
    production=False,
) -> PreparedWheatRun:
    """Create a hash-locked, validated run bundle without starting inference."""
    groups_source, groups_body, groups_sha256 = _verified_source_bytes(
        source_groups, expected_group_sha256, "groups"
    )
    phenotype_source, phenotype_body, phenotype_sha256 = _verified_source_bytes(
        source_phenotype, expected_phenotype_sha256, "phenotype"
    )
    if production:
        if expected_group_sha256.lower() != FROZEN_GROUP_SHA256:
            raise ValueError(
                "production expected group SHA-256 is not the frozen F2143 identity"
            )
        if expected_phenotype_sha256.lower() != FROZEN_PHENOTYPE_SHA256:
            raise ValueError(
                "production expected phenotype SHA-256 is not the frozen release identity"
            )

    family, id_column = _parse_family(groups_body, production=bool(production))
    phenotype_samples = _parse_sample_ids(phenotype_body)
    expanded, family_identity = _family_identity(family)
    if production and len(family.group_ids) != PRODUCTION_GROUP_COUNT:
        raise ValueError(
            "production requires exactly 2143 groups; "
            f"observed {len(family.group_ids)}"
        )
    if production and len(phenotype_samples) != PRODUCTION_SAMPLE_COUNT:
        raise ValueError(
            "production requires exactly 827 unique non-missing samples; "
            f"observed {len(phenotype_samples)}"
        )

    output = Path(out_dir).resolve()
    inputs_dir = output / "inputs"
    configs_dir = output / "configs"
    provenance_dir = output / "provenance"
    checkpoint_dir = output / "checkpoints"
    groups_copy = inputs_dir / "f2143.generated.tsv"
    phenotype_copy = inputs_dir / "phenotype.release.tsv"
    _atomic_write_bytes(groups_copy, groups_body)
    if _sha256_file(groups_copy) != groups_sha256:
        raise RuntimeError("durable groups copy SHA-256 mismatch after atomic copy")
    _atomic_write_bytes(phenotype_copy, phenotype_body)
    if _sha256_file(phenotype_copy) != phenotype_sha256:
        raise RuntimeError("durable phenotype copy SHA-256 mismatch after atomic copy")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _fsync_directory(output)

    genotype, mappings, resource_identities, analysis_samples = (
        _resource_identities(phenotype_samples)
    )
    if production and len(analysis_samples) != PRODUCTION_SAMPLE_COUNT:
        raise ValueError(
            "production genotype/phenotype intersection must contain exactly "
            f"827 samples; observed {len(analysis_samples)}"
        )

    config = _build_config(
        out_dir=output,
        groups=groups_copy,
        phenotype=phenotype_copy,
        genotype=genotype,
        mappings=mappings,
    )
    validate_interact_config(config)
    config_path = configs_dir / "interact.generated.group.omnib.yaml"
    config_bytes = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")
    _atomic_write_bytes(config_path, config_bytes)
    config_sha256 = _sha256_file(config_path)

    validation = {
        "schema": "passed",
        "mapping_fingerprints": "passed",
        "input_preflight": "not_run_nonproduction",
    }
    if production:
        problems = preflight_interact(config)
        if problems:
            raise RuntimeError(
                "generated production config failed input preflight: "
                + "; ".join(problems)
            )
        validation["input_preflight"] = "passed"

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "config_schema": CONFIG_SCHEMA,
        "software": {
            "name": "homoeogwas",
            "version": __version__,
            "preparation_script_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "production": bool(production),
        "source_group_id_column": id_column,
        "subgenomes": list(SUBGENOMES),
        "trait": TRAIT,
        "sample_col": SAMPLE_COL,
        "n_samples": len(phenotype_samples),
        "phenotype_sample_ids_sha256": _joined_sha256(phenotype_samples),
        "analysis_sample_ids_sha256": _joined_sha256(analysis_samples),
        **family_identity,
        "inputs": {
            "groups": {
                "expected_sha256": expected_group_sha256.lower(),
                "source": {
                    "path": str(groups_source),
                    "size_bytes": len(groups_body),
                    "sha256": groups_sha256,
                },
                "durable_copy": _file_identity(groups_copy),
            },
            "phenotype": {
                "expected_sha256": expected_phenotype_sha256.lower(),
                "source": {
                    "path": str(phenotype_source),
                    "size_bytes": len(phenotype_body),
                    "sha256": phenotype_sha256,
                },
                "durable_copy": _file_identity(phenotype_copy),
            },
            **resource_identities,
        },
        "hypothesis": {
            "unit": "edge",
            "family_scope": "primary_only",
            "subset_order": 2,
            "transform": "INT",
            "multiplicity": "bootstrap_minp",
        },
        "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED},
        "checkpoint": {
            "enabled": True,
            "root": str(checkpoint_dir),
            "block_size": CHECKPOINT_BLOCK_SIZE,
        },
        "config": {
            "path": str(config_path),
            "sha256": config_sha256,
        },
        "config_sha256": config_sha256,
        "validation": validation,
    }
    manifest_path = provenance_dir / "pre_run_manifest.json"
    _atomic_write_bytes(manifest_path, _canonical_json_bytes(manifest))
    return PreparedWheatRun(
        groups=str(groups_copy),
        phenotype=str(phenotype_copy),
        config=str(config_path),
        manifest=str(manifest_path),
        n_groups=len(family.group_ids),
        n_unique_edges=len(expanded.edges),
        n_samples=len(phenotype_samples),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and validate the frozen matched wheat F2143 edge-primary "
            "omniB bundle. This command never starts the B=2000 scan."
        )
    )
    parser.add_argument("--source-groups", required=True)
    parser.add_argument("--source-phenotype", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--expected-group-sha256", default=FROZEN_GROUP_SHA256
    )
    parser.add_argument(
        "--expected-phenotype-sha256", default=FROZEN_PHENOTYPE_SHA256
    )
    parser.add_argument(
        "--production",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enforce frozen 2,143-group/827-sample identities (default: true)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = prepare_wheat_inputs(
        source_groups=args.source_groups,
        source_phenotype=args.source_phenotype,
        out_dir=args.out_dir,
        expected_group_sha256=args.expected_group_sha256,
        expected_phenotype_sha256=args.expected_phenotype_sha256,
        production=args.production,
    )
    print(json.dumps({
        "status": "prepared_not_started",
        "n_groups": result.n_groups,
        "n_unique_edges": result.n_unique_edges,
        "n_samples": result.n_samples,
        "config": result.config,
        "manifest": result.manifest,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
