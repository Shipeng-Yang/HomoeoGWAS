"""Contracts for non-SNP biallelic marker inputs.

HomoeoGWAS' statistical engine consumes PLINK BED hard calls.  This module
makes the biological meaning of those columns explicit so a PAV, biallelic SV
or haplotype pseudo-marker is not silently presented as an ordinary SNP.

Only encodings that remain valid hard-call allele counts are accepted.  Raw
integer/continuous copy number, multi-allelic SVs and graph paths must first be
converted into defensible biallelic features or analysed with another model.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .io import plink_path, read_delimited

DEFAULT_ENCODING = "diploid_0_1_2"
ALLOWED_ENCODINGS = {
    DEFAULT_ENCODING,
    "binary_presence_0_2",
    "haplotype_dosage_0_1_2",
    "mixed_biallelic_0_1_2",
}
ALLOWED_MARKER_TYPES = {"SNP", "INDEL", "SV", "PAV", "HAPLOTYPE"}


def _manifest_path(template: str, subgenome: str) -> Path:
    try:
        rendered = template.format(subgenome=subgenome)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            "genotype.marker_manifest_template must use only the "
            "{subgenome} placeholder"
        ) from exc
    return Path(rendered)


def _read_bim_ids(prefix: str | Path) -> list[str]:
    path = plink_path(prefix, ".bim")
    ids: list[str] = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_no} has fewer than 2 BIM fields")
            ids.append(fields[1])
    return ids


def check_binary_presence_values(
    prefix: str | Path, *, chunk_size: int = 100_000
) -> None:
    """Require every observed BED hard call to be 0 or 2.

    Encoding presence as 0/2 preserves the standard diploid allele-frequency
    and VanRaden denominator used by the existing scan/GRM implementation.
    A 0/1 matrix would otherwise be misinterpreted as heterozygous dosage.
    """
    from bed_reader import open_bed

    bed_path = plink_path(prefix, ".bed")
    with open_bed(str(bed_path), count_A1=True) as bed:
        for start in range(0, int(bed.sid_count), chunk_size):
            end = min(start + chunk_size, int(bed.sid_count))
            values = bed.read(
                index=(slice(None), slice(start, end)), dtype="float32"
            )
            observed = values[np.isfinite(values)]
            bad = observed[(observed != 0.0) & (observed != 2.0)]
            if bad.size:
                example = sorted({float(v) for v in bad[:20]})
                raise ValueError(
                    f"{bed_path} declares binary_presence_0_2 but contains "
                    f"non-missing values outside {{0,2}} (e.g. {example}); "
                    "recode absence/presence to 0/2 before analysis"
                )


def validate_marker_contract(
    *,
    subgenomes: Sequence[str],
    bed_prefix: Callable[[str], str | Path],
    encoding: str = DEFAULT_ENCODING,
    manifest_template: str | None = None,
    check_values: bool = True,
    contract_label: str = "genotype",
) -> tuple[list[str], dict]:
    """Validate marker manifests against exact BIM IDs.

    Returns ``(problems, provenance)`` so CLI preflight can aggregate all
    actionable failures instead of stopping at the first subgenome.
    """
    problems: list[str] = []
    provenance = {
        "encoding": encoding,
        "manifest_template": manifest_template,
        "subgenomes": {},
        "scope": (
            "biallelic hard-call features only; raw/multi-allelic CNV is unsupported"
        ),
    }
    if encoding not in ALLOWED_ENCODINGS:
        return [
            f"{contract_label}.marker_encoding must be one of "
            f"{sorted(ALLOWED_ENCODINGS)}, "
            f"got {encoding!r}"
        ], provenance
    if encoding != DEFAULT_ENCODING and not manifest_template:
        return [
            f"non-default {contract_label}.marker_encoding requires "
            f"{contract_label}.marker_manifest_template"
        ], provenance

    for subgenome in subgenomes:
        prefix = bed_prefix(subgenome)
        item: dict = {"bed_prefix": str(prefix)}
        provenance["subgenomes"][subgenome] = item
        if not manifest_template:
            item["manifest"] = None
            continue
        path = _manifest_path(manifest_template, subgenome)
        item["manifest"] = str(path)
        if not path.exists():
            problems.append(f"marker manifest missing for subgenome {subgenome}: {path}")
            continue
        try:
            table = read_delimited(path)
        except Exception as exc:  # noqa: BLE001 - actionable aggregate
            problems.append(f"cannot read marker manifest {path}: {exc}")
            continue
        required = {"variant_id", "marker_type"}
        missing = sorted(required - set(table.columns))
        if missing:
            problems.append(f"marker manifest {path} missing columns {missing}")
            continue
        table = table.copy()
        table["variant_id"] = table["variant_id"].astype(str)
        table["marker_type"] = table["marker_type"].astype(str).str.upper()
        duplicates = table.loc[
            table["variant_id"].duplicated(keep=False), "variant_id"
        ].unique()
        if len(duplicates):
            problems.append(
                f"marker manifest {path} has duplicate variant_id values "
                f"(e.g. {duplicates[:3].tolist()})"
            )
            continue
        invalid_types = sorted(set(table["marker_type"]) - ALLOWED_MARKER_TYPES)
        if invalid_types:
            problems.append(
                f"marker manifest {path} has unsupported marker_type values "
                f"{invalid_types}; allowed={sorted(ALLOWED_MARKER_TYPES)}"
            )
        try:
            bim_ids = _read_bim_ids(prefix)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"cannot read BIM for marker manifest {path}: {exc}")
            continue
        if len(set(bim_ids)) != len(bim_ids):
            problems.append(f"BIM for subgenome {subgenome} has duplicate variant IDs")
            continue
        manifest_ids = table["variant_id"].tolist()
        missing_ids = sorted(set(bim_ids) - set(manifest_ids))
        extra_ids = sorted(set(manifest_ids) - set(bim_ids))
        if missing_ids or extra_ids:
            problems.append(
                f"marker manifest {path} is not an exact BIM-ID manifest: "
                f"{len(missing_ids)} BIM IDs missing, {len(extra_ids)} extra "
                f"(examples missing={missing_ids[:3]}, extra={extra_ids[:3]})"
            )
        elif manifest_ids != bim_ids:
            problems.append(
                f"marker manifest {path} has the same IDs as the BIM but in a "
                "different order; manifest rows must follow BIM order exactly"
            )
        counts = table["marker_type"].value_counts().sort_index()
        item.update(
            {
                "n_variants": len(bim_ids),
                "marker_type_counts": {str(k): int(v) for k, v in counts.items()},
            }
        )
        types = set(table["marker_type"])
        if encoding == "binary_presence_0_2" and not types <= {"PAV", "SV"}:
            problems.append(
                f"binary_presence_0_2 manifest {path} may contain only PAV/SV, "
                f"found {sorted(types)}"
            )
        if encoding == "haplotype_dosage_0_1_2" and types != {"HAPLOTYPE"}:
            problems.append(
                f"haplotype_dosage_0_1_2 manifest {path} must contain only "
                f"HAPLOTYPE markers, found {sorted(types)}"
            )
        if encoding == "binary_presence_0_2" and check_values:
            try:
                check_binary_presence_values(prefix)
                item["binary_value_check"] = "PASS"
            except Exception as exc:  # noqa: BLE001
                item["binary_value_check"] = "FAIL"
                problems.append(str(exc))
    return problems, provenance
