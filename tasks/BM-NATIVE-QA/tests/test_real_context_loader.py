from __future__ import annotations

from pathlib import Path

import pytest
from bm_native_qa_harness import bundle
from bm_native_qa_harness.bundle import _read_ordered_samples


def test_ordered_samples_are_bound_to_exact_fam_source_rows(tmp_path: Path) -> None:
    path = tmp_path / "samples.tsv"
    path.write_text(
        "sample_id\tsource_order\tnote\nS1\t1\ta\nS3\t3\tb\nS4\t4\tc\n",
        encoding="utf-8",
    )

    samples, indices = _read_ordered_samples(
        path,
        reference_samples=("S0", "S1", "S2", "S3", "S4"),
        expected_count=3,
    )

    assert samples == ("S1", "S3", "S4")
    assert indices.tolist() == [1, 3, 4]


def test_ordered_samples_map_filtered_cohort_order_to_physical_fam_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "samples.tsv"
    path.write_text(
        "sample_id\tsource_order\nS1\t1\nS3\t3\n",
        encoding="utf-8",
    )

    samples, indices = _read_ordered_samples(
        path,
        reference_samples=("X0", "S0", "X1", "S1", "S2", "X2", "S3"),
        canonical_samples=("S0", "S1", "S2", "S3"),
        expected_count=2,
    )

    assert samples == ("S1", "S3")
    assert indices.tolist() == [3, 6]


def test_reviewed_family_derives_canonical_id_when_input_uses_triad_id(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wheat-groups.tsv"
    path.write_text(
        "triad_id\tgene_A\tgene_B\tgene_D\n"
        "legacy-label\tA_gene\tB_gene\tD_gene\n",
        encoding="utf-8",
    )

    family = bundle._load_reviewed_family(
        path,
        ("A", "B", "D"),
        ("A_gene|B_gene|D_gene",),
    )

    assert family.group_ids == ("A_gene|B_gene|D_gene",)
    assert family.genes == (("A_gene", "B_gene", "D_gene"),)


def test_ordered_samples_reject_mismatched_or_unsorted_source_rows(
    tmp_path: Path,
) -> None:
    mismatch = tmp_path / "mismatch.tsv"
    mismatch.write_text(
        "sample_id\tsource_order\nWRONG\t1\nS3\t3\nS4\t4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differ from source sample manifest"):
        _read_ordered_samples(
            mismatch,
            reference_samples=("S0", "S1", "S2", "S3", "S4"),
            expected_count=3,
        )

    unsorted = tmp_path / "unsorted.tsv"
    unsorted.write_text(
        "sample_id\tsource_order\nS3\t3\nS1\t1\nS4\t4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact ordered"):
        _read_ordered_samples(
            unsorted,
            reference_samples=("S0", "S1", "S2", "S3", "S4"),
            expected_count=3,
        )
