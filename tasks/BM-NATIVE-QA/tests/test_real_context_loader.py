from __future__ import annotations

from pathlib import Path

import pytest
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


def test_ordered_samples_reject_mismatched_or_unsorted_source_rows(
    tmp_path: Path,
) -> None:
    mismatch = tmp_path / "mismatch.tsv"
    mismatch.write_text(
        "sample_id\tsource_order\nWRONG\t1\nS3\t3\nS4\t4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differ from FAM"):
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
