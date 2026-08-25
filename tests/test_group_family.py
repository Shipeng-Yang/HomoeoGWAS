import pytest

from homoeogwas.group_family import (
    MasterGroupFamily,
    expand_pair_edges,
    load_master_group_family,
)


def test_expand_three_copy_groups_is_deterministic_and_deduplicates_edges():
    fam = MasterGroupFamily(
        subgenomes=("A", "B", "D"),
        group_ids=("g2", "g1", "g3"),
        genes=(("a2", "b2", "d2"),
               ("a1", "b1", "d1"),
               ("a1", "b1", "d9")),
    )
    out = expand_pair_edges(fam)
    assert [e.direction for e in out.edges] == [
        "AB", "AD", "BD", "AB", "AD", "BD", "AD", "BD"
    ]
    assert out.edges[3].source_group_ids == ("g1", "g3")
    assert out.group_edge_indices[1][0] == out.group_edge_indices[2][0]


def test_two_copy_group_has_exactly_one_edge():
    fam = MasterGroupFamily(
        subgenomes=("A", "D"), group_ids=("g1",),
        genes=(("a1", "d1"),),
    )
    out = expand_pair_edges(fam)
    assert len(out.edges) == 1
    assert out.group_edge_indices == ((0,),)


def test_load_legacy_table_derives_stable_group_ids(tmp_path):
    path = tmp_path / "triads.tsv"
    path.write_text("gene_A\tgene_B\tgene_D\na1\tb1\td1\n")
    fam = load_master_group_family(path, ["A", "B", "D"])
    assert fam.group_ids == ("a1|b1|d1",)


@pytest.mark.parametrize("body, message", [
    ("group_id\tgene_A\tgene_B\ng1\ta1\n", "missing"),
    ("group_id\tgene_A\tgene_B\ng1\ta1\tb1\ng1\ta2\tb2\n", "unique"),
])
def test_load_group_table_rejects_invalid_schema(tmp_path, body, message):
    path = tmp_path / "groups.tsv"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        load_master_group_family(path, ["A", "B"], require_group_id=True)
