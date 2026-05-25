"""Tests for Phase 5a species YAML schema + validator."""
from __future__ import annotations

from pathlib import Path

import pytest

from homoeogwas.species_config import (
    GenoSource,
    SpeciesConfig,
    Subgenome,
    load_species_config,
)

ROOT = Path(__file__).resolve().parents[1]
SPECIES_DIR = ROOT / "configs" / "species"


# ---------------------------------------------------------------------------
# 1. Pydantic schema sanity (no I/O)
# ---------------------------------------------------------------------------


def _minimal_kwargs(tmp_path: Path) -> dict:
    """Build a SpeciesConfig kwargs dict pointing at tmp_path so model_validate
    succeeds without requiring real data files."""
    fasta = tmp_path / "ref.fa"; fasta.touch()
    gff = tmp_path / "ref.gff"; gff.touch()
    chrom_map = tmp_path / "chrom_map.tsv"; chrom_map.write_text("panel_chrom\tfasta_chrom\tsubgenome\nchr1A\tchr1A\tA\n")
    bed_root = tmp_path / "bed"
    (bed_root / "A").mkdir(parents=True)
    (bed_root / "A" / "all.bim").write_text("")
    pheno = tmp_path / "pheno.tsv"; pheno.write_text("sample_id\ttrait1\n")
    return dict(
        id="testsp", latin="Test species", common_name="Test",
        genome_type="allopolyploid", ploidy=4,
        subgenomes=[
            Subgenome(id="A", chroms=["chr1A"]),
            Subgenome(id="B", chroms=["chr1B"]),
        ],
        fasta=fasta, gff=gff, chrom_map=chrom_map,
        geno=GenoSource(bed_root=bed_root), pheno=pheno,
    )


def test_minimal_construction(tmp_path: Path):
    cfg = SpeciesConfig(**_minimal_kwargs(tmp_path))
    assert cfg.id == "testsp"
    assert cfg.dosage_model == "diploidized"
    assert cfg.loco_unit == "chrom"
    assert cfg.dl_model_window == 6144
    assert cfg.dl_candidate_window == 100_000


def test_diploid_must_have_one_subgenome(tmp_path: Path):
    kw = _minimal_kwargs(tmp_path)
    kw["genome_type"] = "diploid"
    kw["ploidy"] = 2
    # diploid + 2 subgenomes should fail
    with pytest.raises(Exception):
        SpeciesConfig(**kw)
    # diploid + 1 subgenome should pass
    kw["subgenomes"] = [Subgenome(id="A", chroms=["chr1"])]
    cfg = SpeciesConfig(**kw)
    assert cfg.genome_type == "diploid"


def test_allopolyploid_must_have_ge_two_subgenomes(tmp_path: Path):
    kw = _minimal_kwargs(tmp_path)
    kw["subgenomes"] = [Subgenome(id="A", chroms=["chr1A"])]
    with pytest.raises(Exception):
        SpeciesConfig(**kw)


def test_subgenome_id_must_be_unique(tmp_path: Path):
    kw = _minimal_kwargs(tmp_path)
    kw["subgenomes"] = [
        Subgenome(id="A", chroms=["chr1A"]),
        Subgenome(id="A", chroms=["chr1B"]),
    ]
    with pytest.raises(Exception):
        SpeciesConfig(**kw)


def test_chrom_appearing_in_two_subgenomes_rejected(tmp_path: Path):
    kw = _minimal_kwargs(tmp_path)
    kw["subgenomes"] = [
        Subgenome(id="A", chroms=["chr1"]),
        Subgenome(id="B", chroms=["chr1"]),
    ]
    with pytest.raises(Exception):
        SpeciesConfig(**kw)


def test_geno_requires_exactly_one_source(tmp_path: Path):
    with pytest.raises(Exception):
        GenoSource()  # neither vcf nor bed_root
    with pytest.raises(Exception):
        GenoSource(vcf=tmp_path / "x.vcf", bed_root=tmp_path / "bed")


def test_extra_top_level_field_rejected(tmp_path: Path):
    kw = _minimal_kwargs(tmp_path)
    kw["nonexistent"] = "x"
    with pytest.raises(Exception):
        SpeciesConfig(**kw)


# ---------------------------------------------------------------------------
# 2. Real YAML files for the 3 hardcoded species — schema-only checks
#    (don't run the cross-file validator since reference fasta is on /mnt/nvme
#    and may not exist on every machine running pytest)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaml_name",
    ["wheat_aestivum.yaml", "gossypium_hirsutum.yaml", "brassica_napus.yaml",
     "fragaria_ananassa.yaml"],
)
def test_existing_species_yaml_parses(yaml_name: str):
    yaml_path = SPECIES_DIR / yaml_name
    assert yaml_path.exists(), f"missing {yaml_path}"
    cfg = load_species_config(yaml_path, validate=False)
    assert cfg.genome_type == "allopolyploid"
    assert len(cfg.subgenomes) >= 2
    assert cfg.ploidy in (4, 6, 8)


def test_wheat_yaml_has_three_subgenomes():
    cfg = load_species_config(SPECIES_DIR / "wheat_aestivum.yaml", validate=False)
    assert {sg.id for sg in cfg.subgenomes} == {"A", "B", "D"}
    assert cfg.ploidy == 6
    # 21 chrom total
    total = sum(len(sg.chroms) for sg in cfg.subgenomes)
    assert total == 21


def test_cotton_yaml_has_two_subgenomes_26_chroms():
    cfg = load_species_config(SPECIES_DIR / "gossypium_hirsutum.yaml", validate=False)
    assert {sg.id for sg in cfg.subgenomes} == {"A", "D"}
    total = sum(len(sg.chroms) for sg in cfg.subgenomes)
    assert total == 26


def test_rapeseed_yaml_has_two_subgenomes_19_chroms():
    cfg = load_species_config(SPECIES_DIR / "brassica_napus.yaml", validate=False)
    assert {sg.id for sg in cfg.subgenomes} == {"A", "C"}
    total = sum(len(sg.chroms) for sg in cfg.subgenomes)
    assert total == 19


def test_donor_metadata_carried_through():
    cfg = load_species_config(SPECIES_DIR / "wheat_aestivum.yaml", validate=False)
    donors = {sg.id: sg.donor for sg in cfg.subgenomes}
    assert "Aegilops tauschii" in donors.values()  # D-donor


def test_strawberry_octoploid_has_four_sub_28_chrom():
    """Phase 5b Week 1: framework universality smoke test for strawberry 8n."""
    cfg = load_species_config(SPECIES_DIR / "fragaria_ananassa.yaml", validate=False)
    assert cfg.ploidy == 8
    assert {sg.id for sg in cfg.subgenomes} == {"A", "B", "C", "D"}
    total_chroms = sum(len(sg.chroms) for sg in cfg.subgenomes)
    assert total_chroms == 28
    # Each subgenome contributes 7 chrom
    for sg in cfg.subgenomes:
        assert len(sg.chroms) == 7


def test_strawberry_kernel_auto_picks_pairwise_mean():
    """Phase 5a Step 3 K_hom auto-fallback decision for n_sub=4 = pairwise_mean."""
    import numpy as np
    from homoeogwas.kernel import build_homoeolog_kernel

    cfg = load_species_config(SPECIES_DIR / "fragaria_ananassa.yaml", validate=False)
    # Dummy GRMs one per subgenome (small n for speed)
    rng = np.random.default_rng(0)
    grms = {}
    for sg in cfg.subgenomes:
        Xg = rng.standard_normal((10, 30))
        K = Xg @ Xg.T / 30
        K = 0.5 * (K + K.T)
        grms[sg.id] = K
    K_hom, mode = build_homoeolog_kernel(grms, mode="auto")
    assert mode == "pairwise_mean"
    assert K_hom is not None
    assert K_hom.shape == (10, 10)
