"""Tests for Phase 5a Step 2: generalized subgenome split."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from homoeogwas.species_config import GenoSource, SpeciesConfig, Subgenome
from homoeogwas.species_split import (
    _route_per_chrom,
    cmd_split,
    execute_plan,
    plan_split,
)

# ---------------------------------------------------------------------------
# Tiny VCF + per_chrom dir fixtures (no external bcftools/plink2 required)
# ---------------------------------------------------------------------------

TINY_VCF_HEADER = """\
##fileformat=VCFv4.2
##contig=<ID=chr1A>
##contig=<ID=chr1B>
##contig=<ID=chr1D>
##INFO=<ID=NS,Number=1,Type=Integer,Description="n samples">
##FORMAT=<ID=GT,Number=1,Type=String,Description="genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3
"""


def _write_single_vcf(p: Path) -> Path:
    rows = [
        "chr1A\t100\t.\tA\tG\t.\tPASS\tNS=3\tGT\t0|0\t0|1\t1|1",
        "chr1B\t200\t.\tC\tT\t.\tPASS\tNS=3\tGT\t0|0\t1|1\t0|1",
        "chr1D\t300\t.\tG\tA\t.\tPASS\tNS=3\tGT\t1|1\t0|0\t0|1",
    ]
    p.write_text(TINY_VCF_HEADER + "\n".join(rows) + "\n")
    return p


def _write_per_chrom_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for chrom in ["chr1A", "chr1B", "chr1D"]:
        (d / f"{chrom}.vcf.gz").write_text(
            TINY_VCF_HEADER + f"{chrom}\t100\t.\tA\tG\t.\tPASS\tNS=3\tGT\t0|0\t0|1\t1|1\n"
        )
    return d


@pytest.fixture
def wheat_like_cfg_single(tmp_path: Path) -> SpeciesConfig:
    fasta = tmp_path / "ref.fa"
    fasta.touch()
    gff = tmp_path / "ref.gff"
    gff.touch()
    chrom_map = tmp_path / "chrom_map.tsv"
    chrom_map.write_text(
        "panel_chrom\tfasta_chrom\tsubgenome\nchr1A\tchr1A\tA\nchr1B\tchr1B\tB\nchr1D\tchr1D\tD\n")
    vcf = _write_single_vcf(tmp_path / "panel.vcf.gz")
    pheno = tmp_path / "pheno.tsv"
    pheno.write_text("sample_id\ttrait1\n")
    return SpeciesConfig(
        id="wheat_like", latin="X", common_name="X",
        genome_type="allopolyploid", ploidy=6,
        subgenomes=[
            Subgenome(id="A", chroms=["chr1A"]),
            Subgenome(id="B", chroms=["chr1B"]),
            Subgenome(id="D", chroms=["chr1D"]),
        ],
        fasta=fasta, gff=gff, chrom_map=chrom_map,
        geno=GenoSource(vcf=vcf, layout="single"),
        pheno=pheno,
    )


@pytest.fixture
def wheat_like_cfg_per_chrom(tmp_path: Path) -> SpeciesConfig:
    fasta = tmp_path / "ref.fa"
    fasta.touch()
    gff = tmp_path / "ref.gff"
    gff.touch()
    chrom_map = tmp_path / "chrom_map.tsv"
    chrom_map.write_text(
        "panel_chrom\tfasta_chrom\tsubgenome\nchr1A\tchr1A\tA\nchr1B\tchr1B\tB\nchr1D\tchr1D\tD\n")
    vcf_dir = _write_per_chrom_dir(tmp_path / "vcfs")
    pheno = tmp_path / "pheno.tsv"
    pheno.write_text("sample_id\ttrait1\n")
    return SpeciesConfig(
        id="wheat_like", latin="X", common_name="X",
        genome_type="allopolyploid", ploidy=6,
        subgenomes=[
            Subgenome(id="A", chroms=["chr1A"]),
            Subgenome(id="B", chroms=["chr1B"]),
            Subgenome(id="D", chroms=["chr1D"]),
        ],
        fasta=fasta, gff=gff, chrom_map=chrom_map,
        geno=GenoSource(vcf=vcf_dir, layout="per_chrom"),
        pheno=pheno,
    )


# ---------------------------------------------------------------------------
# 1. Plan construction (no I/O)
# ---------------------------------------------------------------------------


def test_plan_single_layout(wheat_like_cfg_single, tmp_path):
    plan = plan_split(wheat_like_cfg_single, tmp_path / "out", threads=4)
    assert len(plan.subgenomes) == 3
    sg_ids = [sg.sg_id for sg in plan.subgenomes]
    assert sg_ids == ["A", "B", "D"]
    sgA = plan.subgenomes[0]
    # bcftools view -r chr1A
    assert sgA.bcftools_cmd[0] == "bcftools"
    assert sgA.bcftools_cmd[1] == "view"
    assert "-r" in sgA.bcftools_cmd
    r_idx = sgA.bcftools_cmd.index("-r")
    assert sgA.bcftools_cmd[r_idx + 1] == "chr1A"
    # plink2 import with auto var-id
    assert "--set-all-var-ids" in sgA.plink2_cmd
    assert "@:#" in sgA.plink2_cmd
    # MAF + missingness defaults from QCConfig (hwe is None by default → not added)
    assert "--maf" in sgA.plink2_cmd
    assert "--geno" in sgA.plink2_cmd
    assert "--hwe" not in sgA.plink2_cmd  # default hwe_min_p=None
    # Default: pgen + bed both (two-pass since plink2 disallows in one cmd)
    assert "--make-pgen" in sgA.plink2_cmd
    assert sgA.plink2_bed_cmd is not None
    assert "--make-bed" in sgA.plink2_bed_cmd
    assert "--pfile" in sgA.plink2_bed_cmd


def test_plan_per_chrom_layout(wheat_like_cfg_per_chrom, tmp_path):
    plan = plan_split(wheat_like_cfg_per_chrom, tmp_path / "out", threads=4)
    sgA = plan.subgenomes[0]
    # bcftools concat <chr1A.vcf.gz> -o <subset.vcf.gz>
    assert sgA.bcftools_cmd[0] == "bcftools"
    assert sgA.bcftools_cmd[1] == "concat"
    # Inputs come after the -o subset_vcf; pick only those
    o_idx = sgA.bcftools_cmd.index("-o")
    input_args = [a for a in sgA.bcftools_cmd[o_idx + 2:] if a.endswith(".vcf.gz")]
    assert len(input_args) == 1
    assert "chr1A" in input_args[0]


def test_plan_keep_vcf_adds_export(wheat_like_cfg_single, tmp_path):
    plan = plan_split(wheat_like_cfg_single, tmp_path / "out", keep_vcf=True)
    sgA = plan.subgenomes[0]
    assert sgA.plink2_export_vcf_cmd is not None
    assert "--export" in sgA.plink2_export_vcf_cmd
    assert sgA.out_vcf is not None
    assert sgA.out_vcf.name == "all.vcf.gz"


def test_plan_skip_pgen_omits_pgen(wheat_like_cfg_single, tmp_path):
    plan = plan_split(wheat_like_cfg_single, tmp_path / "out", skip_pgen=True)
    sgA = plan.subgenomes[0]
    assert "--make-pgen" not in sgA.plink2_cmd
    assert "--make-bed" in sgA.plink2_cmd
    # No second pass needed when only bed is wanted
    assert sgA.plink2_bed_cmd is None


def test_plan_qc_hwe_only_added_when_set(wheat_like_cfg_single, tmp_path):
    wheat_like_cfg_single.qc.hwe_min_p = 1e-6
    plan = plan_split(wheat_like_cfg_single, tmp_path / "out")
    sgA = plan.subgenomes[0]
    assert "--hwe" in sgA.plink2_cmd
    hwe_idx = sgA.plink2_cmd.index("--hwe")
    assert sgA.plink2_cmd[hwe_idx + 1] == "1e-06"


def test_plan_rejects_bed_root_mode(wheat_like_cfg_single, tmp_path):
    """If user already provided pre-split BED, splitter must refuse."""
    bed_root = tmp_path / "bed"
    (bed_root / "A").mkdir(parents=True)
    (bed_root / "A" / "all.bim").write_text("")
    wheat_like_cfg_single.geno = GenoSource(bed_root=bed_root)
    with pytest.raises(ValueError, match="bed_root mode"):
        plan_split(wheat_like_cfg_single, tmp_path / "out")


# ---------------------------------------------------------------------------
# 2. per_chrom routing
# ---------------------------------------------------------------------------


def test_route_per_chrom_finds_files(tmp_path):
    d = _write_per_chrom_dir(tmp_path / "vcfs")
    files = _route_per_chrom(d, ["chr1A", "chr1B"])
    assert len(files) == 2
    assert all(f.exists() for f in files)


def test_route_per_chrom_missing_raises(tmp_path):
    d = _write_per_chrom_dir(tmp_path / "vcfs")
    with pytest.raises(FileNotFoundError, match="chr9X"):
        _route_per_chrom(d, ["chr9X"])


# ---------------------------------------------------------------------------
# 3. execute_plan with subprocess monkeypatch (no external tools needed)
# ---------------------------------------------------------------------------


def test_execute_plan_dry_run_no_subprocess(wheat_like_cfg_single, tmp_path):
    plan = plan_split(wheat_like_cfg_single, tmp_path / "out")
    with patch("homoeogwas.species_split.subprocess.run") as mock_run:
        summary = execute_plan(plan, dry_run=True)
    mock_run.assert_not_called()
    assert summary["dry_run"] is True
    assert all(sg["status"] == "PLANNED" for sg in summary["subgenomes"])


def test_execute_plan_runs_each_subgenome(wheat_like_cfg_single, tmp_path):
    plan = plan_split(wheat_like_cfg_single, tmp_path / "out")

    call_log = []

    def fake_run(cmd, **kwargs):
        call_log.append(cmd)
        # Simulate plink2 creating the .bed file
        if cmd[0] == "plink2":
            out_idx = cmd.index("--out")
            Path(cmd[out_idx + 1] + ".bed").write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    with patch("homoeogwas.species_split.subprocess.run", side_effect=fake_run):
        summary = execute_plan(plan, dry_run=False, force=True)

    # 3 sg × {bcftools view, bcftools index, plink2 pgen, plink2 bed} = 12 calls
    assert len(call_log) == 12
    assert all(sg["status"] == "OK" for sg in summary["subgenomes"])
    # split_summary.json persisted
    assert (tmp_path / "out" / "split_summary.json").exists()


def test_execute_plan_skips_existing_without_force(wheat_like_cfg_single, tmp_path):
    plan = plan_split(wheat_like_cfg_single, tmp_path / "out")
    # Pre-create the A subgenome BED to simulate prior run
    (tmp_path / "out" / "A").mkdir(parents=True)
    (tmp_path / "out" / "A" / "all.bed").write_text("")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "plink2":
            out_idx = cmd.index("--out")
            Path(cmd[out_idx + 1] + ".bed").write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    with patch("homoeogwas.species_split.subprocess.run", side_effect=fake_run) as mock_run:
        summary = execute_plan(plan, dry_run=False, force=False)
    # A skipped, B+D run → 2 sg × 4 cmds (bcftools view, index, plink2 pgen, plink2 bed) = 8 calls
    assert mock_run.call_count == 8
    statuses = [sg["status"] for sg in summary["subgenomes"]]
    assert "SKIPPED_EXISTING" in statuses
    assert statuses.count("OK") == 2


def test_execute_plan_error_status_on_subprocess_failure(wheat_like_cfg_single, tmp_path):
    plan = plan_split(wheat_like_cfg_single, tmp_path / "out")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "plink2":
            raise subprocess.CalledProcessError(1, cmd, stderr="plink2 failed: bad VCF")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")

    with patch("homoeogwas.species_split.subprocess.run", side_effect=fake_run):
        summary = execute_plan(plan, dry_run=False, force=True)
    assert all(sg["status"] == "ERROR" for sg in summary["subgenomes"])
    assert "plink2 failed" in summary["subgenomes"][0]["error"]


# ---------------------------------------------------------------------------
# 4. CLI subcommand integration
# ---------------------------------------------------------------------------


def test_cli_split_dry_run_via_argparse(wheat_like_cfg_single, tmp_path, capsys):
    """Smoke-test cmd_split via argparse (no subprocess hit, dry-run)."""
    yaml_path = tmp_path / "wheat_like.yaml"
    yaml_path.write_text(_cfg_to_yaml(wheat_like_cfg_single, tmp_path))

    from homoeogwas.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "split", "--species-yaml", str(yaml_path),
        "--out-dir", str(tmp_path / "out"),
        "--dry-run", "--skip-validate",
        "--project-root", str(tmp_path),
    ])
    assert args.subcommand == "split"
    assert args.dry_run is True
    rc = cmd_split(args)
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["species_id"] == "wheat_like"
    assert len(payload["subgenomes"]) == 3


def _cfg_to_yaml(cfg: SpeciesConfig, root: Path) -> str:
    """Tiny ad-hoc YAML emitter for tests; uses abs paths so YAML loader resolves."""
    import yaml

    def _path(p):
        return str(p)

    raw = {
        "id": cfg.id, "latin": cfg.latin, "common_name": cfg.common_name,
        "genome_type": cfg.genome_type, "ploidy": cfg.ploidy,
        "subgenomes": [
            {"id": sg.id, "chroms": sg.chroms, "copy": sg.copy_number}
            for sg in cfg.subgenomes
        ],
        "fasta": _path(cfg.fasta),
        "gff": _path(cfg.gff),
        "chrom_map": _path(cfg.chrom_map),
        "geno": {"vcf": _path(cfg.geno.vcf), "layout": cfg.geno.layout},
        "pheno": _path(cfg.pheno),
    }
    return yaml.safe_dump(raw)
