"""Breeder-level workflow engine — the executable form of ``AGENTS.md``.

Both the MCP server (:mod:`homoeogwas.mcp_server`) and any other agent wrapper
should call these functions rather than re-implementing the workflow. Each
high-level entry point takes a few *biological* inputs, **generates the YAML
config**, validates it, runs the plain CLI, and returns a JSON-serialisable
result (paths + a short interpreted summary). Pass ``dry_run=True`` to get the
planned commands and the generated config path without executing anything — this
is also how the test-suite exercises the logic without a real GWAS run.

Nothing here imports the optional ``mcp`` package, so it is always available.
"""
from __future__ import annotations

import json
import subprocess
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path

# ----------------------------------------------------------------------
# helpers: ploidy / mode, sample-id check, config IO
# ----------------------------------------------------------------------


def infer_interaction_mode(subgenomes: Sequence[str]) -> str:
    """Infer the input adapter; 4+ copies use pair-edge group omniB."""
    n = len(subgenomes)
    if n == 2:
        return "pairwise"
    if n == 3:
        return "triad"
    if n >= 4:
        return "group"
    raise ValueError(
        f"interaction needs at least 2 subgenomes, got {n}")


def check_phenotype_inputs(phenotype: str, sample_col: str,
                           trait: str | None = None) -> dict:
    """Validate a phenotype file before a run: the sample/trait columns exist and
    the integer-sample-id pitfall (which silently breaks the genotype join)."""
    from .io import read_delimited
    try:
        df = read_delimited(
            phenotype, nrows=200, dtype={sample_col: "string"})
    except Exception as e:   # noqa: BLE001 - report any read failure as structured
        return {"ok": False, "reason": f"cannot read phenotype {phenotype!r}: {e}"}
    need = [c for c in (sample_col, trait) if c is not None]
    missing = [c for c in need if c not in df.columns]
    if missing:
        return {"ok": False, "reason": f"columns {missing} not in phenotype; "
                f"have {list(df.columns)}"}
    col = df[sample_col].dropna()
    if len(col) == 0:
        return {"ok": False, "reason": f"phenotype {phenotype!r} has no usable "
                f"rows in sample column {sample_col!r}"}
    integer_like = bool(len(col) and col.astype(str).str.fullmatch(r"-?\d+").all())
    return {"ok": True, "integer_like": integer_like,
            "advice": ("sample ids look integer-like — HomoeoGWAS will preserve "
                       "the phenotype sample column as strings before joining to "
                       ".fam IIDs" if integer_like else
                       "sample ids are non-numeric strings (fine)")}


def check_sample_ids(phenotype: str, sample_col: str) -> dict:
    """Back-compat alias: sample-id check only (see check_phenotype_inputs)."""
    return check_phenotype_inputs(phenotype, sample_col)


def write_config(cfg: dict, path: str | Path) -> str:
    """Dump a config dict to YAML (or JSON if PyYAML is unavailable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        with open(path, "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    except ModuleNotFoundError:
        path = path.with_suffix(".json")
        path.write_text(json.dumps(cfg, indent=2))
    return str(path)


def _materialize_bed_layout(bed_prefixes: Mapping[str, str],
                            subgenomes: Sequence[str],
                            work_dir: Path) -> tuple[str, list[str]]:
    """Symlink the per-subgenome BEDs into a canonical ``geno/<S>/all`` layout.

    Always builds the deterministic template (no fragile path-pattern guessing,
    which can mis-detect when a subgenome letter appears in a shared path
    segment). Returns ``(template, missing)`` where ``missing`` lists subgenomes
    whose ``.bed/.bim/.fam`` could not all be found.
    """
    geno = work_dir / "geno"
    missing: list[str] = []
    for s in subgenomes:
        if s not in bed_prefixes:
            missing.append(s)
            continue
        src = Path(str(bed_prefixes[s]))
        if not all(Path(str(src) + ext).exists()
                   for ext in (".bed", ".bim", ".fam")):
            missing.append(s)
            continue
        d = geno / s
        d.mkdir(parents=True, exist_ok=True)
        for ext in (".bed", ".bim", ".fam"):
            dst = d / ("all" + ext)
            target = Path(str(src) + ext).resolve()
            if dst.is_symlink():
                if dst.resolve(strict=False) == target:
                    continue
                dst.unlink()
            elif dst.exists():
                raise FileExistsError(
                    f"workflow-managed path {dst} exists and is not a symlink; "
                    "use a new out_dir or move that file")
            dst.symlink_to(target)
    return str(geno / "{subgenome}" / "all"), missing


# ----------------------------------------------------------------------
# config builders (pure)
# ----------------------------------------------------------------------


def build_fit_config(*, subgenomes: Sequence[str], phenotype: str,
                     sample_col: str, trait: str, bed_template: str,
                     out_dir: str, panel: str = "panel",
                     include_hadamard: bool = False, loco: bool = False,
                     maf_min: float = 0.05, call_rate_min: float = 0.9,
                     scan_mode: str = "memory", backend: str = "cpu",
                     marker_encoding: str = "diploid_0_1_2",
                     marker_manifest_template: str | None = None) -> dict:
    """Assemble a ``homoeogwas fit`` config dict from high-level inputs."""
    loco_block = ({"enabled": True, "fallback": "error"} if loco
                  else {"enabled": False})
    cfg = {
        "fit_version": 1,
        "panel": {"name": panel, "subgenomes": list(subgenomes)},
        "phenotype": {"path": phenotype, "sample_col": sample_col,
                      "trait": trait},
        "genotype": {
            "scan_bed_prefix_template": bed_template,
            "marker_encoding": marker_encoding,
            "grm": {"source": "bed", "bed_prefix_template": bed_template,
                    "maf_min": maf_min}},
        "kernels": {"normalize": "trace",
                    "include_hadamard": bool(include_hadamard),
                    "hadamard_name": "hom"},
        "reml": {"n_starts": 10, "seed": 2026},
        "scan": {"mode": scan_mode, "backend": backend, "maf_min": maf_min,
                 "call_rate_min": call_rate_min, "loco": loco_block},
        "plots": {"enabled": True},
        "outputs": {"out_dir": out_dir, "prefix": trait},
    }
    if marker_manifest_template is not None:
        cfg["genotype"]["marker_manifest_template"] = marker_manifest_template
    return cfg


def build_interact_config(*, subgenomes: Sequence[str], bed_prefixes: Mapping[str, str],
                          snp_to_gene: Mapping[str, str], phenotype: str,
                          sample_col: str, trait: str, out_dir: str,
                          pairs: str | None = None, triads: str | None = None,
                          perm_b: int = 2000, cap: int = 150,
                          min_snp: int | None = None, statistic: str = "omniB",
                          groups: str | None = None,
                          hypothesis_unit: str | None = None,
                          subset_order: int = 2,
                          family_scope: str = "primary_only") -> dict:
    """Assemble a ``homoeogwas interact`` config dict from high-level inputs."""
    mode = infer_interaction_mode(subgenomes)
    statistic_key = str(statistic).lower()
    if statistic_key not in {"omnib", "burden", "triad3"}:
        raise ValueError("statistic must be omniB, burden, or experimental triad3")
    if statistic_key == "triad3" and mode != "triad":
        raise ValueError("statistic=triad3 requires exactly three subgenomes")
    calibration = (
        {"method": "permutation", "perm_B": perm_b}
        if statistic_key == "burden"
        else {"method": "bootstrap", "B": perm_b, "seed": 2026}
    )
    is_canonical_omnib = statistic_key == "omnib"
    if min_snp is None:
        min_snp = 3 if is_canonical_omnib else 2
    if is_canonical_omnib:
        group_path = groups or (
            pairs if mode == "pairwise" else (
                triads if mode == "triad" else None))
        if not group_path:
            legacy_name = (
                "pairs" if mode == "pairwise" else
                ("triads" if mode == "triad" else "groups"))
            raise ValueError(
                f"{mode} mode needs a {legacy_name} TSV "
                "(gene_<subgenome> columns)")
        if hypothesis_unit is None:
            hypothesis_unit = "edge" if mode == "pairwise" else "group"
    cfg = {
        "interact": {
            "mode": "group" if is_canonical_omnib else mode,
            "statistic": "omniB" if statistic_key == "omnib" else statistic_key,
            "subgenomes": list(subgenomes),
            "genotype": {s: str(bed_prefixes[s]) for s in subgenomes},
            "snp_to_gene": {s: str(snp_to_gene[s]) for s in subgenomes},
            "phenotype": phenotype, "sample_col": sample_col, "trait": trait,
            "burden": {"cap": cap, "min_snp": min_snp},
            "grm": {"method": "grm_from_X", "maf_min": 0.01,
                    "scope": "all_subgenomes"},
            "calibration": calibration},
        "outputs": {
            "out_dir": out_dir,
            **({"full_ranking": True} if is_canonical_omnib else {}),
        },
    }
    if is_canonical_omnib:
        cfg["interact"]["burden"].update({"maf_min": 0.01, "n_pc": 3})
        cfg["interact"].update({
            "groups": str(group_path),
            "hypothesis_unit": hypothesis_unit,
            "subset_order": subset_order,
            "family_scope": family_scope,
            "primary_transform": "INT",
            "primary_multiplicity": "bootstrap_minp",
        })
    elif statistic_key == "triad3":
        cfg["interact"]["primary_multiplicity"] = "bootstrap_minp"
    if not is_canonical_omnib and mode == "triad":
        if not triads:
            raise ValueError("triad mode needs a triads TSV (gene_A/B/C)")
        cfg["interact"]["triads"] = triads
    elif not is_canonical_omnib:
        if not pairs:
            raise ValueError("pairwise mode needs a pairs TSV (gene_A/B)")
        cfg["interact"]["pairs"] = pairs
    return cfg


# ----------------------------------------------------------------------
# CLI runner (monkeypatch point for tests)
# ----------------------------------------------------------------------


def run_cli(args: Sequence[str], *, dry_run: bool = False) -> dict:
    """Invoke ``homoeogwas <args>`` (via ``python -m homoeogwas``).

    Returns ``{command, returncode, stdout_tail}``; with ``dry_run`` the command
    is only planned, not executed.
    """
    cmd = [sys.executable, "-m", "homoeogwas", *map(str, args)]
    if dry_run:
        return {"command": cmd, "dry_run": True, "returncode": None}
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"command": cmd, "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-1000:]}


# ----------------------------------------------------------------------
# output summarisation
# ----------------------------------------------------------------------


def summarize_fit(out_dir: str, trait: str, top_n: int = 10) -> dict:
    """Read a finished fit's summary + sumstats into a short interpreted result."""
    out = Path(out_dir)
    summ_path = out / f"summary_{trait}.json"
    if not summ_path.exists():
        return {"ok": False, "reason": f"no {summ_path}"}
    s = json.loads(summ_path.read_text())
    reml = s.get("reml", {})
    res = {"ok": True, "trait": s.get("trait"), "n": s.get("n_analysis"),
           "subgenomes": s.get("subgenomes"),
           "pve": {k: round(v, 4) for k, v in reml.get("pve", {}).items()},
           "lambda_gc": s.get("lambda_gc", {}).get("all"),
           "summary_json": str(summ_path),
           "figures": [str(p) for p in sorted(out.glob(f"*_{trait}.png"))]}
    ss = s.get("outputs", {}).get("sumstats", [])
    if ss and Path(ss[0]).exists():
        try:
            import pandas as pd
            df = pd.read_csv(ss[0], sep="\t",
                             usecols=["subgenome", "chrom", "pos", "p"])
            res["top_hits"] = df.nsmallest(top_n, "p").to_dict(orient="records")
        except Exception as e:   # noqa: BLE001 - top hits are best-effort
            res["top_hits_error"] = str(e)
    return res


def summarize_interaction(out_dir: str, trait: str) -> dict:
    """Summarize the declared interaction family without changing its claims."""
    out = Path(out_dir)
    result_path = out / f"interact_{trait}.json"
    audit_path = out / "audit" / "homoeogwas_audit.json"
    if not result_path.exists():
        return {"ok": False, "reason": f"no {result_path}"}
    try:
        payload = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": f"cannot read {result_path}: {exc}"}
    provenance = payload.get("provenance") or {}
    primary_key = str(provenance.get("primary_transform", "INT"))
    primary = (payload.get("results") or {}).get(primary_key) or {}
    diagnostics = primary.get("model_diagnostics") or {}
    fwer = diagnostics.get("bootstrap_fwer") or {}
    family = diagnostics.get("family_provenance") or {}
    unit = provenance.get("hypothesis_unit") or fwer.get(
        "declared_hypothesis_unit")
    ranking = out / f"interact_{trait}_ranking_group_{primary_key}.tsv"
    top_descriptive = list(primary.get("top") or [])
    significant = None if fwer.get("inferential") is False else fwer.get("sig")
    driver_records = significant if isinstance(significant, list) else top_descriptive
    summary = {
        "ok": True,
        "trait": payload.get("trait", trait),
        "primary_unit": unit,
        "family_scope": provenance.get("family_scope") or fwer.get("family_scope"),
        "n_samples": primary.get("n"),
        "n_planned": primary.get("n_planned", primary.get("G")),
        "n_valid": primary.get("n_valid"),
        "n_significant": fwer.get("n_rejected"),
        "global_fwer_p": fwer.get("empirical_p"),
        "bootstrap_B": fwer.get("B", primary.get("bootstrap_B")),
        "n_groups_raw": family.get(
            "n_groups_raw", provenance.get("n_groups_raw")),
        "n_unique_edges": family.get(
            "n_unique_edges", provenance.get("n_unique_edges")),
        "group_family_sha256": family.get(
            "group_family_sha256", provenance.get("group_family_sha256")),
        "edge_family_sha256": family.get(
            "edge_family_sha256", provenance.get("edge_family_sha256")),
        "lambda_gc": primary.get("lambda_gc_obs"),
        "significant": significant,
        "evidence_drivers": [
            {
                "hypothesis_id": record.get("hypothesis_id"),
                "component": record.get("driving_component")
                or record.get("smallest_component"),
            }
            for record in driver_records if isinstance(record, dict)
        ],
        "top": top_descriptive,
        "top_descriptive": top_descriptive,
        "top_descriptive_role": "descriptive_raw_p_ranking",
        "ranking_tsv": str(ranking) if ranking.exists() else None,
        "result_json": str(result_path),
    }
    if audit_path.exists():
        summary["audit_json"] = str(audit_path)
        try:
            summary["audit_status"] = json.loads(
                audit_path.read_text()).get("overall_status")
        except (OSError, json.JSONDecodeError):
            summary["audit_status"] = "UNREADABLE"
    return summary


# ----------------------------------------------------------------------
# high-level orchestrators (the MCP tool bodies)
# ----------------------------------------------------------------------


def get_guidance(goal: str = "gwas") -> dict:
    """Return the canonical workflow spec (AGENTS.md) + a short routing hint."""
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "AGENTS.md",  # editable checkout
        Path(sysconfig.get_path("data")) / "share" / "homoeogwas" / "AGENTS.md",
    ]
    agents = next((path for path in candidates if path.exists()), None)
    hints = {
        "gwas": "VCF→split→fit→plot, or BED→fit→plot. Inputs: bed_prefixes or "
                "vcf+species_yaml, phenotype, sample_col, trait, subgenomes.",
        "interaction": "prep-snps→prep-homoeologs→group omniB. Pair edges are "
                       "the common primitive; legacy pair/triad tables adapt "
                       "automatically and 4 copies aggregate six edges.",
        "split": "homoeogwas split --species-yaml … -o …",
        "validate": "homoeogwas validate -c <config>",
        "troubleshoot": "string sample ids; gff/.bim chrom match; diamond 2.1.x; "
                        "never 4-way interaction.",
    }
    return {"goal": goal, "hint": hints.get(goal, hints["gwas"]),
            "spec": agents.read_text() if agents is not None else None}


def run_gwas(*, phenotype: str, sample_col: str, trait: str,
             subgenomes: Sequence[str], out_dir: str,
             bed_prefixes: Mapping[str, str] | None = None,
             include_hadamard: bool = False, loco: bool = False,
             run_plots: bool = True, allow_integer_ids: bool = False,
             dry_run: bool = False) -> dict:
    """Generate a fit config from breeder-level inputs, validate, run, summarize.

    Blocks *before* any expensive run on the common breeder errors — a bad
    phenotype, a missing column, integer-like sample ids, or absent PLINK files —
    returning ``{ok: False, reason, advice}`` instead of a stack trace or a
    doomed GWAS. ``allow_integer_ids`` is retained as a no-op compatibility
    argument; IDs are now always read as strings.
    """
    out = Path(out_dir)
    warnings = []
    if not bed_prefixes:
        return {"ok": False, "reason": "run_gwas needs bed_prefixes "
                "(or run split on a VCF first)"}
    chk = check_phenotype_inputs(phenotype, sample_col, trait)
    if not chk["ok"]:
        return {"ok": False, "reason": chk["reason"]}
    if chk["integer_like"]:
        warnings.append(
            "phenotype sample ids are integer-like; HomoeoGWAS will read the "
            "sample column as strings before joining to BED IIDs")
    try:
        tmpl, missing = _materialize_bed_layout(bed_prefixes, subgenomes, out)
    except (FileExistsError, OSError) as exc:
        return {"ok": False, "reason": str(exc)}
    if missing:
        return {"ok": False, "reason": f"missing .bed/.bim/.fam for subgenomes "
                f"{missing} under the given prefixes"}
    cfg = build_fit_config(subgenomes=subgenomes, phenotype=phenotype,
                           sample_col=sample_col, trait=trait, bed_template=tmpl,
                           out_dir=out_dir, include_hadamard=include_hadamard,
                           loco=loco)
    cfg_path = write_config(cfg, out / "configs" / "fit.generated.yaml")
    # the workflow owns out_dir (it wrote the config there), so fit overwrites it
    steps = [run_cli(["validate", "-c", cfg_path], dry_run=dry_run)]
    if dry_run or steps[-1].get("returncode") == 0:
        steps.append(run_cli(["fit", "-c", cfg_path, "--force"], dry_run=dry_run))
    if run_plots and (dry_run or (len(steps) >= 2 and steps[-1].get("returncode") == 0)):
        steps.append(run_cli(["plot", out_dir], dry_run=dry_run))
    failed = next((s for s in steps if s.get("returncode") not in (None, 0)), None)
    result = {"ok": failed is None, "config": cfg_path, "out_dir": out_dir,
              "steps": steps, "warnings": warnings, "dry_run": dry_run}
    if failed is not None:
        result["reason"] = (
            f"command failed with exit code {failed['returncode']}: "
            f"{' '.join(map(str, failed['command']))}")
    if not dry_run and len(steps) >= 2 and steps[1].get("returncode") == 0:
        result["summary"] = summarize_fit(out_dir, trait)
    return result


def run_interaction(*, phenotype: str, sample_col: str, trait: str,
                    subgenomes: Sequence[str], bed_prefixes: Mapping[str, str],
                    snp_to_gene: Mapping[str, str], out_dir: str,
                    pairs: str | None = None, triads: str | None = None,
                    groups: str | None = None,
                    hypothesis_unit: str | None = None,
                    subset_order: int = 2,
                    family_scope: str = "primary_only",
                    perm_b: int = 2000, n_jobs: int = 8,
                    statistic: str = "omniB",
                    dry_run: bool = False) -> dict:
    """Generate, validate, run, audit and summarize an interaction analysis.

    New omniB runs always use the canonical group table and one experiment-wide
    family. ``pairs`` and ``triads`` remain breeder-level aliases for old input
    tables; users are never asked to write YAML.
    """
    out = Path(out_dir)
    try:
        mode = infer_interaction_mode(subgenomes)
    except ValueError as e:
        return {"ok": False, "reason": str(e)}
    chk = check_phenotype_inputs(phenotype, sample_col, trait)
    if not chk["ok"] and not dry_run:
        return {"ok": False, "reason": chk["reason"]}
    missing_maps = {
        name: [s for s in subgenomes if s not in mapping]
        for name, mapping in (
            ("bed_prefixes", bed_prefixes), ("snp_to_gene", snp_to_gene))
    }
    missing_maps = {name: values for name, values in missing_maps.items() if values}
    if missing_maps:
        return {"ok": False, "reason": f"missing subgenome mappings: {missing_maps}"}
    statistic_key = str(statistic).lower()
    canonical = statistic_key == "omnib"
    table = groups or (triads if mode == "triad" else pairs)
    needed = ([str(bed_prefixes[s]) + ext
               for s in subgenomes for ext in (".bed", ".bim", ".fam")]
              + [str(snp_to_gene[s]) for s in subgenomes]
              + ([table] if table else []))
    absent = [p for p in needed if not Path(p).exists()]
    if not dry_run and absent:
        return {"ok": False, "reason": f"missing interaction inputs: {absent[:6]}"}
    cfg = build_interact_config(subgenomes=subgenomes, bed_prefixes=bed_prefixes,
                                snp_to_gene=snp_to_gene, phenotype=phenotype,
                                sample_col=sample_col, trait=trait,
                                out_dir=out_dir, pairs=pairs, triads=triads,
                                groups=groups,
                                hypothesis_unit=hypothesis_unit,
                                subset_order=subset_order,
                                family_scope=family_scope,
                                perm_b=perm_b, statistic=statistic)
    public_mode = "group" if canonical else mode
    declared_unit = cfg["interact"].get("hypothesis_unit")
    config_name = (
        "interact.generated.group.omnib.yaml"
        if canonical else f"interact.generated.{mode}.{statistic_key}.yaml")
    cfg_path = write_config(cfg, out / "configs" / config_name)
    steps = [run_cli(["validate", "-c", cfg_path], dry_run=dry_run)]
    if dry_run or steps[-1].get("returncode") == 0:
        steps.append(
            run_cli(["interact", "-c", cfg_path, "--n-jobs", str(n_jobs)],
                    dry_run=dry_run))
    if dry_run or (len(steps) >= 2 and steps[-1].get("returncode") == 0):
        steps.append(run_cli(["audit", out_dir], dry_run=dry_run))
    failed = next((s for s in steps if s.get("returncode") not in (None, 0)), None)
    result = {"ok": failed is None, "config": cfg_path, "mode": public_mode,
              "hypothesis_unit": declared_unit,
              "out_dir": out_dir, "steps": steps, "dry_run": dry_run}
    if failed is not None:
        result["reason"] = (
            f"command failed with exit code {failed['returncode']}: "
            f"{' '.join(map(str, failed['command']))}")
    if not dry_run and failed is None:
        result["summary"] = summarize_interaction(out_dir, trait)
    return result


def split_genotype(*, species_yaml: str, out_dir: str, vcf: str | None = None,
                   threads: int = 8,
                   dry_run: bool = False) -> dict:
    """Split a VCF into per-subgenome BEDs via ``homoeogwas split``."""
    args = ["split", "--species-yaml", species_yaml, "-o", out_dir,
            "--threads", str(threads)]
    if vcf:
        args += ["--vcf", vcf]
    step = run_cli(args, dry_run=dry_run)
    return {"ok": step.get("returncode") in (None, 0), "out_dir": out_dir,
            "step": step, "dry_run": dry_run}


def make_plots(*, results_dir: str, formats: str = "png,pdf,svg",
               dry_run: bool = False) -> dict:
    """Regenerate publication figures from a finished run (no recompute)."""
    step = run_cli(["plot", results_dir, "--formats", formats], dry_run=dry_run)
    return {"results_dir": results_dir, "step": step, "dry_run": dry_run}
