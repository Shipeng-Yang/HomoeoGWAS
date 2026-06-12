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
from collections.abc import Mapping, Sequence
from pathlib import Path

# ----------------------------------------------------------------------
# helpers: ploidy / mode, sample-id check, config IO
# ----------------------------------------------------------------------


def infer_interaction_mode(subgenomes: Sequence[str]) -> str:
    """2 subgenomes -> ``pairwise``, 3 -> ``triad``. Refuse a single 4-way test."""
    n = len(subgenomes)
    if n == 2:
        return "pairwise"
    if n == 3:
        return "triad"
    raise ValueError(
        f"interaction needs 2 (pairwise) or 3 (triad) subgenomes, got {n}; "
        "for 4+ subgenomes run over 2-/3-subgenome subsets and aggregate "
        "(never a single 4-way test) — see AGENTS.md §5")


def check_phenotype_inputs(phenotype: str, sample_col: str,
                           trait: str | None = None) -> dict:
    """Validate a phenotype file before a run: the sample/trait columns exist and
    the integer-sample-id pitfall (which silently breaks the genotype join)."""
    import pandas as pd
    try:
        df = pd.read_csv(phenotype, sep=None, engine="python", nrows=200)
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
            "advice": ("sample ids look integer-like — coerce both phenotype and "
                       ".fam ids to strings (e.g. prefix them) or the GRM∩pheno "
                       "join returns 0 overlap" if integer_like else
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
        src = Path(str(bed_prefixes[s]))
        if not all(Path(str(src) + ext).exists()
                   for ext in (".bed", ".bim", ".fam")):
            missing.append(s)
            continue
        d = geno / s
        d.mkdir(parents=True, exist_ok=True)
        for ext in (".bed", ".bim", ".fam"):
            dst = d / ("all" + ext)
            if not dst.exists():
                dst.symlink_to(Path(str(src) + ext).resolve())
    return str(geno / "{subgenome}" / "all"), missing


# ----------------------------------------------------------------------
# config builders (pure)
# ----------------------------------------------------------------------


def build_fit_config(*, subgenomes: Sequence[str], phenotype: str,
                     sample_col: str, trait: str, bed_template: str,
                     out_dir: str, panel: str = "panel",
                     include_hadamard: bool = False, loco: bool = False,
                     maf_min: float = 0.05, call_rate_min: float = 0.9,
                     scan_mode: str = "memory", backend: str = "cpu") -> dict:
    """Assemble a ``homoeogwas fit`` config dict from high-level inputs."""
    loco_block = ({"enabled": True, "fallback": "error"} if loco
                  else {"enabled": False})
    return {
        "fit_version": 1,
        "panel": {"name": panel, "subgenomes": list(subgenomes)},
        "phenotype": {"path": phenotype, "sample_col": sample_col,
                      "trait": trait},
        "genotype": {
            "scan_bed_prefix_template": bed_template,
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


def build_interact_config(*, subgenomes: Sequence[str], bed_prefixes: Mapping[str, str],
                          snp_to_gene: Mapping[str, str], phenotype: str,
                          sample_col: str, trait: str, out_dir: str,
                          pairs: str | None = None, triads: str | None = None,
                          perm_b: int = 200, cap: int = 150,
                          min_snp: int = 2) -> dict:
    """Assemble a ``homoeogwas interact`` config dict from high-level inputs."""
    mode = infer_interaction_mode(subgenomes)
    cfg = {
        "interact": {
            "mode": mode, "subgenomes": list(subgenomes),
            "genotype": {s: str(bed_prefixes[s]) for s in subgenomes},
            "snp_to_gene": {s: str(snp_to_gene[s]) for s in subgenomes},
            "phenotype": phenotype, "sample_col": sample_col, "trait": trait,
            "burden": {"cap": cap, "min_snp": min_snp},
            "grm": {"method": "grm_from_X"},
            "calibration": {"perm_B": perm_b}},
        "outputs": {"out_dir": out_dir},
    }
    if mode == "triad":
        if not triads:
            raise ValueError("triad mode needs a triads TSV (gene_A/B/C)")
        cfg["interact"]["triads"] = triads
    else:
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


# ----------------------------------------------------------------------
# high-level orchestrators (the MCP tool bodies)
# ----------------------------------------------------------------------


def get_guidance(goal: str = "gwas") -> dict:
    """Return the canonical workflow spec (AGENTS.md) + a short routing hint."""
    root = Path(__file__).resolve().parents[2]
    agents = root / "AGENTS.md"
    hints = {
        "gwas": "VCF→split→fit→plot, or BED→fit→plot. Inputs: bed_prefixes or "
                "vcf+species_yaml, phenotype, sample_col, trait, subgenomes.",
        "interaction": "prep-snps→prep-homoeologs→interact. mode: 2→pairwise, "
                       "3→triad, 4+→subsets.",
        "split": "homoeogwas split --species-yaml … -o …",
        "validate": "homoeogwas validate -c <config>",
        "troubleshoot": "string sample ids; gff/.bim chrom match; diamond 2.1.x; "
                        "never 4-way interaction.",
    }
    return {"goal": goal, "hint": hints.get(goal, hints["gwas"]),
            "spec": agents.read_text() if agents.exists() else None}


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
    doomed GWAS. Set ``allow_integer_ids`` only if you have already coerced ids.
    """
    out = Path(out_dir)
    warnings = []
    if not bed_prefixes:
        return {"ok": False, "reason": "run_gwas needs bed_prefixes "
                "(or run split on a VCF first)"}
    chk = check_phenotype_inputs(phenotype, sample_col, trait)
    if not chk["ok"]:
        return {"ok": False, "reason": chk["reason"]}
    if chk["integer_like"] and not allow_integer_ids:
        return {"ok": False, "blocked": "integer_sample_ids",
                "reason": "phenotype sample ids are integer-like, which silently "
                "breaks the genotype↔phenotype join", "advice": chk["advice"]}
    tmpl, missing = _materialize_bed_layout(bed_prefixes, subgenomes, out)
    if missing:
        return {"ok": False, "reason": f"missing .bed/.bim/.fam for subgenomes "
                f"{missing} under the given prefixes"}
    cfg = build_fit_config(subgenomes=subgenomes, phenotype=phenotype,
                           sample_col=sample_col, trait=trait, bed_template=tmpl,
                           out_dir=out_dir, include_hadamard=include_hadamard,
                           loco=loco)
    cfg_path = write_config(cfg, out / "configs" / "fit.generated.yaml")
    # the workflow owns out_dir (it wrote the config there), so fit overwrites it
    steps = [run_cli(["validate", "-c", cfg_path], dry_run=dry_run),
             run_cli(["fit", "-c", cfg_path, "--force"], dry_run=dry_run)]
    if run_plots:
        steps.append(run_cli(["plot", out_dir], dry_run=dry_run))
    result = {"ok": True, "config": cfg_path, "out_dir": out_dir,
              "steps": steps, "warnings": warnings, "dry_run": dry_run}
    if not dry_run and steps[1].get("returncode") == 0:
        result["summary"] = summarize_fit(out_dir, trait)
    return result


def run_interaction(*, phenotype: str, sample_col: str, trait: str,
                    subgenomes: Sequence[str], bed_prefixes: Mapping[str, str],
                    snp_to_gene: Mapping[str, str], out_dir: str,
                    pairs: str | None = None, triads: str | None = None,
                    perm_b: int = 200, n_jobs: int = 8,
                    dry_run: bool = False) -> dict:
    """Generate an interact config (mode inferred from ploidy), precheck inputs,
    then run. (``homoeogwas validate`` is fit-schema only, so the interaction
    path instead verifies its inputs exist before the long run.)"""
    out = Path(out_dir)
    try:
        mode = infer_interaction_mode(subgenomes)
    except ValueError as e:
        return {"ok": False, "reason": str(e)}
    chk = check_phenotype_inputs(phenotype, sample_col, trait)
    if not chk["ok"] and not dry_run:
        return {"ok": False, "reason": chk["reason"]}
    table = triads if mode == "triad" else pairs
    needed = ([str(bed_prefixes[s]) + ".bed" for s in subgenomes]
              + [str(snp_to_gene[s]) for s in subgenomes]
              + ([table] if table else []))
    absent = [p for p in needed if not Path(p).exists()]
    if not dry_run and absent:
        return {"ok": False, "reason": f"missing interaction inputs: {absent[:6]}"}
    cfg = build_interact_config(subgenomes=subgenomes, bed_prefixes=bed_prefixes,
                                snp_to_gene=snp_to_gene, phenotype=phenotype,
                                sample_col=sample_col, trait=trait,
                                out_dir=out_dir, pairs=pairs, triads=triads,
                                perm_b=perm_b)
    cfg_path = write_config(cfg, out / "configs" / f"interact.generated.{mode}.yaml")
    steps = [run_cli(["interact", "-c", cfg_path, "--n-jobs", str(n_jobs)],
                     dry_run=dry_run)]
    return {"ok": True, "config": cfg_path, "mode": mode, "out_dir": out_dir,
            "steps": steps, "dry_run": dry_run}


def split_genotype(*, species_yaml: str, out_dir: str, threads: int = 8,
                   dry_run: bool = False) -> dict:
    """Split a VCF into per-subgenome BEDs via ``homoeogwas split``."""
    step = run_cli(["split", "--species-yaml", species_yaml, "-o", out_dir,
                    "--threads", str(threads)], dry_run=dry_run)
    return {"out_dir": out_dir, "step": step, "dry_run": dry_run}


def make_plots(*, results_dir: str, formats: str = "png,pdf,svg",
               dry_run: bool = False) -> dict:
    """Regenerate publication figures from a finished run (no recompute)."""
    step = run_cli(["plot", results_dir, "--formats", formats], dry_run=dry_run)
    return {"results_dir": results_dir, "step": step, "dry_run": dry_run}
