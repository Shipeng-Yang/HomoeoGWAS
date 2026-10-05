"""MCP server exposing HomoeoGWAS to any MCP client (Claude, Cursor, Cline, …).

Thin wrappers over :mod:`homoeogwas.workflow`: every tool takes breeder-level
inputs, generates the YAML config, runs the CLI, and returns paths + a short
interpreted result. The heavy logic lives in ``workflow.py`` (and is unit-tested
there); this module only adapts it to MCP tool calls.

Run it with the ``homoeogwas-mcp`` console entry, or ``homoeogwas mcp``. Requires
the optional dependency: ``pip install "homoeogwas[mcp]"``.
"""
from __future__ import annotations

from . import workflow


def _safe(fn, **kw) -> dict:
    """Call a workflow function, returning a structured error instead of raising
    (an agent-facing server should never surface a raw traceback)."""
    try:
        return fn(**kw)
    except Exception as e:   # noqa: BLE001 - deliberately structured for clients
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def _prep_homoeologs_command(
        subgenomes: list[str], genes_template: str, out: str, *,
        from_table: str | None = None, table_format: str = "long",
        proteins: dict[str, str] | None = None,
        subgenome_map: str | None = None, diamond: str | None = None,
        restrict_base_group: bool = True) -> tuple[str, list[str]]:
    """Build the MCP preparation command without requiring the MCP dependency."""
    mode = workflow.infer_interaction_mode(subgenomes)
    args = ["prep-homoeologs", "--mode", mode,
            "--subgenomes", ",".join(subgenomes),
            "--genes", genes_template, "--out", out]
    if from_table:
        args += ["--from-table", from_table, "--table-format", table_format]
    elif proteins:
        args += ["--method", "diamond-rbh"]
        for subgenome, path in proteins.items():
            args += ["--proteins", f"{subgenome}={path}"]
        if restrict_base_group and subgenome_map:
            args += ["--restrict-base-group", "--subgenome-map", subgenome_map]
        if diamond:
            args += ["--diamond", diamond]
    else:
        raise ValueError("provide from_table or proteins")
    return mode, args


def build_server():
    """Construct the FastMCP server (imported lazily so core installs stay lean)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ModuleNotFoundError as e:   # pragma: no cover - exercised via CLI msg
        raise SystemExit(
            "ERR: the MCP server needs the optional 'mcp' dependency.\n"
            "      install it with:  pip install \"homoeogwas[mcp]\"") from e

    mcp = FastMCP("homoeogwas")

    @mcp.tool()
    def get_guidance(goal: str = "gwas") -> dict:
        """Return the canonical HomoeoGWAS workflow spec (AGENTS.md) plus a short
        routing hint. ``goal`` is one of gwas|interaction|split|validate|troubleshoot.
        Call this first to learn the required inputs and pipeline order."""
        return _safe(workflow.get_guidance, goal=goal)

    @mcp.tool()
    def validate_inputs(phenotype: str, sample_col: str, trait: str) -> dict:
        """Check phenotype inputs before a run: confirms the trait/sample columns
        exist and reports whether sample IDs look integer-like. The analysis
        reader preserves that column as strings before joining to BED IIDs."""
        return _safe(workflow.check_phenotype_inputs, phenotype=phenotype,
                     sample_col=sample_col, trait=trait)

    @mcp.tool()
    def split_genotype(species_yaml: str, out_dir: str, vcf: str | None = None,
                       threads: int = 8,
                       dry_run: bool = False) -> dict:
        """Split one VCF into per-subgenome PLINK BEDs using a species YAML."""
        return _safe(workflow.split_genotype, species_yaml=species_yaml,
                     out_dir=out_dir, vcf=vcf, threads=threads, dry_run=dry_run)

    @mcp.tool()
    def run_gwas(phenotype: str, sample_col: str, trait: str,
                 subgenomes: list[str], bed_prefixes: dict[str, str],
                 out_dir: str, include_hadamard: bool = False,
                 loco: bool = False, run_plots: bool = True,
                 scan_jobs: int = 1, dry_run: bool = False) -> dict:
        """Run a subgenome-stratified GWAS from breeder-level inputs: generates
        the fit YAML, validates, runs ``fit`` (+ figures), and returns the
        per-subgenome PVE, λ_GC, top hits and figure paths. ``bed_prefixes`` maps
        each subgenome label to a PLINK prefix. ``scan_jobs`` > 1 scores
        streaming chunks in parallel worker processes (same results). For a
        multi-environment trait pass an environment-adjusted line mean, not a
        raw average over environments."""
        return _safe(
            workflow.run_gwas, phenotype=phenotype, sample_col=sample_col,
            trait=trait, subgenomes=subgenomes, bed_prefixes=bed_prefixes,
            out_dir=out_dir, include_hadamard=include_hadamard, loco=loco,
            run_plots=run_plots, scan_jobs=scan_jobs, dry_run=dry_run)

    @mcp.tool()
    def prep_snps(gff: str, subgenome_map: str, bed_prefixes: dict[str, str],
                  out_dir: str, feature: str = "gene", id_attr: str = "ID",
                  flank_bp: int = 2000, min_snp: int = 2,
                  dry_run: bool = False) -> dict:
        """Build the snp_to_gene NPZ + genes TSV that ``interact`` needs, from a
        GFF + per-subgenome BEDs + a subgenome map (chrom,subgenome[,base_group]).
        GFF and .bim chromosome names must match exactly."""
        args = ["prep-snps", "--gff", gff, "--subgenome-map", subgenome_map,
                "--feature", feature, "--id-attr", id_attr,
                "--flank-bp", str(flank_bp), "--min-snp", str(min_snp),
                "--out-dir", out_dir]
        for s, p in bed_prefixes.items():
            args += ["--bed", f"{s}={p}"]
        step = _safe(workflow.run_cli, args=args, dry_run=dry_run)
        return {"ok": step.get("returncode") in (None, 0), "step": step,
                "out_dir": out_dir}

    @mcp.tool()
    def prep_homoeologs(subgenomes: list[str], genes_template: str, out: str,
                        from_table: str | None = None,
                        table_format: str = "long",
                        proteins: dict[str, str] | None = None,
                        subgenome_map: str | None = None,
                        diamond: str | None = None,
                        restrict_base_group: bool = True,
                        dry_run: bool = False) -> dict:
        """Build a wide gene_<S> homoeolog-group table for ``interact`` — from a user
        orthology table (``from_table``) or DIAMOND reciprocal best hits
        (``proteins`` + a 2.1.x ``diamond`` binary; 2.2.0 deadlocks).
        ``genes_template`` is the genes_{S}.tsv path from prep_snps."""
        try:
            mode, args = _prep_homoeologs_command(
                subgenomes, genes_template, out, from_table=from_table,
                table_format=table_format, proteins=proteins,
                subgenome_map=subgenome_map, diamond=diamond,
                restrict_base_group=restrict_base_group)
        except ValueError as e:
            return {"ok": False, "reason": str(e)}
        step = _safe(workflow.run_cli, args=args, dry_run=dry_run)
        return {"ok": step.get("returncode") in (None, 0), "mode": mode,
                "out": out, "step": step}

    @mcp.tool()
    def run_interaction(phenotype: str, sample_col: str, trait: str,
                        subgenomes: list[str], bed_prefixes: dict[str, str],
                        snp_to_gene: dict[str, str], out_dir: str,
                        pairs: str | None = None, triads: str | None = None,
                        groups: str | None = None,
                        hypothesis_unit: str | None = None,
                        subset_order: int = 2,
                        family_scope: str = "primary_only",
                        perm_b: int = 2000, n_jobs: int = 8,
                        statistic: str = "omniB",
                        null_variance: str = "smooth_pc4",
                        dry_run: bool = False) -> dict:
        """Run the unified homoeolog-group interaction workflow.

        Supply one ``groups`` table with ``gene_<subgenome>`` columns. omniB
        tests all pair edges in one family; four copies yield six edges and no
        fourth-order coefficient. The workflow generates YAML, validates,
        interacts, audits and summarizes. Legacy ``pairs``/``triads`` aliases
        remain accepted. ``null_variance`` defaults to ``smooth_pc4``, the
        heteroscedasticity-robust bootstrap null; ``homoscedastic`` is the
        sensitivity alternative.
        """
        return _safe(
            workflow.run_interaction, phenotype=phenotype, sample_col=sample_col,
            trait=trait, subgenomes=subgenomes, bed_prefixes=bed_prefixes,
            snp_to_gene=snp_to_gene, out_dir=out_dir, pairs=pairs, triads=triads,
            groups=groups, hypothesis_unit=hypothesis_unit,
            subset_order=subset_order, family_scope=family_scope,
            perm_b=perm_b, n_jobs=n_jobs, statistic=statistic,
            null_variance=null_variance, dry_run=dry_run)

    @mcp.tool()
    def audit_results(out_dir: str, dry_run: bool = False) -> dict:
        """Audit a finished fit or interaction directory: returns the overall
        status (e.g. AUDIT_COMPLETE, REVIEW_REQUIRED,
        INTERNAL_DISCOVERY_REPLICATION_REQUIRED), each record's status, flags
        and report paths. Report this status with any result."""
        return _safe(workflow.audit_results, out_dir=out_dir, dry_run=dry_run)

    @mcp.tool()
    def summarize_results(out_dir: str, trait: str) -> dict:
        """Summarize a finished run without recomputing: per-subgenome PVE,
        λ_GC and top hits for GWAS; family, discoveries, evidence-driving
        components and audit status for interaction."""
        return _safe(workflow.summarize_results, out_dir=out_dir, trait=trait)

    @mcp.tool()
    def make_plots(results_dir: str, formats: str = "png,pdf,svg",
                   dry_run: bool = False) -> dict:
        """Regenerate the publication figures from a finished run (no recompute)."""
        return _safe(workflow.make_plots, results_dir=results_dir,
                     formats=formats, dry_run=dry_run)

    return mcp


def main() -> int:
    """Console entry (``homoeogwas-mcp``): run the MCP server over stdio."""
    build_server().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
