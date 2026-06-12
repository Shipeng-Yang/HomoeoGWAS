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
        exist and flags the integer-sample-id pitfall (which silently breaks the
        genotype↔phenotype join). Returns advice, not a stack trace."""
        return _safe(workflow.check_phenotype_inputs, phenotype=phenotype,
                     sample_col=sample_col, trait=trait)

    @mcp.tool()
    def split_genotype(species_yaml: str, out_dir: str, threads: int = 8,
                       dry_run: bool = False) -> dict:
        """Split one VCF into per-subgenome PLINK BEDs using a species YAML."""
        return _safe(workflow.split_genotype, species_yaml=species_yaml,
                     out_dir=out_dir, threads=threads, dry_run=dry_run)

    @mcp.tool()
    def run_gwas(phenotype: str, sample_col: str, trait: str,
                 subgenomes: list[str], bed_prefixes: dict[str, str],
                 out_dir: str, include_hadamard: bool = False,
                 loco: bool = False, run_plots: bool = True,
                 dry_run: bool = False) -> dict:
        """Run a subgenome-stratified GWAS from breeder-level inputs: generates
        the fit YAML, validates, runs ``fit`` (+ figures), and returns the
        per-subgenome PVE, λ_GC, top hits and figure paths. ``bed_prefixes`` maps
        each subgenome label to a PLINK prefix."""
        return _safe(
            workflow.run_gwas, phenotype=phenotype, sample_col=sample_col,
            trait=trait, subgenomes=subgenomes, bed_prefixes=bed_prefixes,
            out_dir=out_dir, include_hadamard=include_hadamard, loco=loco,
            run_plots=run_plots, dry_run=dry_run)

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
        return {"ok": True, "step": _safe(workflow.run_cli, args=args,
                                          dry_run=dry_run), "out_dir": out_dir}

    @mcp.tool()
    def prep_homoeologs(subgenomes: list[str], genes_template: str, out: str,
                        from_table: str | None = None,
                        proteins: dict[str, str] | None = None,
                        subgenome_map: str | None = None,
                        diamond: str | None = None,
                        restrict_base_group: bool = True,
                        dry_run: bool = False) -> dict:
        """Build the gene_<S> pair/triad table for ``interact`` — from a user
        orthology table (``from_table``) or DIAMOND reciprocal best hits
        (``proteins`` + a 2.1.x ``diamond`` binary; 2.2.0 deadlocks).
        ``genes_template`` is the genes_{S}.tsv path from prep_snps."""
        try:
            mode = workflow.infer_interaction_mode(subgenomes)
        except ValueError as e:
            return {"ok": False, "reason": str(e)}
        args = ["prep-homoeologs", "--mode", mode,
                "--subgenomes", ",".join(subgenomes),
                "--genes", genes_template, "--out", out]
        if from_table:
            args += ["--from-table", from_table]
        elif proteins:
            args += ["--method", "diamond-rbh"]
            for s, p in proteins.items():
                args += ["--proteins", f"{s}={p}"]
            if restrict_base_group and subgenome_map:
                args += ["--restrict-base-group", "--subgenome-map", subgenome_map]
            if diamond:
                args += ["--diamond", diamond]
        else:
            return {"ok": False, "reason": "provide from_table or proteins"}
        return {"ok": True, "mode": mode, "out": out,
                "step": _safe(workflow.run_cli, args=args, dry_run=dry_run)}

    @mcp.tool()
    def run_interaction(phenotype: str, sample_col: str, trait: str,
                        subgenomes: list[str], bed_prefixes: dict[str, str],
                        snp_to_gene: dict[str, str], out_dir: str,
                        pairs: str | None = None, triads: str | None = None,
                        perm_b: int = 200, n_jobs: int = 8,
                        dry_run: bool = False) -> dict:
        """Run the homoeolog-pair interaction test; mode is inferred from ploidy
        (2→pairwise needs ``pairs``, 3→triad needs ``triads``; 4+ is refused —
        run subsets). Generates the interact YAML and runs it."""
        return _safe(
            workflow.run_interaction, phenotype=phenotype, sample_col=sample_col,
            trait=trait, subgenomes=subgenomes, bed_prefixes=bed_prefixes,
            snp_to_gene=snp_to_gene, out_dir=out_dir, pairs=pairs, triads=triads,
            perm_b=perm_b, n_jobs=n_jobs, dry_run=dry_run)

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
