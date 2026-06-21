"""HomoeoGWAS: allopolyploid GWAS framework.

Per-subgenome random-effect LMM plus a homoeolog Hadamard kernel for
co-association across paralogous subgenomes (e.g. wheat AABBDD).

Main entry points (also re-exported at the package top level):

- ``load_bed_hardcall``: read PLINK1 BED hard calls as a {0,1,2,NaN} float32 matrix.
- ``vcf_to_bed``: convert VCF.gz to PLINK bed/bim/fam via plink2.
- ``compute_grm`` / ``compute_grm_panel``: per-subgenome VanRaden GRM.
- ``hadamard_kernel``: elementwise product across subgenome GRMs.
- ``sum_kernel``: weighted sum of subgenome GRMs (additive baseline).
- ``normalize_kernel``: trace/frobenius scaling so kernels are LMM-comparable.
- ``fit_reml`` / ``fit_multi_reml``: single- and multi-kernel REML LMM.
"""
from .grm import (
    GRMPart,
    compute_grm,
    compute_grm_panel,
    compute_grm_parts,
    compute_loco_grm_parts,
    loco_grm_from_parts,
)
from .io import GenoChunk, load_bed_hardcall, vcf_to_bed
from .kernel import hadamard_kernel, normalize_kernel, sum_kernel
from .lmm import MultiREMLResult, REMLResult, fit_multi_reml, fit_reml
from .scan import (
    LOCOContext,
    ScanContext,
    ScanResult,
    StreamScanSummary,
    build_loco_scan_contexts,
    build_scan_context,
    lambda_gc,
    scan_bed_stream,
    scan_bed_stream_loco,
    scan_snps,
    scan_snps_loco,
)

__version__ = "1.0.2"

__all__ = [
    "GRMPart",
    "GenoChunk",
    "LOCOContext",
    "MultiREMLResult",
    "REMLResult",
    "ScanContext",
    "ScanResult",
    "StreamScanSummary",
    "build_loco_scan_contexts",
    "build_scan_context",
    "compute_grm",
    "compute_grm_panel",
    "compute_grm_parts",
    "compute_loco_grm_parts",
    "fit_multi_reml",
    "fit_reml",
    "hadamard_kernel",
    "lambda_gc",
    "load_bed_hardcall",
    "loco_grm_from_parts",
    "normalize_kernel",
    "scan_bed_stream",
    "scan_bed_stream_loco",
    "scan_snps",
    "scan_snps_loco",
    "sum_kernel",
    "vcf_to_bed",
    "__version__",
]
