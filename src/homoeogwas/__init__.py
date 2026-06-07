"""HomoeoGWAS — allopolyploid GWAS framework.

Core idea: per-subgenome random-effect LMM + homoeolog Hadamard kernel for
co-association across paralogous subgenomes (e.g. wheat AABBDD).

Public API (v0.1, M2.1 + M2.2 + M2.3 + M2.4):
- io.GenoChunk / homoeogwas.GenoChunk
- io.load_bed_hardcall / homoeogwas.load_bed_hardcall
    -> read PLINK1 BED hard calls as {0,1,2,NaN} float32 matrix
- io.vcf_to_bed / homoeogwas.vcf_to_bed
    -> convert VCF.gz to PLINK bed/bim/fam via plink2
- grm.compute_grm / homoeogwas.compute_grm
    -> per-subgenome VanRaden GRM
- grm.compute_grm_panel / homoeogwas.compute_grm_panel
    -> compute G for each subgenome from panel manifest
- kernel.hadamard_kernel / homoeogwas.hadamard_kernel
    -> elementwise product across subgenome GRMs (homoeolog co-association)
- kernel.sum_kernel / homoeogwas.sum_kernel
    -> weighted sum of subgenome GRMs (additive baseline)
- kernel.normalize_kernel / homoeogwas.normalize_kernel
    -> trace/frobenius scale so kernels are LMM-comparable
- lmm.fit_reml / homoeogwas.fit_reml
    -> single-kernel REML LMM (CPU baseline, GPU via CuPy if importable)
- lmm.REMLResult / homoeogwas.REMLResult
- lmm.fit_multi_reml / homoeogwas.fit_multi_reml
    -> M2.4 multi-kernel REML; per-subgenome variance components
- lmm.MultiREMLResult / homoeogwas.MultiREMLResult
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

__version__ = "1.0.0"

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
