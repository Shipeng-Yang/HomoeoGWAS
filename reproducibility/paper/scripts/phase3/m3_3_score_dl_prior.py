#!/usr/bin/env python3
"""Phase 3 M3.3 — zero-shot LLR scoring of candidate SNPs with PlantCaduceus
and/or AgroNT.

  primary score = log P(alt | masked_context) − log P(ref | masked_context)

PlantCaduceus_l32 (kuleshov-group/PlantCaduceus_l32):
  225M Caduceus DNA-LM, plant pre-training, 512 bp single-nt window,
  trust_remote_code=True. Score the centre base.

AgroNT-1B (InstaDeepAI/agro-nucleotide-transformer-1b):
  1B nucleotide-transformer, 6-mer tokeniser, 1024 tokens ≈ 6144 bp.
  Score the 6-mer token containing the SNP; phase-shift 6× and take median.

Outputs per model (gzip TSV):
  snp_id chrom pos ref alt model llr_signed llr_abs ref_logp alt_logp
  window_bp token_phase n_masked_tokens runtime_ms status
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


# =====================================================================
# Model wrappers
# =====================================================================


class PlantCaduceusScorer:
    """PlantCaduceus_l32 single-nt masked-LM scorer for centre-base SNPs."""

    HF_NAME = "kuleshov-group/PlantCaduceus_l32"
    WINDOW_BP = 512                     # centred on SNP, mask the centre base
    TOKENS = ("A", "C", "G", "T")

    def __init__(self, device: str = "cuda:0", dtype: str = "fp16"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
        print(f"  loading {self.HF_NAME} on {device} ({dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.HF_NAME, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.HF_NAME, trust_remote_code=True, torch_dtype=torch_dtype
        ).to(device).eval()
        self.device = device
        self.torch = torch
        # tokeniser id for mask + A/C/G/T base tokens
        self.mask_id = self.tokenizer.mask_token_id
        # PlantCaduceus tokenizer uses lowercase a/c/g/t in its vocab; tokenizer
        # encoding is case-insensitive but ``convert_tokens_to_ids`` is literal.
        self.base_to_id = {
            b: self.tokenizer.convert_tokens_to_ids(b.lower()) for b in self.TOKENS
        }
        assert all(self.base_to_id[b] is not None and self.base_to_id[b] != self.tokenizer.unk_token_id
                   for b in self.TOKENS), f"base tokens missing: {self.base_to_id}"

    def score_window(self, seq: str, ref: str, alt: str, pos_in_seq: int):
        """Compute log P(ref/alt | masked centre); seq has the SNP at pos_in_seq."""
        import torch
        ids = self.tokenizer(seq, return_tensors="pt",
                              add_special_tokens=False).input_ids.to(self.device)
        ids[0, pos_in_seq] = self.mask_id
        with torch.no_grad():
            out = self.model(input_ids=ids).logits[0, pos_in_seq]
            logp = torch.log_softmax(out.float(), dim=-1)
        ref_id = self.base_to_id[ref]
        alt_id = self.base_to_id[alt]
        return float(logp[ref_id].item()), float(logp[alt_id].item())


class AgroNTScorer:
    """AgroNT-1B 6-mer masked-LM scorer with 6-phase median."""

    HF_NAME = "InstaDeepAI/agro-nucleotide-transformer-1b"
    WINDOW_BP = 6144
    KMER = 6

    def __init__(self, device: str = "cuda:1", dtype: str = "fp16"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
        print(f"  loading {self.HF_NAME} on {device} ({dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.HF_NAME, trust_remote_code=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.HF_NAME, trust_remote_code=True, torch_dtype=torch_dtype
        ).to(device).eval()
        self.device = device
        self.torch = torch
        self.mask_id = self.tokenizer.mask_token_id

    def _tokenise(self, seq: str):
        return self.tokenizer(seq, return_tensors="pt",
                                add_special_tokens=False).input_ids.to(self.device)

    def score_phase(self, seq: str, snp_pos_in_seq: int, ref: str, alt: str,
                     phase: int):
        """Score one 6-mer phase; returns (ref_logp, alt_logp, status, k_tok_idx)."""
        import torch
        # shift sequence by `phase` so the SNP falls at different positions
        # within its 6-mer token. Clip / pad if necessary.
        start = phase
        end = phase + (len(seq) - phase) // self.KMER * self.KMER
        s = seq[start:end]
        snp_pos_shifted = snp_pos_in_seq - start
        if snp_pos_shifted < 0 or snp_pos_shifted >= len(s):
            return None, None, "PHASE_OOB", -1
        k_tok_idx = snp_pos_shifted // self.KMER
        # Build ref / alt sequences by substituting the SNP centre base
        s_ref = s[:snp_pos_shifted] + ref + s[snp_pos_shifted+1:]
        s_alt = s[:snp_pos_shifted] + alt + s[snp_pos_shifted+1:]
        # Tokenise both
        ids_ref = self._tokenise(s_ref)
        ids_alt = self._tokenise(s_alt)
        if k_tok_idx >= ids_ref.shape[1]:
            return None, None, "TOKEN_OOB", k_tok_idx
        # mask the SNP-containing token in BOTH sequences, get logp at that pos
        ids_ref_mask = ids_ref.clone()
        ids_alt_mask = ids_alt.clone()
        ids_ref_mask[0, k_tok_idx] = self.mask_id
        ids_alt_mask[0, k_tok_idx] = self.mask_id
        with torch.no_grad():
            log_ref = torch.log_softmax(
                self.model(input_ids=ids_ref_mask).logits[0, k_tok_idx].float(), dim=-1)
            log_alt = torch.log_softmax(
                self.model(input_ids=ids_alt_mask).logits[0, k_tok_idx].float(), dim=-1)
        # Use the token id of the ACTUAL ref/alt 6-mer
        ref_tok = int(ids_ref[0, k_tok_idx].item())
        alt_tok = int(ids_alt[0, k_tok_idx].item())
        return float(log_ref[ref_tok].item()), float(log_alt[alt_tok].item()), "OK", k_tok_idx


# =====================================================================
# FASTA window helper
# =====================================================================


def fetch_window(fa, chrom: str, pos1: int, half_bp: int,
                  reject_n_in_flank_bp: int = 50):
    """Fetch ±half_bp around 1-based pos. Returns (seq, snp_offset, status).

    Wheat IWGSC v1.0 has ~21 % of M3.2 candidate windows with N bases (assembly
    gaps); 6-mer/single-nt tokenisers will produce UNK tokens that cause CUDA
    asserts in AgroNT/PlantCaduceus. We:
      * fail with WINDOW_HAS_N_NEAR_SNP if the immediate ±50 bp around the SNP
        contains N (the SNP's 6-mer token cannot be scored cleanly)
      * pass otherwise; the flank can have N (they become UNK far from the SNP
        and do not corrupt the score)
    """
    start = max(0, pos1 - 1 - half_bp)
    end = pos1 + half_bp                # pysam end is 0-based exclusive
    seq = fa.fetch(chrom, start, end).upper()
    snp_offset = (pos1 - 1) - start     # 0-based offset in seq
    if snp_offset < 0 or snp_offset >= len(seq):
        return None, -1, "WINDOW_OOB"
    if len(seq) < 2 * half_bp:
        return seq, snp_offset, "WINDOW_TRUNCATED"
    # check immediate flank for N
    flank_lo = max(0, snp_offset - reject_n_in_flank_bp)
    flank_hi = min(len(seq), snp_offset + reject_n_in_flank_bp + 1)
    if "N" in seq[flank_lo:flank_hi]:
        return None, -1, "WINDOW_HAS_N_NEAR_SNP"
    # Far-flank N → A, so tokenizer never emits UNK / out-of-vocab id
    # (UNK tokens can trigger CUDA asserts when their id != tokenizer.unk_token_id
    # in some HF models). Replacement is benign because these positions are
    # far from the masked centre and only contribute to context.
    if "N" in seq:
        seq = seq.replace("N", "A")
    return seq, snp_offset, "OK"


# =====================================================================
# Main
# =====================================================================


def main():
    ap = argparse.ArgumentParser(description="M3.3 zero-shot DL prior scoring")
    ap.add_argument("--model", choices=["plantcad", "agront"], required=True)
    ap.add_argument("--candidates", default=str(
        ROOT / "results/phase3/m3_3_dl_prior/wheat_watkins/candidates.tsv.gz"))
    ap.add_argument("--fasta", default=
        "/mnt/nvme/wheat_ref/iwgsc_refseqv1.0_all_chromosomes/"
        "iwgsc_refseqv1.0_all_chromosomes.fa")
    ap.add_argument("--out-dir", default=str(
        ROOT / "results/phase3/m3_3_dl_prior/wheat_watkins"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--max-snps", type=int, default=None,
                    help="cap to first N candidates (pilot)")
    ap.add_argument("--shard", type=str, default="0/1",
                    help="<idx>/<total>, e.g. 0/2 takes rows [0::2], 1/2 takes [1::2]")
    ap.add_argument("--n-phases-agront", type=int, default=6)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pysam
    print(f"=== M3.3 score DL prior — model={args.model} device={args.device} ===")
    cand = pd.read_csv(args.candidates, sep="\t", compression="gzip")
    # Scorable = direct + reverse-complement (strand-flipped) REF matches. For
    # RC rows ref_fasta is still the + strand FASTA base, so seq[center]==ref
    # holds and the alt is already complemented to the + strand frame.
    cand = cand[cand["ref_match_status"].isin(
        ["A1_IS_REF", "A2_IS_REF", "A1_IS_REF_RC", "A2_IS_REF_RC"]
    )].reset_index(drop=True)
    if args.max_snps:
        cand = cand.head(args.max_snps).reset_index(drop=True)
    # Shard for multi-GPU split
    shard_idx, shard_total = (int(x) for x in args.shard.split("/"))
    if shard_total > 1:
        cand = cand.iloc[shard_idx::shard_total].reset_index(drop=True)
        print(f"  shard {shard_idx}/{shard_total}: {len(cand)} candidates")
    print(f"  scoring {len(cand)} candidates (REF-matched, shard {args.shard})")

    if args.model == "plantcad":
        scorer = PlantCaduceusScorer(device=args.device, dtype=args.dtype)
        half_bp = scorer.WINDOW_BP // 2
    else:
        scorer = AgroNTScorer(device=args.device, dtype=args.dtype)
        half_bp = scorer.WINDOW_BP // 2

    fa = pysam.FastaFile(args.fasta)
    shard_suffix = f"_shard{shard_idx}of{shard_total}" if shard_total > 1 else ""
    out_path = out_dir / f"dl_scores_{args.model}{shard_suffix}.tsv.gz"
    rows: list[dict] = []
    t0 = time.time()
    n_ok = n_err = 0
    for i, r in cand.iterrows():
        # Prefer fasta_chrom column (added by m3_3_v2_prepare_candidates) so
        # panel chrom names (e.g. chrA01, A01) are translated to fasta chrom
        # (e.g. NC_027757.2, CM032202.1) for pysam.fetch.
        chrom = str(r["fasta_chrom"]) if "fasta_chrom" in r.index and pd.notna(r.get("fasta_chrom")) else str(r["chrom"])
        pos = int(r["pos"])
        ref = str(r["ref_fasta"])
        alt = str(r["alt_fasta"])
        if alt not in ("A","C","G","T") or ref not in ("A","C","G","T"):
            rows.append(dict(snp_id=r["snp_id"], chrom=chrom, pos=pos,
                              ref=ref, alt=alt, model=args.model,
                              llr_signed=np.nan, llr_abs=np.nan,
                              ref_logp=np.nan, alt_logp=np.nan,
                              window_bp=scorer.WINDOW_BP, token_phase=-1,
                              n_masked_tokens=0, runtime_ms=0,
                              status="BAD_ALLELE"))
            n_err += 1
            continue
        seq, snp_off, w_status = fetch_window(fa, chrom, pos, half_bp)
        if seq is None or w_status != "OK":
            rows.append(dict(snp_id=r["snp_id"], chrom=chrom, pos=pos,
                              ref=ref, alt=alt, model=args.model,
                              llr_signed=np.nan, llr_abs=np.nan,
                              ref_logp=np.nan, alt_logp=np.nan,
                              window_bp=scorer.WINDOW_BP, token_phase=-1,
                              n_masked_tokens=0, runtime_ms=0,
                              status=w_status))
            n_err += 1
            continue
        # Verify FASTA base matches ref
        if seq[snp_off] != ref:
            rows.append(dict(snp_id=r["snp_id"], chrom=chrom, pos=pos,
                              ref=ref, alt=alt, model=args.model,
                              llr_signed=np.nan, llr_abs=np.nan,
                              ref_logp=np.nan, alt_logp=np.nan,
                              window_bp=scorer.WINDOW_BP, token_phase=-1,
                              n_masked_tokens=0, runtime_ms=0,
                              status=f"FASTA_REF_MISMATCH={seq[snp_off]}"))
            n_err += 1
            continue
        t_snp = time.time()
        if args.model == "plantcad":
            try:
                ref_logp, alt_logp = scorer.score_window(seq, ref, alt, snp_off)
                llr_signed = alt_logp - ref_logp
                rows.append(dict(snp_id=r["snp_id"], chrom=chrom, pos=pos,
                                  ref=ref, alt=alt, model=args.model,
                                  llr_signed=float(llr_signed),
                                  llr_abs=float(abs(llr_signed)),
                                  ref_logp=float(ref_logp),
                                  alt_logp=float(alt_logp),
                                  window_bp=scorer.WINDOW_BP, token_phase=0,
                                  n_masked_tokens=1,
                                  runtime_ms=int((time.time()-t_snp)*1000),
                                  status="OK"))
                n_ok += 1
            except Exception as e:
                rows.append(dict(snp_id=r["snp_id"], chrom=chrom, pos=pos,
                                  ref=ref, alt=alt, model=args.model,
                                  llr_signed=np.nan, llr_abs=np.nan,
                                  ref_logp=np.nan, alt_logp=np.nan,
                                  window_bp=scorer.WINDOW_BP, token_phase=-1,
                                  n_masked_tokens=0,
                                  runtime_ms=int((time.time()-t_snp)*1000),
                                  status=f"INFER_ERR:{type(e).__name__}"))
                n_err += 1
        else:                                       # agront
            llrs = []
            for phase in range(args.n_phases_agront):
                ref_lp, alt_lp, status, kix = scorer.score_phase(
                    seq, snp_off, ref, alt, phase)
                if status == "OK":
                    llrs.append((phase, ref_lp, alt_lp, alt_lp - ref_lp, kix))
            if not llrs:
                rows.append(dict(snp_id=r["snp_id"], chrom=chrom, pos=pos,
                                  ref=ref, alt=alt, model=args.model,
                                  llr_signed=np.nan, llr_abs=np.nan,
                                  ref_logp=np.nan, alt_logp=np.nan,
                                  window_bp=scorer.WINDOW_BP, token_phase=-1,
                                  n_masked_tokens=0,
                                  runtime_ms=int((time.time()-t_snp)*1000),
                                  status="ALL_PHASES_FAILED"))
                n_err += 1
                continue
            signed_arr = np.array([x[3] for x in llrs])
            median = float(np.median(signed_arr))
            best_phase = llrs[int(np.argmin(np.abs(signed_arr - median)))][0]
            rows.append(dict(snp_id=r["snp_id"], chrom=chrom, pos=pos,
                              ref=ref, alt=alt, model=args.model,
                              llr_signed=median,
                              llr_abs=float(abs(median)),
                              ref_logp=float(np.median([x[1] for x in llrs])),
                              alt_logp=float(np.median([x[2] for x in llrs])),
                              window_bp=scorer.WINDOW_BP,
                              token_phase=int(best_phase),
                              n_masked_tokens=int(len(llrs)),
                              runtime_ms=int((time.time()-t_snp)*1000),
                              status="OK"))
            n_ok += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(cand):
            avg_ms = (time.time()-t0)/(i+1)*1000
            print(f"  [{args.model}] {i+1}/{len(cand)} "
                  f"({n_ok} ok / {n_err} err) avg {avg_ms:.0f} ms/SNP")
    fa.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    runtime = time.time() - t0
    print(f"\nwrote {out_path}  ({n_ok} ok / {n_err} err  {runtime:.0f}s)")

    summary = {
        "model": args.model,
        "n_input": int(len(cand)),
        "n_ok": int(n_ok),
        "n_err": int(n_err),
        "runtime_sec": round(runtime, 1),
        "device": args.device,
        "dtype": args.dtype,
        "window_bp": scorer.WINDOW_BP,
        "outputs": {"scores": str(out_path)},
    }
    if n_ok > 0:
        ok_df = df[df["status"] == "OK"]
        summary["llr_abs"] = {
            "min": float(ok_df["llr_abs"].min()),
            "max": float(ok_df["llr_abs"].max()),
            "median": float(ok_df["llr_abs"].median()),
            "mean": float(ok_df["llr_abs"].mean()),
            "std": float(ok_df["llr_abs"].std()),
        }
    summary_path = out_dir / f"dl_scores_{args.model}{shard_suffix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
