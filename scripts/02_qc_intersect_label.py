#!/usr/bin/env python
"""
02_qc_intersect_label.py — sample QC, layer intersection, and the N=3yr PFI label.

Outputs data/processed/manifest.csv: one row per modeling sample with
  sample, patient, label (0/1), pam50, t_yr, pfi
plus a console report of N at each merge step (interview questions live here).

Design decisions (surfaced, not silent):
  - Primary tumor only: barcode sample-type suffix "-01" (drop normals "-11", mets).
  - Dedup: one sample per patient (first by sorted barcode) — TCGA rarely has
    multiple primaries; we keep it deterministic and report collisions.
  - Label horizon N=3yr with require-observed-status rule (see 01_label_horizon).
  - Intersection is on SAMPLE barcode across expression + methylation + label.
"""
from pathlib import Path
import pandas as pd

ROOT = Path("/home/ubuntu/survival-tcga")
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

N_YEARS = 3
DAYS_PER_YR = 365.25

def header_samples(path, gz=False):
    import gzip
    op = gzip.open if gz else open
    with op(path, "rt") as fh:
        cols = fh.readline().rstrip("\n").split("\t")
    return [c for c in cols[1:] if c.startswith("TCGA")]

def primary_only(barcodes):
    return sorted({b for b in barcodes if b[13:15] == "01"})

def main():
    rep = []
    # --- sample rosters per layer (header only; cheap) ---
    expr_s = primary_only(header_samples(RAW / "TCGA-BRCA.rnaseq.tsv"))
    meth_s = primary_only(header_samples(RAW / "TCGA-BRCA.methyl450.tsv.gz", gz=True))
    rep.append(("expression primary-tumor samples", len(expr_s)))
    rep.append(("methylation primary-tumor samples", len(meth_s)))

    # --- survival/label ---
    sv = pd.read_csv(RAW / "TCGA-BRCA.survival.tsv", sep="\t", low_memory=False)
    sv = sv[sv["cancer type abbreviation"] == "BRCA"].copy()
    sv = sv[sv["sample"].str.slice(13, 15) == "01"]
    sv = sv.dropna(subset=["PFI", "PFI.time"])
    sv["t_yr"] = sv["PFI.time"].astype(float) / DAYS_PER_YR
    sv["PFI"] = sv["PFI"].astype(int)
    rep.append(("survival primary-tumor w/ PFI", len(sv)))

    # apply N=3yr label with require-observed rule
    pos = (sv["PFI"] == 1) & (sv["t_yr"] <= N_YEARS)
    neg = sv["t_yr"] > N_YEARS
    sv_lab = sv[pos | neg].copy()
    sv_lab["label"] = pos[pos | neg].astype(int)
    rep.append((f"labelled (N={N_YEARS}yr, ambiguous dropped)", len(sv_lab)))
    rep.append(("  positives", int(sv_lab["label"].sum())))
    rep.append(("  negatives", int((sv_lab["label"] == 0).sum())))

    # --- clinical PAM50 (covariate/stratifier, sparse) ---
    cl = pd.read_csv(RAW / "TCGA-BRCA.clinical.tsv", sep="\t", low_memory=False)
    pam = cl[["sampleID", "PAM50Call_RNAseq"]].rename(
        columns={"sampleID": "sample", "PAM50Call_RNAseq": "pam50"})

    # --- intersect: label ∩ expression ∩ methylation ---
    lab_s = set(sv_lab["sample"])
    inter = sorted(lab_s & set(expr_s) & set(meth_s))
    rep.append(("label ∩ expression ∩ methylation (MODELING SET)", len(inter)))

    man = sv_lab[sv_lab["sample"].isin(inter)][["sample", "_PATIENT", "label", "t_yr", "PFI"]]
    man = man.rename(columns={"_PATIENT": "patient"})
    man = man.merge(pam, on="sample", how="left")
    # dedup patients (report collisions)
    dup = man["patient"].duplicated().sum()
    man = man.sort_values("sample").drop_duplicates("patient", keep="first").reset_index(drop=True)
    rep.append((f"  patient collisions dropped", int(dup)))
    rep.append(("FINAL modeling samples", len(man)))
    rep.append(("  final positives", int(man["label"].sum())))
    rep.append(("  final negatives", int((man['label'] == 0).sum())))
    rep.append(("  event rate", round(man["label"].mean(), 3)))
    rep.append(("  PAM50 available", int(man["pam50"].notna().sum())))

    man.to_csv(PROC / "manifest.csv", index=False)
    print(f"\n{'STEP':<52}{'N':>8}")
    print("-" * 60)
    for k, v in rep:
        print(f"{k:<52}{v:>8}")
    print(f"\nwrote {PROC/'manifest.csv'}  ({len(man)} samples)")

if __name__ == "__main__":
    main()
