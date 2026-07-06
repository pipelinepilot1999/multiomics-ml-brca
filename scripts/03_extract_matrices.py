#!/usr/bin/env python
"""03_extract_matrices.py - build aligned float32 feature matrices (fast path).

Writes data/processed/{expr,meth}.npz : X [n_samples,n_feat] float32, samples, features.

Preprocessing is global + label-blind only (no leakage; supervised selection happens
per-fold downstream). Methylation: drop probes >20% missing, row-median impute,
UNSUPERVISED top-50k variance prefilter (label-invariant -> does not affect the stage-09
permutation null; purely for tractability of 1000 permutations on 4 CPUs).

Perf: explicit dtype=float32 + numpy ops avoid pandas object-dtype inference on the
485k x 786 wide gzip, which was pathologically slow.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/ubuntu/survival-tcga")
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
METH_MISSING_MAX, METH_TOPK_VAR = 0.20, 50_000

def main():
    samples = pd.read_csv(PROC / "manifest.csv")["sample"].tolist()
    print(f"modeling samples: {len(samples)}", flush=True)

    print("reading expression ...", flush=True)
    expr = pd.read_csv(RAW / "TCGA-BRCA.rnaseq.tsv", sep="\t", index_col=0,
                       usecols=["sample"] + samples,
                       dtype={s: np.float32 for s in samples})
    expr = expr[samples].T                              # samples x genes
    genes = expr.columns.to_numpy()
    Xe = expr.to_numpy(np.float32)
    keep = np.nanstd(Xe, axis=0) > 0
    Xe, genes = Xe[:, keep], genes[keep]
    np.savez(PROC / "expr.npz", X=Xe, samples=np.array(samples), features=genes)
    print(f"  expr matrix: {Xe.shape}", flush=True)

    print("reading methylation (float32, may take a few min) ...", flush=True)
    meth = pd.read_csv(RAW / "TCGA-BRCA.methyl450.tsv.gz", sep="\t", index_col=0,
                       usecols=["sample"] + samples,
                       dtype={s: np.float32 for s in samples})
    probes = meth.index.to_numpy()
    M = meth[samples].to_numpy(np.float32)              # probes x samples
    print(f"  parsed methylation: {M.shape}", flush=True)

    miss = np.isnan(M).mean(axis=1)
    keep = miss <= METH_MISSING_MAX
    M, probes = M[keep], probes[keep]
    print(f"  probes after <= {METH_MISSING_MAX:.0%} missing: {M.shape[0]}", flush=True)
    # row-median impute (label-blind)
    med = np.nanmedian(M, axis=1)
    idx = np.where(np.isnan(M))
    M[idx] = np.take(med, idx[0])
    var = M.var(axis=1)
    top = np.argsort(var)[::-1][:METH_TOPK_VAR]
    M, probes = M[top], probes[top]
    Xm = M.T.copy()                                     # samples x probes
    np.savez(PROC / "meth.npz", X=Xm.astype(np.float32),
             samples=np.array(samples), features=probes)
    print(f"  meth matrix (top-{METH_TOPK_VAR} var): {Xm.shape}", flush=True)
    print("done.", flush=True)

if __name__ == "__main__":
    main()
