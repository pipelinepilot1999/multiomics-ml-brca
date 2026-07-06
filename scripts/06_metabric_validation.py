#!/usr/bin/env python
"""06_metabric_validation.py - external validation of the EXPRESSION arm only.

Scope, stated out loud: METABRIC validates the expression baseline as an
out-of-distribution check. The FUSION claim stays single-cohort (TCGA) - METABRIC
has no matched 450k methylation, so we do NOT externally validate the integrated model.

Cross-platform harmonisation: TCGA is RNA-seq log2, METABRIC is Illumina microarray.
Match genes by Hugo symbol (intersection); z-score each gene independently within each
cohort (rank-preserving, removes platform scale/offset). Train on ALL TCGA (expr, 3yr
PFI label); predict METABRIC 3yr relapse-free-survival (RFS) label.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from common import load_labels, PROC, RES, RAW

HORIZON_M = 36  # 3 years

def load_metabric():
    expr = pd.read_csv(RAW / "metabric_mrna.txt", sep="\t", low_memory=False)
    expr = expr.drop(columns=[c for c in ["Entrez_Gene_Id"] if c in expr.columns])
    expr = expr.dropna(subset=["Hugo_Symbol"]).drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")
    with open(RAW / "metabric_clinical_patient.txt") as fh:
        lines = [l for l in fh if not l.startswith("#")]
    clin = pd.read_csv(pd.io.common.StringIO("".join(lines)), sep="\t", low_memory=False)
    return expr, clin

def metabric_label(clin):
    c = clin.rename(columns={"PATIENT_ID": "sample"})
    st = c["RFS_STATUS"].astype(str).str.extract(r"(\d)").astype(float)[0]
    mo = pd.to_numeric(c["RFS_MONTHS"], errors="coerce")
    pos = (st == 1) & (mo <= HORIZON_M)
    neg = mo > HORIZON_M
    keep = pos | neg
    return pd.DataFrame({"sample": c["sample"], "label": pos.astype(int)})[keep].dropna()

def zscore_genes(df):
    return (df - df.mean(0)) / (df.std(0).replace(0, np.nan))

def main():
    d = np.load(PROC / "expr.npz", allow_pickle=True)
    Xe, genes = d["X"], list(d["features"])
    y_tcga = load_labels()["label"].to_numpy(int)
    tcga = pd.DataFrame(Xe, columns=genes)

    expr_mb, clin = load_metabric()
    lab = metabric_label(clin)
    mb = expr_mb.T
    common_s = mb.index.intersection(lab["sample"])
    mb = mb.loc[common_s]
    lab = lab.set_index("sample").loc[common_s]
    y_mb = lab["label"].to_numpy(int)

    shared = sorted(set(genes) & set(mb.columns))
    print(f"TCGA genes={len(genes)}  METABRIC genes={mb.shape[1]}  shared={len(shared)}")
    print(f"METABRIC labelled={len(y_mb)}  events={y_mb.sum()}  rate={y_mb.mean():.3f}")

    Xtr = zscore_genes(tcga[shared]).fillna(0).to_numpy()
    Xte = zscore_genes(mb[shared].astype(float)).fillna(0).to_numpy()
    pipe = Pipeline([
        ("select", SelectKBest(f_classif, k=min(500, len(shared)))),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1,
                                   solver="saga", max_iter=5000, class_weight="balanced")),
    ]).fit(Xtr, y_tcga)
    p = pipe.predict_proba(Xte)[:, 1]
    out = dict(metabric_n=len(y_mb), metabric_events=int(y_mb.sum()),
               metabric_event_rate=round(float(y_mb.mean()), 3), shared_genes=len(shared),
               auroc=round(float(roc_auc_score(y_mb, p)), 4),
               auprc=round(float(average_precision_score(y_mb, p)), 4))
    pd.Series(out).to_csv(RES / "06_metabric_summary.csv")
    print("\n=== METABRIC EXPRESSION-ARM VALIDATION ===")
    for k, v in out.items():
        print(f"  {k:20s} {v}")

if __name__ == "__main__":
    main()
