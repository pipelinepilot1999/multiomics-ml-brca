#!/usr/bin/env python
"""07_biology_check.py - feature-level sanity check.

Fits the elastic-net on ALL samples per layer (this is interpretation, not evaluation
- honest CV numbers come from stage 04), extracts nonzero-coefficient features, maps
methylation cg-probes to genes, and checks against curated breast-cancer loci with known
methylation regulation / prognostic value. Also asks: do the two layers point at the
same genes (concordant biology) or different ones (orthogonal signal)?
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from common import load_layer, load_labels, make_pipe, RAW, RES

# curated BRCA loci with documented promoter methylation / prognostic role
KNOWN = {"BRCA1","ESR1","PITX2","CDH1","RASSF1A","GSTP1","TWIST1","CCND2","APC",
         "FOXA1","PGR","FOXC1","SFRP1","WIF1","DAPK1","CDKN2A","MGMT","HIN1","SCGB3A1"}

def selected(Xname, y, k=500, topn=40):
    X, _, feats = load_layer(Xname)
    pipe = make_pipe("lasso", k).fit(X, y)
    mask = pipe.named_steps["select"].get_support()
    coef = pipe.named_steps["clf"].coef_[0]
    sub = feats[mask]
    df = pd.DataFrame({"feature": sub, "coef": coef})
    df = df[df["coef"] != 0].reindex(df["coef"].abs().sort_values(ascending=False).index)
    return df.head(topn).reset_index(drop=True)

def map_probes(probes):
    pm = pd.read_csv(RAW / "methyl450_probemap.tsv", sep="\t")
    pm.columns = ["id","gene","chrom","start","end","strand"]
    m = pm.set_index("id")["gene"].to_dict()
    out = {}
    for p in probes:
        g = m.get(p, ".")
        out[p] = [x for x in str(g).split(",") if x and x != "."]
    return out

def main():
    y = load_labels()["label"].to_numpy(int)
    expr_sel = selected("expr", y)
    meth_sel = selected("meth", y)
    p2g = map_probes(meth_sel["feature"].tolist())
    meth_sel["genes"] = meth_sel["feature"].map(lambda p: ",".join(p2g.get(p, [])))

    expr_genes = set(expr_sel["feature"])
    meth_genes = set(g for gs in p2g.values() for g in gs)

    expr_known = sorted(expr_genes & KNOWN)
    meth_known = sorted(meth_genes & KNOWN)
    overlap = sorted(expr_genes & meth_genes)

    expr_sel.to_csv(RES / "07_expr_selected.csv", index=False)
    meth_sel.to_csv(RES / "07_meth_selected.csv", index=False)
    with open(RES / "07_biology_summary.txt", "w") as fh:
        def w(s): print(s); fh.write(s + "\n")
        w("=== BIOLOGY SANITY CHECK ===")
        w(f"expr: {len(expr_genes)} selected genes; top: {', '.join(list(expr_sel['feature'][:12]))}")
        w(f"meth: {len(meth_genes)} genes across selected probes; "
          f"top probe genes: {', '.join([g for g in meth_sel['genes'][:12] if g])}")
        w(f"\nknown-BRCA-loci hit by EXPRESSION selection: {expr_known or 'none'}")
        w(f"known-BRCA-loci hit by METHYLATION selection: {meth_known or 'none'}")
        w(f"\ngenes selected by BOTH layers (concordant): {overlap or 'none'}")
        w("interpretation: little/no gene overlap -> layers point at different biology "
          "(consistent with methylation carrying orthogonal signal, even if it did not "
          "translate into a clear AUROC gain in stage 04).")
    print(f"\nwrote {RES/'07_biology_summary.txt'}")

if __name__ == "__main__":
    main()
