#!/usr/bin/env python
"""08_figures.py - all publication figures. Run after stages 01-07.
  fig_km_horizon.png     : KM curve + at-risk table justifying N=3yr
  fig_regime_metrics.png : AUROC & AUPRC (mean+/-std) across regimes, identical folds
  fig_permutation.png    : late-fusion real AUROC vs permutation null (noise floor)
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from common import RAW, RES, PROC

def fig_km():
    sv = pd.read_csv(RAW / "TCGA-BRCA.survival.tsv", sep="\t", low_memory=False)
    sv = sv[sv["cancer type abbreviation"] == "BRCA"]
    sv = sv[sv["sample"].str.slice(13,15) == "01"].dropna(subset=["PFI","PFI.time"])
    t = sv["PFI.time"].astype(float)/365.25; e = sv["PFI"].astype(int)
    kmf = KaplanMeierFitter().fit(t, e)
    fig, ax = plt.subplots(figsize=(7,4.5))
    kmf.plot_survival_function(ax=ax, color="#2b6cb0")
    ax.axvline(3, ls="--", color="crimson", label="chosen horizon N=3yr")
    for N in [2,3,4,5]:
        ax.text(N, 0.55, f"n@{N}yr={(t>=N).sum()}", rotation=90, fontsize=8, color="gray")
    ax.set(xlabel="years", ylabel="PFI survival", xlim=(0,10), ylim=(0.5,1.0),
           title="TCGA-BRCA PFI Kaplan-Meier — horizon justification")
    ax.legend(); fig.tight_layout(); fig.savefig(RES/"fig_km_horizon.png", dpi=140); plt.close()

def fig_regimes():
    tab = pd.read_csv(RES/"04_regime_metrics.csv").sort_values("auroc_mean")
    base = tab["auprc_baseline"].iloc[0]
    fig, axes = plt.subplots(1,2, figsize=(11,4.5))
    for ax, (m,s,ttl) in zip(axes, [("auroc_mean","auroc_std","AUROC"),
                                    ("auprc_mean","auprc_std","AUPRC")]):
        c = ["#c0392b" if "late" in r else "#2b6cb0" for r in tab["regime"]]
        ax.barh(tab["regime"], tab[m], xerr=tab[s], color=c, alpha=.85)
        ref = 0.5 if ttl=="AUROC" else base
        ax.axvline(ref, ls="--", color="gray", label=f"baseline {ref:.2f}")
        ax.set(title=f"{ttl} (mean+/-std, identical folds)", xlim=(0, max(0.7, tab[m].max()+0.15)))
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(RES/"fig_regime_metrics.png", dpi=140); plt.close()

def fig_perm():
    f = RES/"05_permutation_null.npz"
    if not f.exists():
        print("  (permutation not done yet — skipping fig_permutation)"); return
    d = np.load(f)
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.hist(d["null_late"], bins=40, color="#95a5a6", alpha=.8, label="permutation null (shuffled labels)")
    ax.axvline(d["real_late"], color="crimson", lw=2, label=f"real late-fusion AUROC={float(d['real_late']):.3f}")
    ax.axvline(0.5, ls=":", color="k")
    ax.set(xlabel="pooled OOF AUROC", ylabel="permutations",
           title="Permutation noise floor — late fusion")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(RES/"fig_permutation.png", dpi=140); plt.close()

if __name__ == "__main__":
    fig_km(); print("wrote fig_km_horizon.png")
    fig_regimes(); print("wrote fig_regime_metrics.png")
    fig_perm()
