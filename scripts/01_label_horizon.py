#!/usr/bin/env python
"""
01_label_horizon.py — PFI label + horizon (N) decision evidence for TCGA-BRCA.

Produces the Kaplan-Meier at-risk table and, for each candidate horizon N,
the breakdown that decides whether N is defensible:

  positives      : PFI event observed at time <= N            -> label 1
  negatives      : event-free AND followed beyond N (t > N)   -> label 0  (clean)
  ambiguous      : censored before N with no event (t <= N, PFI==0) -> DROP
                   (these are NOT clean negatives — we don't know their fate)

The horizon is defensible only where the KM at-risk count past N is still large
enough that the negatives are genuinely observed, not assumed. Short TCGA-BRCA
follow-up (median ~2.3 yr) is exactly why this table, not a guess, picks N.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = Path("/home/ubuntu/survival-tcga")
RAW = ROOT / "data" / "raw"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

DAYS_PER_YR = 365.25
CANDIDATES = [2, 3, 4, 5]  # years

def main():
    df = pd.read_csv(RAW / "TCGA-BRCA.survival.tsv", sep="\t", low_memory=False)
    # Pan-Cancer table spans all cohorts; keep BRCA.
    df = df[df["cancer type abbreviation"] == "BRCA"].copy()
    # Primary tumor only: TCGA barcode sample-type suffix "-01".
    df = df[df["sample"].str.slice(13, 15) == "01"].copy()
    df = df.dropna(subset=["PFI", "PFI.time"])
    df["PFI"] = df["PFI"].astype(int)
    df["PFI.time"] = df["PFI.time"].astype(float)
    df["t_yr"] = df["PFI.time"] / DAYS_PER_YR

    n = len(df)
    med_fu = df.loc[df["PFI"] == 0, "t_yr"].median()  # reverse-KM proxy: median censoring time
    print(f"Primary-tumor BRCA patients with PFI: N={n}")
    print(f"Total PFI events: {df['PFI'].sum()}  (rate {df['PFI'].mean():.3f})")
    print(f"Median follow-up (censored median): {med_fu:.2f} yr\n")

    # KM at-risk table at each candidate horizon.
    kmf = KaplanMeierFitter().fit(df["t_yr"], df["PFI"])
    rows = []
    for N in CANDIDATES:
        pos = int(((df["PFI"] == 1) & (df["t_yr"] <= N)).sum())
        neg = int((df["t_yr"] > N).sum())                       # observed past N (event or not, t>N)
        neg_clean = int(((df["PFI"] == 0) & (df["t_yr"] > N)).sum())
        pos_late = int(((df["PFI"] == 1) & (df["t_yr"] > N)).sum())  # event after N -> negative at horizon
        amb = int(((df["PFI"] == 0) & (df["t_yr"] <= N)).sum())  # censored before N -> DROP
        at_risk = int((df["t_yr"] >= N).sum())
        usable = pos + neg
        rows.append(dict(
            horizon_yr=N, at_risk_at_N=at_risk,
            positives=pos, negatives=neg,
            neg_eventfree=neg_clean, neg_late_event=pos_late,
            dropped_ambiguous=amb, usable_N=usable,
            event_rate=round(pos / usable, 3) if usable else np.nan,
            km_surv_at_N=round(float(kmf.predict(N)), 3),
        ))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "01_horizon_table.csv", index=False)
    print(tab.to_string(index=False))
    print(f"\nwrote {RES/'01_horizon_table.csv'}")

if __name__ == "__main__":
    main()
