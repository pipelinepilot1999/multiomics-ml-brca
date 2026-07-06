#!/usr/bin/env python
"""
common.py — shared data loading, the ONE canonical CV split, and leak-free pipelines.

The load-bearing rule of this project: every regime (expr-only, meth-only, early
fusion, late fusion, permutation null) is scored on the *identical* folds returned
by shared_folds(). Nothing else may create its own split.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path("/home/ubuntu/survival-tcga")
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED = 20260706
N_SPLITS = 5

def load_layer(name):
    d = np.load(PROC / f"{name}.npz", allow_pickle=True)
    return d["X"], d["samples"], d["features"]

def load_labels():
    man = pd.read_csv(PROC / "manifest.csv")
    return man  # sample, patient, label, pam50, ...

def load_aligned():
    """Return Xe, Xm, y, samples with a single consistent sample order."""
    Xe, se, _ = load_layer("expr")
    Xm, sm, _ = load_layer("meth")
    man = load_labels()
    assert list(se) == list(sm) == man["sample"].tolist(), "layer/manifest order mismatch"
    return Xe, Xm, man["label"].to_numpy(int), man["sample"].to_numpy()

def shared_folds(y):
    """The single canonical split. Deterministic; identical for every regime."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))

# ---- canonical leak-free pipelines (selection + scaling INSIDE the fold) ----
def lasso_pipe(k=500):
    # elastic-net logistic regression: L1/L2 mix suits high-dim low-EPV, keeps selection.
    return Pipeline([
        ("select", SelectKBest(f_classif, k=k)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1,
                                   solver="saga", max_iter=5000, class_weight="balanced")),
    ])

def rf_pipe(k=500):
    return Pipeline([
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                       class_weight="balanced_subsample",
                                       random_state=SEED, n_jobs=-1)),
    ])

def make_pipe(kind, k=500):
    return lasso_pipe(k) if kind == "lasso" else rf_pipe(k)
