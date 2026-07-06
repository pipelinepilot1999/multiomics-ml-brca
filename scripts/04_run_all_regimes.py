#!/usr/bin/env python
"""04_run_all_regimes.py - every regime on the ONE shared fold split.

Regimes: expr/meth {lasso,rf} single-omics; early_{lasso,rf} balanced concat;
late_lasso HEADLINE OOF-stacked meta-LR. Selection+scaling inside each fold
(leak-free). Late-fusion base probs are out-of-fold by construction. Metrics:
AUROC+AUPRC per fold -> mean+/-std; AUPRC baseline = event rate. Fixed regularised
config everywhere (~68 events -> tuning buys optimistic bias); same config reused
by the permutation null so real vs null is fair.
"""
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score
from common import load_aligned, shared_folds, make_pipe, RES, SEED, N_SPLITS
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

K_SINGLE = 500
K_EARLY = 250

def oof_probs(X, y, folds, kind, k):
    p = np.zeros(len(y))
    for tr, te in folds:
        mdl = clone(make_pipe(kind, k)).fit(X[tr], y[tr])
        p[te] = mdl.predict_proba(X[te])[:, 1]
    return p

def _early_clf(kind):
    if kind == "lasso":
        return LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.1,
                                  solver="saga", max_iter=5000, class_weight="balanced")
    return RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                  class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)

def early_oof(Xe, Xm, y, folds, kind, k=K_EARLY):
    p = np.zeros(len(y))
    for tr, te in folds:
        se = SelectKBest(f_classif, k=min(k, Xe.shape[1])).fit(Xe[tr], y[tr])
        sm = SelectKBest(f_classif, k=min(k, Xm.shape[1])).fit(Xm[tr], y[tr])
        sce = StandardScaler().fit(se.transform(Xe[tr]))
        scm = StandardScaler().fit(sm.transform(Xm[tr]))
        Ztr = np.hstack([sce.transform(se.transform(Xe[tr])), scm.transform(sm.transform(Xm[tr]))])
        Zte = np.hstack([sce.transform(se.transform(Xe[te])), scm.transform(sm.transform(Xm[te]))])
        clf = _early_clf(kind).fit(Ztr, y[tr])
        p[te] = clf.predict_proba(Zte)[:, 1]
    return p

def per_fold_metrics(y, p, folds):
    au = np.array([roc_auc_score(y[te], p[te]) for _, te in folds])
    ap = np.array([average_precision_score(y[te], p[te]) for _, te in folds])
    return au, ap

def main():
    Xe, Xm, y, samples = load_aligned()
    folds = shared_folds(y)
    print(f"N={len(y)}  events={y.sum()}  rate={y.mean():.3f}  folds={N_SPLITS}")
    oof = {}
    oof["expr_lasso"] = oof_probs(Xe, y, folds, "lasso", K_SINGLE)
    oof["expr_rf"]    = oof_probs(Xe, y, folds, "rf",    K_SINGLE)
    oof["meth_lasso"] = oof_probs(Xm, y, folds, "lasso", K_SINGLE)
    oof["meth_rf"]    = oof_probs(Xm, y, folds, "rf",    K_SINGLE)
    oof["early_lasso"] = early_oof(Xe, Xm, y, folds, "lasso")
    oof["early_rf"]    = early_oof(Xe, Xm, y, folds, "rf")
    Z = np.column_stack([oof["expr_lasso"], oof["meth_lasso"]])
    p_late = np.zeros(len(y))
    for tr, te in folds:
        meta = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])
        p_late[te] = meta.predict_proba(Z[te])[:, 1]
    oof["late_lasso"] = p_late
    meta_full = LogisticRegression(max_iter=1000).fit(Z, y)
    betas = dict(intercept=float(meta_full.intercept_[0]),
                 beta_expr=float(meta_full.coef_[0, 0]),
                 beta_meth=float(meta_full.coef_[0, 1]))
    rows = []
    for name, p in oof.items():
        au, ap = per_fold_metrics(y, p, folds)
        rows.append(dict(regime=name, auroc_mean=au.mean(), auroc_std=au.std(),
                         auprc_mean=ap.mean(), auprc_std=ap.std()))
    tab = pd.DataFrame(rows).sort_values("auroc_mean", ascending=False)
    tab["auprc_baseline"] = round(float(y.mean()), 3)
    np.savez(RES / "04_oof_probs.npz", **oof, y=y, samples=samples)
    tab.to_csv(RES / "04_regime_metrics.csv", index=False)
    pd.Series(betas).to_csv(RES / "04_late_fusion_betas.csv")
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(tab.to_string(index=False))
    print("\nLate-fusion meta-learner betas (THE RESULT):")
    for k, v in betas.items():
        print(f"  {k:10s} {v:+.4f}")
    print(f"\nwrote {RES/'04_regime_metrics.csv'}")

if __name__ == "__main__":
    main()
