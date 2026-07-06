#!/usr/bin/env python
"""05_permutation_test.py - full-pipeline label-shuffle noise floor (CROWN JEWEL).

Proves the headline AUROC reflects real feature-outcome structure, not high-dimensional
luck. The shuffle wraps the ENTIRE supervised procedure: SelectKBest(supervised) +
scaling + classifier + late-fusion stacking rerun on each permuted label vector. The
unsupervised variance prefilter (stage 03) is label-invariant -> identical under every
permutation, so it cannot inflate the null.

Statistic = pooled out-of-fold AUROC on the shared folds. Null centres ~0.5.
Empirical p via Phipson & Smyth (2010): p=(b+1)/(m+1), never zero.
Parallelised across permutations (independent) on 4 CPUs.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, time
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from joblib import Parallel, delayed
from common import load_aligned, shared_folds, make_pipe, RES, SEED

N_PERM = 1000
K = 500
N_JOBS = 4

def oof(X, y, folds, k=K):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = clone(make_pipe("lasso", k)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p

def stats_for(Xe, Xm, y, folds):
    pe = oof(Xe, y, folds); pm = oof(Xm, y, folds)
    Z = np.column_stack([pe, pm]); pl = np.zeros(len(y))
    for tr, te in folds:
        pl[te] = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    return roc_auc_score(y, pe), roc_auc_score(y, pl)

def emp_p(null, real):
    return (int(np.sum(np.asarray(null) >= real)) + 1) / (len(null) + 1)

def main():
    Xe, Xm, y, _ = load_aligned()
    folds = shared_folds(y)
    rng = np.random.default_rng(SEED)
    real_expr, real_late = stats_for(Xe, Xm, y, folds)
    print(f"REAL  expr_lasso AUROC={real_expr:.4f}   late_lasso AUROC={real_late:.4f}", flush=True)

    perms = [rng.permutation(y) for _ in range(N_PERM)]
    print(f"running {N_PERM} permutations on {N_JOBS} cores ...", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(stats_for)(Xe, Xm, yp, folds) for yp in perms)
    null_expr = [r[0] for r in res]; null_late = [r[1] for r in res]
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)

    out = dict(real_expr=real_expr, real_late=real_late,
               null_expr_mean=float(np.mean(null_expr)), null_late_mean=float(np.mean(null_late)),
               null_expr_p95=float(np.percentile(null_expr, 95)),
               null_late_p95=float(np.percentile(null_late, 95)),
               p_expr=emp_p(null_expr, real_expr), p_late=emp_p(null_late, real_late), n_perm=N_PERM)
    np.savez(RES / "05_permutation_null.npz", null_expr=np.array(null_expr),
             null_late=np.array(null_late), real_expr=real_expr, real_late=real_late)
    pd.Series(out).to_csv(RES / "05_permutation_summary.csv")
    print("\n=== PERMUTATION RESULT ===")
    for k, v in out.items():
        print(f"  {k:16s} {v}")

if __name__ == "__main__":
    main()
