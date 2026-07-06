# multiomics-ml-brca — does methylation add orthogonal signal for BRCA relapse?

**Central question.** Does late-fusion multi-omics integration (RNA-seq expression +
DNA methylation) improve prediction of relapse (PFI event) in TCGA-BRCA, versus the best
single-omics baseline — measured with **identical CV folds**, a **permutation noise
floor**, and **external validation** of the expression arm on METABRIC?

**The result is the measurement, not the direction.** This pipeline is built to honestly
detect whether methylation helps — and to trust a null if that is what the data say.

## Design (why each choice)
- **Relapse, not PAM50.** PAM50 is ~95% predictable from expression alone — no headroom
  for a second layer. Relapse is where expression is mediocre, so it is the honest test
  bed for orthogonal signal.
- **Methylation, not CNV.** CNV is largely redundant with expression; promoter
  methylation is a distinct regulatory mechanism that *can* carry information expression
  does not. Framed as a hypothesis the comparison tests — not a claim.
- **Fixed-horizon binary label (PFI, N=3yr).** TCGA follow-up is short (median ~2.1yr)
  and relapse is late, so time-to-event is underpowered. We binarise at N=3yr and name
  the censoring limitation out loud. N chosen from the KM at-risk table (stage 01), not
  arbitrarily. Patients censored before N are dropped (not clean negatives).
- **Late fusion is the headline.** Each layer gets its own model -> P(relapse); a
  logistic meta-learner learns weights on `[p_expr, p_meth]`. The **betas are the
  result**: beta_meth ~ 0 => methylation adds nothing beyond expression. Late fusion
  sidesteps the 450k-vs-20k dimensionality-dominance problem instead of fighting it.
  Early fusion is demoted to a baseline; DIABLO is cut.
- **Identical folds everywhere.** Every regime is scored on the one split from
  `common.shared_folds`. Feature selection + scaling + label filtering fit inside the
  training fold only. Late-fusion base probs are out-of-fold (no stacking leakage).
- **Metrics: AUROC + AUPRC (mean+/-std).** AUPRC baseline = event rate (~0.185), not 0.5.
  Accuracy is banned (an "everyone survives" model scores ~82% and catches zero relapses).
- **Permutation test (crown jewel).** Shuffle labels, rerun the ENTIRE supervised
  pipeline (incl. feature selection) 1000x -> null AUROC distribution ~0.5. Empirical p
  via Phipson-Smyth (never zero). Rules out high-dimensional luck.
- **METABRIC (expression arm only).** Out-of-distribution check of the expression
  baseline. The fusion claim stays single-cohort (TCGA) — stated as the primary limitation.

## Pipeline
| stage | script | output |
|---|---|---|
| 00 | `00_download_data.sh` | raw TCGA-BRCA (Xena) + METABRIC (cBioPortal) inputs |
| 01 | `01_label_horizon.py` | KM at-risk table -> horizon N decision |
| 02 | `02_qc_intersect_label.py` | sample QC, layer intersection, N=3yr label, `manifest.csv` |
| 03 | `03_extract_matrices.py` | aligned float32 `expr.npz` / `meth.npz` (label-blind preprocessing) |
| 04 | `04_run_all_regimes.py` | all regimes on shared folds; metrics + late-fusion betas |
| 05 | `05_permutation_test.py` | full-pipeline label-shuffle null + empirical p |
| 06 | `06_metabric_validation.py` | expression baseline on METABRIC |
| 07 | `07_biology_check.py` | selected loci; known-BRCA-methylation sanity check |
| 08 | `08_figures.py` | KM, regime metrics, permutation figures |

## Data (UCSC Xena TCGA-BRCA + cBioPortal METABRIC)
- Expression: `HiSeqV2` (IlluminaHiSeq RNA-seq, log2), 20,531 genes.
- Methylation: `HumanMethylation450` beta values, 485,577 probes.
- Outcome/subtype: Pan-Cancer survival table (Liu et al. 2018, PFI) + BRCA clinical (PAM50).
- METABRIC: `data_mrna_illumina_microarray` + `data_clinical_patient` (RFS).

## Run
```bash
conda env create -f env/environment.yml   # first time
conda activate brca-surv
bash scripts/00_download_data.sh           # ~1.5 GB raw inputs
python scripts/01_label_horizon.py
python scripts/02_qc_intersect_label.py
python scripts/03_extract_matrices.py
python scripts/04_run_all_regimes.py
python scripts/05_permutation_test.py     # ~50 min, 4 cores
python scripts/06_metabric_validation.py
python scripts/07_biology_check.py
python scripts/08_figures.py
```
See `RESULTS.md` for findings and discussion.
