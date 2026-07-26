# CIND820 Capstone — Predictive Shipment Delay Classification

**Student:** Mubarak Ahmed (501345730)
**Course:** CIND820 — Big Data Analytics Project, Toronto Metropolitan University
**Supervisor:** Dr. Ceni Babaoglu

---

## Project Summary
A machine learning system to classify incoming shipments by delay risk category
(on-time / at-risk / delayed) using the USAID SCMS Delivery History Dataset,
and to operationalize the model output into a dynamic safety-stock adjustment formula
for procurement planners in global health supply chains.

---

## Research Questions
- **RQ1:** Which pre-shipment attributes are the strongest predictors of delay risk?
- **RQ2:** Does XGBoost achieve superior macro-averaged F1 over a logistic regression baseline?
- **RQ3:** Can predicted delay probability be embedded into a dynamic safety-stock formula?

---

## Dataset
- **Name:** USAID SCMS Delivery History Dataset
- **Source:** https://www.kaggle.com/datasets/princehobby/supply-chain-shipment-dataset
- **Records:** 10,324 rows, 33 columns
- **Coverage:** 2006–2015, 43 destination countries
---

## Repository Structure
```
CIND820-Capstone-Project/
├── data/
│   ├── raw/              ← USAID SCMS CSV + README_data.md
│   └── processed/        ← cleaned splits + feature matrices (parquet, split indices)
├── docs/                 ← pipeline diagram
├── notebooks/            ← M2_EDA, 02_preprocessing, 03_feature_engineering,
│                           04_modelling, 05_evaluation, 06_final_modelling,
│                           07_final_evaluation_impact
├── outputs/
│   ├── figures/          ← saved plots and charts
│   ├── models/           ← fitted pipeline + champion/baseline models (joblib)
│   ├── reports/          ← compiled HTML versions of all executed notebooks
│   └── tables/           ← CV results, test metrics, selection and exclusion logs
├── src/                  ← preprocessing.py, features.py, models.py, final_models.py, impact.py
├── ai_use_declaration.md
├── requirements.txt
└── README.md
```

## How to Run

**Environment:** Python 3.11+ (developed on 3.13), VS Code with the Jupyter extension.

1. Download the dataset from the Kaggle link above and place the CSV in `/data/raw/`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks **in order** (each stage writes artifacts the next stage reads):

| Order | Notebook | What it does | Writes | Approx. runtime |
|---|---|---|---|---|
| 1 | `notebooks/M2_EDA.ipynb` | Exploratory data analysis (Milestone 2) | figures, profiling report | ~2 min |
| 2 | `notebooks/02_preprocessing.ipynb` | Target engineering, cross-reference recovery, leakage removal, stratified 80/20 split | `data/processed/train.parquet`, `test.parquet`, `split_indices.json` | <1 min |
| 3 | `notebooks/03_feature_engineering.ipynb` | Imputation + flags, log1p, chi-squared selection, one-hot encoding, vendor risk tier | `X_train.parquet`, `X_test.parquet`, `feature_engineer.joblib` | ~1 min |
| 4 | `notebooks/04_modelling.ipynb` | 3 models × 2 imbalance strategies × 5-fold stratified CV; fits final champion + baseline | CV tables, boxplot, `xgboost_champion.joblib`, `logreg_baseline.joblib` | 1–3 min |
| 5 | `notebooks/05_evaluation.ipynb` | Held-out test evaluation, confusion matrix, per-class report, Wilcoxon test | test metric tables, confusion matrix figure | <1 min |
| 6 | `notebooks/06_final_modelling.ipynb` | **M4:** hyperparameter search (20 configs × 5-fold CV), repeated 3×5 stratified CV, criterion-based champion selection, deterministic final fit | tuning/repeated-CV tables, `xgboost_champion_final.joblib`, `logreg_baseline_final.joblib`, `final_champion_meta.json` | ~15 min |
| 7 | `notebooks/07_final_evaluation_impact.ipynb` | **M4:** final held-out test evaluation, SHAP attribute-level attribution, cost-sensitive decision rule (selected on out-of-fold train predictions), dynamic safety-stock simulation | final test tables, SHAP/decision/safety-stock tables and figures | ~5 min |

Compiled HTML versions of all executed notebooks are in `outputs/reports/`.
Reusable pipeline logic lives in `src/` (`preprocessing.py`, `features.py`, `models.py`) —
notebooks import from these modules rather than duplicating code.

**Reproducibility notes:** the train-test split and CV folds are pinned with `random_state=42`
and the exact split row IDs are saved in `data/processed/split_indices.json`. Model rankings are
stable across machines; third-decimal scores may vary slightly with library versions.
**Milestone 4 determinism:** the final champion and baseline are fit with `n_jobs=1`, a fixed
seed, and a fixed tree count (no early stopping). Notebook 06 verifies that two independent
refits produce bit-identical predictions, so every Milestone 4 test number is exactly
reproducible from the committed code and data.

---

## Milestones
| Milestone | Status | Description |
|---|---|---|
| M1 — Design & Strategy | ✅ Complete | Problem framing, dataset, RQs, ethics |
| M2 — Architecture & Data Audit | ✅ Complete | EDA, pipeline, literature review |
| M3 — Initial Results | ✅ Complete | Preprocessing→evaluation pipeline, champion vs baseline, initial metrics |
| M4 — Final Results | ✅ Complete | Hyperparameter search, repeated CV, champion selection, SHAP (RQ1), cost-sensitive decision rule, safety-stock simulation (RQ3), final report |
| M5 — Presentation | ⏳ Upcoming | Live demo and Q&A |

---

## AI Use Declaration
See `ai_use_declaration.md` for full GenAI usage log per TMU Policy 60.
