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
- **Coverage:** 2006–2015, 70+ countries
- **Note:** Raw data is not committed to this repository due to file size.
  Download from the link above and place in `/data/raw/`

---

## Repository Structure
```
CIND820-Capstone-Project/
├── data/
│   ├── raw/          ← original downloaded CSV (not committed)
│   └── processed/    ← cleaned and feature-engineered data
├── notebooks/        ← Jupyter notebooks by milestone
├── outputs/
│   ├── figures/      ← saved plots and charts
│   └── reports/      ← HTML profiling report
├── docs/             ← pipeline diagram and architecture notes
├── ai_use_declaration.md
└── README.md

```

## How to Run
1. Download the dataset from the Kaggle link above and place CSV in `/data/raw/`
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scipy ydata-profiling xgboost scikit-learn shap`
3. Open and run `/notebooks/M2_EDA.ipynb` from top to bottom

---

## Milestones
| Milestone | Status | Description |
|---|---|---|
| M1 — Design & Strategy | ✅ Complete | Problem framing, dataset, RQs, ethics |
| M2 — Architecture & Data Audit | 🔄 In Progress | EDA, pipeline, literature review |
| M3 — Initial Results | ⏳ Upcoming | Baseline model, metrics, validation |
| M4 — Final Results | ⏳ Upcoming | Full report, model comparison |
| M5 — Presentation | ⏳ Upcoming | Live demo and Q&A |

---

## AI Use Declaration
See `ai_use_declaration.md` for full GenAI usage log per TMU Policy 60.
