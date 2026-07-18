# AI Use Declaration — CIND820 Capstone Project

**Student:** Mubarak Ahmed (501345730)  
**Course:** CIND820 — Big Data Analytics Project, Toronto Metropolitan University  
**Policy:** TMU Academic Integrity Policy 60

This document logs all generative AI tool use across project milestones,
in accordance with TMU Policy 60 and the verification plan established in Milestone 1.

---

## Milestone 2 — Architecture & Data Audit

| Activity | Use Description | Verification Method | Status |
|---|---|---|---|
| Literature search assistance | Used to identify initial paper titles and author names for the supply chain delay prediction literature; all papers subsequently located and read via TMU library databases | Every cited source personally accessed and verified against original publication | Verified |
| Writing assistance (grammar and structure) | Used to review prose clarity in the literature review and architecture sections; all substantive analytical content represents the student's own reasoning | AI-polished phrasing rewritten in student's own voice before submission | Verified |
| Pipeline architecture design | Not used; pipeline design derived from literature review findings and dataset profiling constraints | N/A | N/A |
| Code generation | Not yet applied at Milestone 2; will apply in Milestone 3 with per-block verification | All code will be manually run, output inspected, and logic verified before submission | Planned |
| Research question formulation and interpretation | Not used | N/A | N/A |

---

## Milestone 3 — Initial Results & Coding

| Activity | Use Description | Verification Method | Status |
|---|---|---|---|
| Code generation (pipeline modules and notebooks) | Used Claude (Anthropic) to draft `src/preprocessing.py`, `src/features.py`, `src/models.py` and notebooks 02–05 implementing the pipeline design specified in the Milestone 2 report | Every notebook executed locally end-to-end on my machine; outputs inspected cell by cell; class counts verified against the Milestone 2 report (9,138 / 691 / 495); CV and test tables cross-checked between notebook display, saved CSVs, and markdown text; discrepancies corrected before commit | Verified |
| Debugging and environment support | Used to diagnose a missing `pyarrow` dependency (added to requirements.txt), a missing `imbalanced-learn` install, `.gitignore` setup, and a logistic regression convergence issue (resolved with within-fold StandardScaler) | Each fix applied and re-run locally; final runs complete without errors or warnings | Verified |
| Methodology discussion | Used to reason through design decisions: cross-reference recovery for weight/freight, vendor risk tier threshold choice, one-sided Wilcoxon justification with n=5 folds, SMOTE vs class-weight interpretation | Each decision reviewed against the Milestone 2 design and course materials; I can explain and defend every choice independently | Verified |
| Documentation drafting | Used to draft notebook markdown narration and README run instructions | All narration reviewed and edited into my own words; numbers updated to match my own local runs where they differed from the assistant's reference runs | Verified |
| Analytical interpretation of results | Interpretation of CV rankings, confusion matrix, per-class weaknesses, and generalization gap developed jointly through discussion; final written interpretation reviewed and owned by me | Verified against my own run outputs and saved artifact tables | Verified |

---

## Milestone 4 — Final Results & Report

| Activity | Use Description | Verification Method | Status |
|---|---|---|---|
| Code generation (final modelling and impact modules) | Used Claude (Anthropic) to draft `src/final_models.py`, `src/impact.py`, and Notebooks 06–07 implementing the Milestone 4 agenda set out in the Milestone 3 report §7 (hyperparameter search, repeated cross-validation, deterministic final fit, SHAP attribution, cost-sensitive decision rule, safety-stock simulation) | Every notebook executed locally end-to-end; outputs inspected cell by cell; all reported numbers cross-checked between notebook display output and the saved CSV artifacts in `outputs/tables/` before commit; the Notebook 06 determinism check (bit-identical refits) verified on my machine | Verified |
| Methodology discussion | Used to reason through design decisions: champion selection criterion when tuned and default configurations tie within fold noise, leakage-free selection of the cost-sensitive decision weights on out-of-fold training predictions, class-mixture variance formulation of the Milestone 1 safety-stock formula, and expressing buffers in days of cover given the absence of demand data | Each decision reviewed against the Milestone 1 design, Milestone 2/3 reports, and course materials; I can explain and defend every choice independently | Verified |
| Documentation and report drafting | Used to draft notebook markdown narration, README updates, and prose sections of the final report | All narration and report text reviewed and edited into my own words; every quantitative claim traced to a committed artifact from my own local runs | Verified |
| Literature comparison support | Used to help structure the comparison between this project's results and prior work identified in Milestone 2 | Every cited source personally accessed and verified against the original publication; no source cited that I have not read | Verified |

---

*(End of declaration)*
