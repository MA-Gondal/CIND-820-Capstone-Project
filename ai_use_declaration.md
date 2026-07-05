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
*(To be completed)*
