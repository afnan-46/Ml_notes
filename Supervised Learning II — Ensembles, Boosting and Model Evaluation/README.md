# CSE475 — Experiment 3: Ensembles, Boosting and Model Evaluation

**Notebook:** `exp3_ensembles_evaluation.ipynb` (also referred to as Lab 4)
**Course:** CSE475 — Machine Learning
**Related theory:** Module 2 — Ensemble Learning and Performance Evaluation
**Scheduled:** Week 3 (2-hour laboratory session)

## 1. What this notebook does

This notebook studies how bagging, boosting and meta-ensembles improve on a
single learner, and evaluates every model with 5-fold stratified
cross-validation and a full set of metrics — not accuracy alone. It works
through nine tasks from the lab sheet plus three written analysis
questions, using the **IBM Telco Customer Churn** dataset (7,043 customers,
26.5% churn) as the default dataset for all parts.

The notebook is self-contained: it downloads its own data, runs top to
bottom with a fixed random seed, and produces every table and figure the
lab report needs.

## 2. Dataset

| | |
|---|---|
| Source | [IBM Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) (downloaded automatically, no manual upload needed) |
| Rows | 7,043 customers |
| Features | 19 (after dropping `customerID` and the target `Churn`) |
| Class mapping | `Churn == "Yes"` → **1** (positive class, the event to predict) · `"No"` → **0** |
| Class balance | 73.46% no-churn / 26.54% churn (moderate imbalance) |
| Known data quirk | `TotalCharges` is stored as text and has 11 blank values (zero-tenure customers); the loader coerces it to numeric, leaving those 11 as `NaN` for the pipeline's imputer to handle |

To use a different dataset (e.g. from Experiment 2), replace the loader in
section 0.2 so it returns `X` (a DataFrame of raw features) and `y` (an
integer 0/1 Series) — every cell downstream is dataset-agnostic as long as
that contract is met.

## 3. How to run it

1. Open the notebook in Jupyter, JupyterLab, VS Code, or Google Colab (a
   "Open in Colab" badge is included at the top of the notebook).
2. Run all cells top to bottom — **Kernel → Restart & Run All** before
   submitting, so the output matches a clean run.
3. No local files or manual downloads are required; the dataset is fetched
   from the URL above the first time section 0.2 runs. An internet
   connection is needed for that one request.
4. Everything is seeded with `SEED = 42`, so a clean run reproduces every
   number and figure in this notebook exactly.

### Requirements

```
python >= 3.9
numpy
pandas
matplotlib
scikit-learn >= 1.6
xgboost >= 3.0
```

Install with:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

(Developed and tested with numpy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1,
xgboost 3.4.1 — the first code cell prints the exact versions in use.)

## 4. Structure of the notebook

| Section | Contents |
|---|---|
| **0. Setup** | Imports & seed · load data · column-by-column EDA · stratified train/test split · preprocessing pipeline (`prep`) and CV helpers (`wrapped`, `cv_report`, `fmt`) |
| **Task 1** | Baseline: a single, fully grown decision tree |
| **Task 2** | Bagging vs Random Forest (100 trees each) |
| **Task 3** | AdaBoost and XGBoost tuned over a 3×3 grid (`n_estimators` × `learning_rate`); includes a fold-1 worked example computing every metric by hand |
| **Task 4** | Soft voting and stacking over logistic regression, Random Forest and XGBoost |
| **Task 5** | Full 8-model comparison table and error-bar figure, sorted by macro-F1 |
| **Task 6** | Out-of-fold confusion matrix and hand-computed precision / recall / specificity / F1 / accuracy for the best model |
| **Task 7** | ROC curve and precision–recall curve, with AUC and average precision |
| **Task 8** | Cost-based decision threshold (`cost = 10·FN + FP`), chosen on out-of-fold scores and frozen **before** the held-out test set is opened once, at the very end |
| **Task 9** | Random Forest feature importances, per encoded column and aggregated back to the original variables |
| **Analysis questions** | Q1 (bagging vs boosting: variance vs bias), Q2 (why accuracy is misleading under heavy imbalance), Q3 (effect of lowering the decision threshold on recall and precision), with a numeric check for Q3 |
| **Deliverables checklist** | What must be filled in and present before submission |

## 5. Rules the notebook follows throughout

1. **The test set is touched exactly once**, in Task 8, after the model and
   the decision threshold are both frozen using only the training set.
   Every comparison before that uses 5-fold stratified cross-validation on
   the training set.
2. **Every CV score is reported as mean ± standard deviation** across the 5
   folds, never as a single number — so the report can judge whether two
   models are really different or just within noise of each other.
3. **Preprocessing lives inside the pipeline** (`wrapped()`), so imputation,
   scaling and one-hot encoding are refit inside every fold and never see
   that fold's validation rows — avoiding data leakage.

## 6. Outputs produced

Running the notebook end to end produces:

- **Tables 0a–9 / Table Q3** — EDA summary, split distribution, per-model CV
  results, grid-search results, confusion-matrix metrics, test-set results,
  feature importances, and the Q3 threshold sweep.
- **Figures 0a–5** — class histograms, churn rate per category, split
  distribution, per-model comparison bars, confusion matrices, ROC/PR
  curves, cost-vs-threshold curve, and feature-importance bar chart.
- Each table/figure has a caption cell directly beneath it; a few captions
  end with a *"Write here"* prompt that must be filled in by hand before
  submission (see the checklist in section 9 of the notebook).

## 7. What to submit

Per the notebook's own deliverables checklist:

- [ ] Notebook runs top to bottom with `SEED = 42` (Restart & Run All).
- [ ] Tables 1–9 and Figures 1–5 present, each with its caption filled in.
- [ ] A written report (Section 3 format) answering Q1–Q3.
- [ ] The class mapping (which label is 1) stated explicitly in the report.
- [ ] The test set appears only in Task 8.

**Marks:** Implementation 4 · Result analysis 3 · Report and discussion 3 ·
**Total 10**.

## 8. Known gaps to check before submitting

- `config_table()` (Task 3) is defined but never called in the supplied
  notebook — add `display(config_table(gs_ada))` / `display(config_table(gs_xgb))`
  if your lab sheet expects the full nine-configuration grid table.
- The optional nested-CV cell at the end of Task 3 is set to
  `RUN_NESTED = False` by default; it gives an unbiased post-tuning score
  but is slow (225 fits per model) — switch it on if your report needs the
  unbiased figure rather than the slightly optimistic `best_score_`.
