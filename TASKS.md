# Intro to Machine Learning Final Project — Task List

Group 51  
Project: Academy Nova course cancellation prediction  
Main metric: ROC-AUC

---

## 0. Project Setup and Safety

- [x] Create private GitHub repository
- [x] Create folder structure
- [x] Add `.gitignore`
- [x] Keep raw data local and outside GitHub
- [x] Save best known leaderboard submission as safe backup
- [x] Add partner as collaborator
- [ ] Confirm student names and IDs
- [ ] Confirm final file names:
  - `Submission_51_Group.csv`
  - `Notebook_51_Group.ipynb`
  - `Report_51_Group.pdf`
  - `51_Group.zip`

---

## 1. Data Loading and Validation

- [x] Place `Train_Data.csv` inside `data/raw/`
- [x] Place `Test_Data_No_Target.csv` inside `data/raw/`
- [ ] Implement `src/config.py`
- [ ] Implement `src/data_loading.py`
- [ ] Implement `src/check_data.py`
- [ ] Run `python -m src.check_data`
- [ ] Verify train shape is `(63464, 29)`
- [ ] Verify test shape is `(15866, 28)`
- [ ] Verify target exists only in train
- [ ] Verify no duplicate `Client_ID`
- [ ] Verify train/test columns match

---

## 2. EDA and Missing Values — 25%

- [x] Dataset overview table
- [x] Target distribution plot
- [x] Numeric feature summary
- [] Categorical feature summary
- [ ] Date feature analysis
- [ ] Train/test distribution comparison
- [x] Missing-value count and percentage table
- [x] Missing-value visualization
- [x] Missingness by target
- [ ] Missing-value imputation decisions
- [x] Explain every important EDA plot in markdown
- [ ] Add most important EDA plots to report

---

## 3. Feature Engineering and Outlier Analysis — 20%

- [ ] Create participant-count features
- [ ] Create participant-ratio features
- [ ] Create course-hour features
- [ ] Create cost-per-participant/hour features
- [ ] Create previous-history features
- [ ] Create support and registration-change features
- [ ] Create lab-configuration mismatch feature
- [ ] Create missing-indicator features
- [ ] Create date-derived features
- [ ] Analyze outliers with plots
- [ ] Decide whether to keep, cap, transform, or remove outliers
- [ ] Explain every engineered feature and why it may help prediction
- [ ] Discuss dimensionality reduction / feature selection

---

## 4. Modeling and Hyperparameter Tuning — 20%

- [ ] Build simple baseline model
- [ ] Train Logistic Regression
- [ ] Train HistGradientBoosting or Random Forest
- [ ] Train LightGBM
- [ ] Train XGBoost
- [ ] Optional: Train CatBoost
- [ ] Use 5-fold Stratified CV
- [ ] Track train AUC and validation AUC
- [ ] Track overfitting gap
- [ ] Tune hyperparameters
- [ ] Compare models in a clear table
- [ ] Select final model using AUC and bias-variance tradeoff
- [ ] Test ensemble/blending only if OOF AUC improves

---

## 5. Evaluation and Explainability — 15%

- [ ] ROC curve for every fold
- [ ] Mean ROC curve
- [ ] Fold AUC table
- [ ] Confusion matrix
- [ ] Precision, recall, F1, accuracy
- [ ] Explain why AUC is the main metric
- [ ] SHAP summary bar plot
- [ ] SHAP beeswarm plot
- [ ] SHAP dependence plots
- [ ] Local SHAP explanations for high-risk and low-risk examples
- [ ] Error analysis for false positives and false negatives
- [ ] Analyze uncertain predictions

---

## 6. Report and Notebook

- [ ] Create `Notebook_51_Group.ipynb`
- [ ] Notebook has clean sections and markdown explanations
- [ ] Notebook runs from top to bottom
- [ ] Create `Report_51_Group.pdf`
- [ ] Report tells complete CRISP-DM story
- [ ] Report is max 10 pages before appendices
- [ ] Include executive summary
- [ ] Include model comparison
- [ ] Include per-fold ROC results
- [ ] Include SHAP results
- [ ] Include limitations and future improvements

---

## 7. Final Submission

- [ ] Generate final test predictions
- [ ] Ensure predictions are probabilities, not 0/1 labels
- [ ] Ensure final CSV has exactly:
  - `Client_ID`
  - `Drop_Probability`
- [ ] Ensure row count is 15866
- [ ] Ensure no missing probabilities
- [ ] Ensure no duplicate `Client_ID`
- [ ] Ensure probability range is between 0 and 1
- [ ] Preserve original test `Client_ID` order
- [ ] Save final CSV as `Submission_51_Group.csv`
- [ ] Copy final notebook to `final_submission/`
- [ ] Copy final report to `final_submission/`
- [ ] Create `51_Group.zip`
- [ ] Open ZIP and verify it contains exactly the required files