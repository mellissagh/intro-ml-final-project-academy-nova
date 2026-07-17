# Academy Nova Course Cancellation Prediction  
## Final Project Report — Group 51

**Course:** Introduction to Machine Learning  
**Project:** Nova Academy Course Cancellation Prediction  
**Group:** 51  
**Metric:** ROC-AUC  
**Best known public leaderboard AUC:** 0.889030  

---

## 1. Executive Summary

Nova Academy wants to predict the probability that a client will cancel or drop a course registration before the course starts. This is a binary classification problem where each registration receives a cancellation probability.

We followed a full machine-learning workflow based on CRISP-DM: business understanding, data exploration, preprocessing, feature engineering, model training, evaluation, explainability, and final submission generation.

The final submitted prediction file contains two columns: `Client_ID` and `Drop_Probability`. The final file was validated to ensure correct row count, no missing predictions, no duplicate IDs, valid probability values, and exact preservation of the test-set `Client_ID` order.

Our best known public leaderboard result for Group 51 was **AUC = 0.889030**, placing the group in **2nd place** on the leaderboard.

---

## 2. Business Understanding

The business goal is to help Nova Academy identify registrations with high cancellation risk. Accurate cancellation probabilities can support operational decisions such as proactive customer support, better resource planning, course capacity management, and reducing financial loss from cancellations.

The evaluation metric is ROC-AUC. This metric is appropriate because the goal is not only to classify each client using a fixed threshold, but to rank clients by cancellation risk. The final submission therefore contains probabilities rather than hard labels.

---

## 3. Data Understanding and EDA

The training set contains 63,464 rows and 29 columns, including the target variable `Dropped_Course`. The test set contains 15,866 rows and 28 columns, with the same features but without the target.

The target distribution showed moderate class imbalance: approximately 41.4% of the training registrations were cancelled and 58.6% were not cancelled. Because of this, accuracy alone would be less informative than ROC-AUC.

Exploratory data analysis included numeric distributions, categorical distributions, target relationships, missing-value patterns, and date-based trends.

A key finding was that the train and test sets are temporally shifted. The training dates range from 2015-07-01 to 2017-04-26, while the test set starts around 2017-04-26 and continues until 2017-08-31. This suggests that random cross-validation may be optimistic, so a time-aware validation split was also evaluated.

---

## 4. Missing Values and Preprocessing

Several columns contained missing values. `Company_ID` had very high missingness, while `Agent_ID`, `Registration_Days_Before`, `Requested_Lab_Config`, `Physical_Course_Kits`, and several categorical columns also had missing values.

Instead of simply removing rows or columns, we treated missingness as a potential predictive signal. Missing-indicator features were created for important columns, such as company, agent, registration timing, requested lab configuration, daily tuition cost, and participant count.

Numerical missing values were handled with median imputation, while categorical missing values were handled with a constant `"Missing"` category. Preprocessing was implemented inside a pipeline to avoid data leakage. The preprocessing pipeline was fit only on the training data and then applied to validation or test data.

Categorical variables were encoded using one-hot encoding with `handle_unknown="ignore"` to safely handle categories appearing in the test set but not in training.

---

## 5. Feature Engineering and Outlier Analysis

Feature engineering was based on the business logic of course registrations. We created participant-based, cost-based, previous-history, support-ticket, registration-change, missingness, lab-configuration, and date-based features.

Important engineered features included:

- `Total_Participants`
- `Active_Participants`
- `Observer_Ratio`
- `Student_Ratio`
- `Professional_Ratio`
- `Total_Hours`
- `Practical_Ratio`
- `Cost_Per_Hour`
- `Cost_Per_Participant`
- `Prev_Total_Courses`
- `Prev_Dropout_Rate`
- `Support_Per_Participant`
- `Changes_Per_Waiting_Day`
- `Lab_Config_Mismatch`
- `Has_Company`
- `Has_Agent`
- date features such as month, quarter, day of week, and week of year

Outliers were analyzed using distribution plots, boxplots, and IQR-based summaries. We avoided blindly deleting outliers because many extreme values may represent real business cases. Since the main models were tree-based, they could handle nonlinear effects and extreme values more robustly than linear models. Test rows were never removed.

---

## 6. Modeling and Validation Strategy

We compared several model families:

- Logistic Regression
- Random Forest
- HistGradientBoosting
- LightGBM

The baseline models showed that gradient-boosting methods performed best. Logistic Regression gave a useful simple baseline, but boosting models captured nonlinear interactions more effectively.

The validation strategy included:

1. Stratified K-Fold cross-validation  
2. Train AUC and validation AUC comparison  
3. Overfitting gap analysis  
4. Per-fold ROC curves  
5. Time-aware validation because of the temporal train/test shift  

This allowed us to compare models not only by average validation AUC, but also by stability and overfitting behavior.

---

## 7. Model Results and Hyperparameter Tuning

The main model comparison showed that LightGBM achieved the strongest validation performance. The tuned deeper LightGBM model achieved the highest random cross-validation AUC, while the baseline LightGBM had a slightly smaller train-validation gap.

Approximate validation results:

| Model | Validation AUC | Notes |
|---|---:|---|
| Logistic Regression | 0.9037 | Simple linear baseline |
| Random Forest | 0.8865 | Low overfitting but weaker performance |
| HistGradientBoosting | 0.9477 | Strong boosting baseline |
| LightGBM baseline | 0.9497 | Strong and stable |
| LightGBM tuned/deeper | 0.9523 | Highest random CV AUC |

Although random cross-validation AUC was high, time-aware validation was lower, around 0.902. This supported the conclusion that the data has temporal distribution shift and that leaderboard performance may be lower than random CV performance.

The final known public leaderboard AUC was **0.889030**.

---

## 8. Model Evaluation and Explainability

Model evaluation focused on ROC-AUC because the final output is a probability ranking. Per-fold ROC curves were used to check that performance was consistent across validation folds.

We also used explainability methods such as feature importance and SHAP-style analysis to understand which features influenced cancellation probability. The model relied on business-relevant patterns, including registration timing, previous client behavior, support interactions, course structure, missing company or agent information, and lab-configuration mismatch.

High-risk predictions were generally associated with patterns such as higher previous dropout behavior, missing company information, unusual registration behavior, and certain timing/configuration patterns. Lower-risk predictions were generally associated with more stable registration history and clearer organizational metadata.

---

## 9. Final Submission

The final submission file is:

`Group_51_Submission.csv`

It contains exactly two columns:

1. `Client_ID`
2. `Drop_Probability`

The file was validated using the following checks:

- Row count equals the test set row count: 15,866
- Columns are exactly `Client_ID` and `Drop_Probability`
- No missing values
- No duplicate `Client_ID` values
- All probabilities are between 0 and 1
- `Client_ID` order exactly matches the original test file

The final submitted file uses the best known leaderboard-safe prediction file, which achieved **AUC = 0.889030**.

---

## 10. Limitations and Future Improvements

The main limitation is temporal distribution shift between training and test data. Random cross-validation likely overestimates performance because it mixes dates, while the test set represents a future time period.

Future improvements could include:

- stronger time-aware validation
- model retraining with more recent data
- more careful treatment of high-cardinality IDs
- safer target encoding inside cross-validation
- probability calibration
- additional ensemble/blending experiments
- deeper error analysis by client segment and course type

Overall, the project produced a validated machine-learning pipeline, strong public leaderboard performance, and interpretable insights into cancellation risk.