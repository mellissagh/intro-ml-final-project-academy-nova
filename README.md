cat > README.md <<'EOF'
# Intro to Machine Learning Final Project — Academy Nova

Final project for the Introduction to Machine Learning course.

## Goal

Predict the probability that a course registration will be cancelled/dropped using historical Academy Nova CRM data.

The project follows the CRISP-DM process:

1. Business understanding
2. Data understanding and EDA
3. Data cleaning and missing value handling
4. Feature engineering and outlier analysis
5. Model training and hyperparameter tuning
6. Model evaluation and explainability
7. Final submission generation

## Evaluation Metric

The main evaluation metric is ROC-AUC.

The final submission file must contain exactly two columns:

- `Client_ID`
- `Drop_Probability`

## Repository Structure

```text
data/
  raw/              Raw train/test data
  processed/        Processed intermediate files
  submissions/      Submission CSV files

notebooks/          Jupyter notebooks
src/                Reusable Python code
reports/            Final report and figures
experiments/         Experiment logs and model results
final_submission/   Final files for Moodle submission