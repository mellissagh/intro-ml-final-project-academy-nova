from pathlib import Path

# ============================================================
# Project metadata
# ============================================================

GROUP_NUMBER = 51

# TODO: Fill before final submission
STUDENT_1_NAME = "Mellissa Ghandour"
STUDENT_1_ID = "TODO"
STUDENT_2_NAME = "TODO"
STUDENT_2_ID = "TODO"

# ============================================================
# Reproducibility
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FINAL_SUBMISSION_DIR = PROJECT_ROOT / "final_submission"

# ============================================================
# Input files
# ============================================================

TRAIN_PATH = RAW_DATA_DIR / "Train_Data.csv"
TEST_PATH = RAW_DATA_DIR / "Test_Data_No_Target.csv"

# Best known public leaderboard submission backup
BEST_PUBLIC_SUBMISSION_PATH = (
    SUBMISSIONS_DIR / "Group_51_Submission_AUC_0.889030_SAFE_BACKUP.csv"
)

# ============================================================
# Output files
# ============================================================

FINAL_SUBMISSION_CSV = FINAL_SUBMISSION_DIR / f"Submission_{GROUP_NUMBER}_Group.csv"
FINAL_NOTEBOOK = FINAL_SUBMISSION_DIR / f"Notebook_{GROUP_NUMBER}_Group.ipynb"
FINAL_REPORT = FINAL_SUBMISSION_DIR / f"Report_{GROUP_NUMBER}_Group.pdf"
FINAL_ZIP = FINAL_SUBMISSION_DIR / f"{GROUP_NUMBER}_Group.zip"

# ============================================================
# Column names
# ============================================================

ID_COL = "Client_ID"

# In the uploaded train file the target is named Dropped_Course.
TARGET_COL = "Dropped_Course"

# The leaderboard/submission checker accepts Drop_Probability.
SUBMISSION_PROB_COL = "Drop_Probability"

EXPECTED_TRAIN_SHAPE = (63464, 29)
EXPECTED_TEST_SHAPE = (15866, 28)