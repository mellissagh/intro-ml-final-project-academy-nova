import pandas as pd

from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    ID_COL,
    TARGET_COL,
    EXPECTED_TRAIN_SHAPE,
    EXPECTED_TEST_SHAPE,
)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw training and test datasets.

    Returns
    -------
    train_df:
        Training dataset including the target column.
    test_df:
        Test dataset without the target column.
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Train file not found at {TRAIN_PATH}. "
            "Make sure Train_Data.csv is inside data/raw/."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test file not found at {TEST_PATH}. "
            "Make sure Test_Data_No_Target.csv is inside data/raw/."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def validate_raw_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Validate the expected structure of the raw files.

    This protects us from common project mistakes:
    - wrong file names
    - missing target
    - target accidentally appearing in test
    - duplicate Client_ID values
    - train/test column mismatch
    - row count mismatch
    """
    print("Validating raw data...")

    assert train_df.shape == EXPECTED_TRAIN_SHAPE, (
        f"Unexpected train shape: {train_df.shape}, "
        f"expected {EXPECTED_TRAIN_SHAPE}"
    )

    assert test_df.shape == EXPECTED_TEST_SHAPE, (
        f"Unexpected test shape: {test_df.shape}, "
        f"expected {EXPECTED_TEST_SHAPE}"
    )

    assert ID_COL in train_df.columns, f"{ID_COL} is missing from train data."
    assert ID_COL in test_df.columns, f"{ID_COL} is missing from test data."

    assert TARGET_COL in train_df.columns, f"{TARGET_COL} is missing from train data."
    assert TARGET_COL not in test_df.columns, (
        f"{TARGET_COL} should not appear in test data."
    )

    train_features = set(train_df.columns) - {TARGET_COL}
    test_features = set(test_df.columns)

    missing_in_test = train_features - test_features
    extra_in_test = test_features - train_features

    assert not missing_in_test, f"Columns missing in test data: {missing_in_test}"
    assert not extra_in_test, f"Unexpected columns in test data: {extra_in_test}"

    assert train_df[ID_COL].is_unique, f"{ID_COL} contains duplicates in train."
    assert test_df[ID_COL].is_unique, f"{ID_COL} contains duplicates in test."

    assert train_df[TARGET_COL].isin([0, 1]).all(), (
        f"{TARGET_COL} must contain only 0/1 values."
    )

    print("Raw data validation passed.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Target positive rate: {train_df[TARGET_COL].mean():.4f}")


def get_feature_target_split(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split training data into features X and target y.
    """
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]

    return X, y