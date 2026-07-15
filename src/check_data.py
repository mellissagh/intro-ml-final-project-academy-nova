from src.config import ID_COL, TARGET_COL
from src.data_loading import load_raw_data, validate_raw_data


def main() -> None:
    train_df, test_df = load_raw_data()
    validate_raw_data(train_df, test_df)

    print("\n==============================")
    print("Train columns")
    print("==============================")
    print(train_df.columns.tolist())

    print("\n==============================")
    print("Test columns")
    print("==============================")
    print(test_df.columns.tolist())

    print("\n==============================")
    print("Data types")
    print("==============================")
    print(train_df.dtypes)

    print("\n==============================")
    print("Target distribution")
    print("==============================")
    print(train_df[TARGET_COL].value_counts())
    print(train_df[TARGET_COL].value_counts(normalize=True))

    print("\n==============================")
    print("Missing values in train")
    print("==============================")
    print(train_df.isna().sum().sort_values(ascending=False).head(20))

    print("\n==============================")
    print("Missing values in test")
    print("==============================")
    print(test_df.isna().sum().sort_values(ascending=False).head(20))

    print("\n==============================")
    print("Duplicate Client_ID check")
    print("==============================")
    print(f"Train duplicated {ID_COL}: {train_df[ID_COL].duplicated().sum()}")
    print(f"Test duplicated {ID_COL}: {test_df[ID_COL].duplicated().sum()}")

    print("\nData check completed successfully.")


if __name__ == "__main__":
    main()