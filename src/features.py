import numpy as np
import pandas as pd


def safe_divide(numerator, denominator):
    """
    Safely divide two values or pandas Series.
    If denominator is 0 or missing, return 0.
    """
    return np.where(
        (denominator == 0) | pd.isna(denominator),
        0,
        numerator / denominator
    )


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business-driven engineered features.

    This function does not use the target column, so it can be safely applied
    to both train and test data.

    Feature groups:
    - participant composition
    - course workload and structure
    - cost-related features
    - previous history
    - support and registration behavior
    - missingness indicators
    - lab configuration mismatch
    - date-derived features
    """
    df = df.copy()

    # ------------------------------------------------------------
    # Participant composition
    # ------------------------------------------------------------
    df["Total_Participants"] = (
        df["Professionals_Count"].fillna(0)
        + df["Students_Count"].fillna(0)
        + df["Observers_Count"].fillna(0)
    )

    df["Active_Participants"] = (
        df["Professionals_Count"].fillna(0)
        + df["Students_Count"].fillna(0)
    )

    df["Observer_Ratio"] = safe_divide(
        df["Observers_Count"].fillna(0),
        df["Total_Participants"]
    )

    df["Student_Ratio"] = safe_divide(
        df["Students_Count"].fillna(0),
        df["Total_Participants"]
    )

    df["Professional_Ratio"] = safe_divide(
        df["Professionals_Count"].fillna(0),
        df["Total_Participants"]
    )

    # ------------------------------------------------------------
    # Course structure / workload
    # ------------------------------------------------------------
    df["Total_Hours"] = (
        df["Practical_Hours"].fillna(0)
        + df["Theory_Hours"].fillna(0)
    )

    df["Practical_Ratio"] = safe_divide(
        df["Practical_Hours"].fillna(0),
        df["Total_Hours"]
    )

    df["Theory_Ratio"] = safe_divide(
        df["Theory_Hours"].fillna(0),
        df["Total_Hours"]
    )

    # ------------------------------------------------------------
    # Cost-related features
    # ------------------------------------------------------------
    df["Cost_Per_Hour"] = safe_divide(
        df["Daily_Tuition_Cost"],
        df["Total_Hours"]
    )

    df["Cost_Per_Participant"] = safe_divide(
        df["Daily_Tuition_Cost"],
        df["Total_Participants"]
    )

    # ------------------------------------------------------------
    # Previous history features
    # ------------------------------------------------------------
    df["Prev_Total_Courses"] = (
        df["Prev_Course_Dropouts"].fillna(0)
        + df["Prev_Course_Attended"].fillna(0)
    )

    df["Prev_Dropout_Rate"] = safe_divide(
        df["Prev_Course_Dropouts"].fillna(0),
        df["Prev_Total_Courses"]
    )

    df["Has_Previous_Course_History"] = (
        df["Prev_Total_Courses"] > 0
    ).astype(int)

    # ------------------------------------------------------------
    # Support and registration behavior
    # ------------------------------------------------------------
    df["Support_Per_Participant"] = safe_divide(
        df["Pre_Course_Supports_Tickets"].fillna(0),
        df["Total_Participants"]
    )

    df["Changes_Per_Waiting_Day"] = safe_divide(
        df["Registration_Changes"].fillna(0),
        df["Waiting_List_Days"].fillna(0) + 1
    )

    df["High_Registration_Changes"] = (
        df["Registration_Changes"] >= df["Registration_Changes"].median()
    ).astype(int)

    df["Has_Support_Tickets"] = (
        df["Pre_Course_Supports_Tickets"] > 0
    ).astype(int)

    # ------------------------------------------------------------
    # Missingness indicators
    # ------------------------------------------------------------
    important_missing_cols = [
        "Company_ID",
        "Agent_ID",
        "Registration_Days_Before",
        "Requested_Lab_Config",
        "Physical_Course_Kits",
        "Enrollment_Type",
        "Submission_Source",
        "Payment_Terms",
        "Origin_Country",
        "Catering_Package",
        "Daily_Tuition_Cost",
        "Students_Count",
    ]

    for col in important_missing_cols:
        if col in df.columns:
            df[f"{col}_Was_Missing"] = df[col].isna().astype(int)

    df["Has_Company"] = df["Company_ID"].notna().astype(int)
    df["Has_Agent"] = df["Agent_ID"].notna().astype(int)

    # ------------------------------------------------------------
    # Lab configuration mismatch
    # ------------------------------------------------------------
    df["Lab_Config_Mismatch"] = (
        df["Requested_Lab_Config"].fillna("Missing")
        != df["Assigned_Lab_Config"].fillna("Missing")
    ).astype(int)

    # ------------------------------------------------------------
    # Date features
    # ------------------------------------------------------------
    df["Course_Start_Date"] = pd.to_datetime(
        df["Course_Start_Date"],
        errors="coerce"
    )

    df["Course_Start_Year"] = df["Course_Start_Date"].dt.year
    df["Course_Start_Month"] = df["Course_Start_Date"].dt.month
    df["Course_Start_Quarter"] = df["Course_Start_Date"].dt.quarter
    df["Course_Start_DayOfWeek"] = df["Course_Start_Date"].dt.dayofweek
    df["Course_Start_WeekOfYear"] = (
        df["Course_Start_Date"].dt.isocalendar().week.astype("Int64")
    )
    df["Course_Start_IsWeekend"] = (
        df["Course_Start_DayOfWeek"].isin([5, 6]).astype(int)
    )

    return df