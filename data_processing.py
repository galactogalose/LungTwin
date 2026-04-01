import pandas as pd

def load_and_process_data(path):
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df = df.drop(columns=[
        "Unnamed: 0", "MWT1", "MWT2", "AGEquartiles", "copd"
    ])

    # Fill missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Feature engineering
    df["LungEfficiency"] = df["FEV1"] / df["FEV1PRED"]

    # Smoking category
    def smoking_category(x):
        if x < 10:
            return 0
        elif x < 30:
            return 1
        else:
            return 2

    df["SmokingRisk"] = df["PackHistory"].apply(smoking_category)

    return df