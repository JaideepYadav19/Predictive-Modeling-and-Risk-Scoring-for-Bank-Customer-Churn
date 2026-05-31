import os
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = "D:/Projects/Jaideep/data/raw/European_Bank.csv"
PROCESSED_PATH = "D:/Projects/Jaideep/data/processed"


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data():
    return pd.read_csv(RAW_DATA_PATH)


def preprocess_data(df):
    df = df.copy()

    # Drop non-informative columns
    columns_to_drop = ["CustomerId", "Surname"]

    if "Year" in df.columns:
        columns_to_drop.append("Year")

    df.drop(columns=columns_to_drop, inplace=True)

    # Feature engineering
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ProductEngagement"] = df["NumOfProducts"] * df["IsActiveMember"]
    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)

    # Separate target
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # One-hot encoding
    X = pd.get_dummies(
        X,
        columns=["Geography", "Gender"],
        drop_first=True
    )

    return X, y


def split_and_save_data(X, y):
    create_folder(PROCESSED_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train.to_csv(f"{PROCESSED_PATH}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_PATH}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_PATH}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_PATH}/y_test.csv", index=False)

    print("\nPreprocessing completed successfully.")
    print("Processed files saved in data/processed/")

    print("\nShapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("\nFinal Model Features:")
    print(X_train.columns.tolist())


def main():
    df = load_data()
    X, y = preprocess_data(df)
    split_and_save_data(X, y)


if __name__ == "__main__":
    main()