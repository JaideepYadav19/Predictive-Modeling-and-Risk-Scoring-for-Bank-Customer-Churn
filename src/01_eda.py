import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


RAW_DATA_PATH = "D:/Projects/Jaideep/data/raw/European_Bank.csv"
VISUALS_PATH = "D:/Projects/Jaideep/reports/visuals"


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def basic_dataset_report(df):
    print("\n========== DATASET BASIC INFO ==========")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nTarget Distribution:")
    print(df["Exited"].value_counts())

    print("\nTarget Distribution Percentage:")
    print(df["Exited"].value_counts(normalize=True) * 100)


def save_countplot(df, column, title, filename):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column, hue="Exited")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Customer Count")
    plt.tight_layout()
    plt.savefig(f"{VISUALS_PATH}/{filename}")
    plt.close()


def save_histplot(df, column, title, filename):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, hue="Exited", bins=30, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{VISUALS_PATH}/{filename}")
    plt.close()


def perform_eda(df):
    create_folder(VISUALS_PATH)

    # 1. Churn distribution
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="Exited")
    plt.title("Customer Churn Distribution")
    plt.xlabel("Exited")
    plt.ylabel("Customer Count")
    plt.tight_layout()
    plt.savefig(f"{VISUALS_PATH}/churn_distribution.png")
    plt.close()

    # 2. Categorical feature analysis
    save_countplot(df, "Geography", "Churn by Geography", "churn_by_geography.png")
    save_countplot(df, "Gender", "Churn by Gender", "churn_by_gender.png")
    save_countplot(df, "HasCrCard", "Churn by Credit Card Ownership", "churn_by_credit_card.png")
    save_countplot(df, "IsActiveMember", "Churn by Active Membership", "churn_by_active_member.png")
    save_countplot(df, "NumOfProducts", "Churn by Number of Products", "churn_by_products.png")
    save_countplot(df, "Tenure", "Churn by Tenure", "churn_by_tenure.png")

    # 3. Numerical feature analysis
    save_histplot(df, "Age", "Age Distribution by Churn", "age_distribution_churn.png")
    save_histplot(df, "CreditScore", "Credit Score Distribution by Churn", "credit_score_churn.png")
    save_histplot(df, "Balance", "Balance Distribution by Churn", "balance_churn.png")
    save_histplot(df, "EstimatedSalary", "Estimated Salary Distribution by Churn", "salary_churn.png")

    # 4. Correlation heatmap
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    plt.figure(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{VISUALS_PATH}/correlation_heatmap.png")
    plt.close()

    print("\nEDA visuals saved successfully in reports/visuals/")


def main():
    df = load_data()
    basic_dataset_report(df)
    perform_eda(df)


if __name__ == "__main__":
    main()