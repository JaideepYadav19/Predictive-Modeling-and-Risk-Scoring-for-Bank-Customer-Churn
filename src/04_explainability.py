import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROCESSED_PATH = "data/processed"
MODEL_PATH = "models/trained_models"
VISUALS_PATH = "reports/visuals"


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_files():
    model = joblib.load(f"{MODEL_PATH}/churn_model.pkl")
    best_model_name = joblib.load(f"{MODEL_PATH}/best_model_name.pkl")
    X_train = pd.read_csv(f"{PROCESSED_PATH}/X_train.csv")

    return model, best_model_name, X_train


def plot_feature_importance(model, best_model_name, X_train):
    create_folder(VISUALS_PATH)

    if not hasattr(model, "feature_importances_"):
        print(f"{best_model_name} does not support feature_importances_.")
        print("Feature importance skipped.")
        return

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    importance_df.to_csv("reports/feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df.head(10),
        x="Importance",
        y="Feature"
    )
    plt.title(f"Top 10 Feature Importances - {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{VISUALS_PATH}/feature_importance.png")
    plt.close()

    print("\nTop 10 Important Features:")
    print(importance_df.head(10))

    print("\nFeature importance saved at:")
    print("reports/feature_importance.csv")
    print(f"{VISUALS_PATH}/feature_importance.png")


def main():
    model, best_model_name, X_train = load_files()
    plot_feature_importance(model, best_model_name, X_train)


if __name__ == "__main__":
    main()