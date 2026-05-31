import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


PROCESSED_PATH = "data/processed"
MODEL_PATH = "models/trained_models"
REPORT_PATH = "reports"


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_processed_data():
    X_train = pd.read_csv(f"{PROCESSED_PATH}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_PATH}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_PATH}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{PROCESSED_PATH}/y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    return models


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_prob)
    }

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = get_models()
    results = []

    best_model = None
    best_model_name = None
    best_score = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

        if metrics["ROC AUC"] > best_score:
            best_score = metrics["ROC AUC"]
            best_model = model
            best_model_name = name

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="ROC AUC", ascending=False)

    return results_df, best_model, best_model_name


def save_outputs(results_df, best_model, best_model_name, X_train):
    create_folder(MODEL_PATH)
    create_folder(REPORT_PATH)

    results_df.to_csv(f"{REPORT_PATH}/model_comparison.csv", index=False)

    joblib.dump(best_model, f"{MODEL_PATH}/churn_model.pkl")
    joblib.dump(best_model_name, f"{MODEL_PATH}/best_model_name.pkl")
    joblib.dump(X_train.columns.tolist(), f"{MODEL_PATH}/model_columns.pkl")

    print("\nModel training completed successfully.")
    print(f"Best Model: {best_model_name}")
    print("\nModel comparison saved at:")
    print(f"{REPORT_PATH}/model_comparison.csv")
    print("\nBest model saved at:")
    print(f"{MODEL_PATH}/churn_model.pkl")


def main():
    X_train, X_test, y_train, y_test = load_processed_data()

    results_df, best_model, best_model_name = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nFinal Model Comparison:")
    print(results_df)

    save_outputs(results_df, best_model, best_model_name, X_train)


if __name__ == "__main__":
    main()