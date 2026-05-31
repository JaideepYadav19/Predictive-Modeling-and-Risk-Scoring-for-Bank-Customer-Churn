import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


MODEL_PATH = "models/trained_models/churn_model.pkl"
COLUMNS_PATH = "models/trained_models/model_columns.pkl"
BEST_MODEL_NAME_PATH = "models/trained_models/best_model_name.pkl"
MODEL_COMPARISON_PATH = "reports/model_comparison.csv"
FEATURE_IMPORTANCE_PATH = "reports/feature_importance.csv"


def load_model():
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    best_model_name = joblib.load(BEST_MODEL_NAME_PATH)
    return model, model_columns, best_model_name


def risk_category(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.70:
        return "Medium Risk"
    else:
        return "High Risk"


def prepare_input(data, model_columns):
    df = pd.DataFrame([data])

    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ProductEngagement"] = df["NumOfProducts"] * df["IsActiveMember"]
    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)

    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    return df


def project_overview():
    st.title("Bank Customer Churn Prediction System")

    st.write("""
    This dashboard predicts whether a bank customer is likely to churn.
    It provides a churn probability score and classifies the customer into
    Low, Medium, or High risk categories.
    """)

    st.subheader("Project Objective")
    st.write("""
    The objective is to help banks identify customers who are likely to leave
    before they actually churn, so that proactive retention strategies can be applied.
    """)

    st.subheader("System Flow")
    st.write("""
    Dataset → Preprocessing → Feature Engineering → Model Training → Risk Scoring → Dashboard
    """)


def churn_predictor(model, model_columns):
    st.title("Customer Churn Risk Calculator")

    st.subheader("Enter Customer Details")

    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card?", [1, 0])
    is_active = st.selectbox("Is Active Member?", [1, 0])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0)

    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": estimated_salary
    }

    if st.button("Predict Churn Risk"):
        final_input = prepare_input(input_data, model_columns)

        probability = model.predict_proba(final_input)[0][1]
        prediction = model.predict(final_input)[0]
        risk = risk_category(probability)

        st.subheader("Prediction Result")

        st.metric("Churn Probability", f"{probability:.2%}")
        st.metric("Risk Category", risk)

        if prediction == 1:
            st.error("The customer is likely to churn.")
        else:
            st.success("The customer is likely to stay.")

        fig, ax = plt.subplots()
        ax.bar(["Retention Probability", "Churn Probability"], [1 - probability, probability])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Customer Churn Probability")
        st.pyplot(fig)


def model_performance():
    st.title("Model Performance")

    try:
        results = pd.read_csv(MODEL_COMPARISON_PATH)
        st.dataframe(results)
    except FileNotFoundError:
        st.warning("Model comparison file not found. Run 03_model_training.py first.")


def feature_importance():
    st.title("Feature Importance Dashboard")

    try:
        importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        st.dataframe(importance.head(15))

        fig, ax = plt.subplots(figsize=(8, 5))
        top_features = importance.head(10)
        ax.barh(top_features["Feature"], top_features["Importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Churn Drivers")
        st.pyplot(fig)

    except FileNotFoundError:
        st.warning("Feature importance file not found. Run 04_explainability.py first.")


def what_if_simulator(model, model_columns):
    st.title("What-if Churn Scenario Simulator")

    st.write("""
    Change customer engagement and product values to see how churn risk changes.
    """)

    base_data = {
        "CreditScore": st.slider("Credit Score", 300, 900, 650),
        "Geography": st.selectbox("Geography", ["France", "Germany", "Spain"], key="geo_sim"),
        "Gender": st.selectbox("Gender", ["Male", "Female"], key="gender_sim"),
        "Age": st.slider("Age", 18, 100, 40),
        "Tenure": st.slider("Tenure", 0, 10, 5),
        "Balance": st.number_input("Balance", min_value=0.0, value=50000.0, key="balance_sim"),
        "NumOfProducts": st.slider("Number of Products", 1, 4, 1),
        "HasCrCard": st.selectbox("Has Credit Card?", [1, 0], key="card_sim"),
        "IsActiveMember": st.selectbox("Is Active Member?", [1, 0], key="active_sim"),
        "EstimatedSalary": st.number_input("Estimated Salary", min_value=0.0, value=100000.0, key="salary_sim")
    }

    final_input = prepare_input(base_data, model_columns)
    probability = model.predict_proba(final_input)[0][1]

    st.metric("Updated Churn Probability", f"{probability:.2%}")
    st.metric("Risk Category", risk_category(probability))


def main():
    st.set_page_config(
        page_title="Bank Churn Prediction",
        page_icon="🏦",
        layout="wide"
    )

    model, model_columns, best_model_name = load_model()

    st.sidebar.title("Navigation")
    st.sidebar.write(f"Best Model: {best_model_name}")

    page = st.sidebar.radio(
        "Go to",
        [
            "Project Overview",
            "Churn Predictor",
            "Model Performance",
            "Feature Importance",
            "What-if Simulator"
        ]
    )

    if page == "Project Overview":
        project_overview()

    elif page == "Churn Predictor":
        churn_predictor(model, model_columns)

    elif page == "Model Performance":
        model_performance()

    elif page == "Feature Importance":
        feature_importance()

    elif page == "What-if Simulator":
        what_if_simulator(model, model_columns)


if __name__ == "__main__":
    main()