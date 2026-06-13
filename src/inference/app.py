import streamlit as st

from src.inference.predict import predict

st.set_page_config(
    page_title="Customer Churn Prediction", page_icon="📊", layout="centered"
)

st.title("📊 Customer Churn Prediction")

st.markdown("""
    Enter customer details below to predict whether
    the customer is likely to churn.
    """)

# -----------------------------
# Customer Inputs
# -----------------------------

gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox("Senior Citizen", [0, 1])

partner = st.selectbox("Has Partner?", ["Yes", "No"])

dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

online_security = st.selectbox("Online Security", ["Yes", "No"])

online_backup = st.selectbox("Online Backup", ["Yes", "No"])

device_protection = st.selectbox("Device Protection", ["Yes", "No"])

tech_support = st.selectbox("Tech Support", ["Yes", "No"])

streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])

streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

# -----------------------------
# Input Dictionary
# -----------------------------

input_dict = {
    "customerID": "0000-BGTRT",
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": tenure * monthly_charges,
}

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    prediction, yes_prob, no_prob = predict(input_dict)

    st.divider()

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")

    st.markdown(f"### Churn Probability: {yes_prob:.2f}")

    st.markdown(f"### No Churn Probability: {no_prob:.2f}")

    st.progress(int(yes_prob * 100))

    st.caption("Prediction generated using the trained MLflow registered model.")
