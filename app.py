import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please check your repository.")
    st.stop()

# App title
st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# Sliders for numeric inputs
applicant_income = st.slider("Applicant Income", min_value=0, max_value=100000, value=5000, step=500)
coapplicant_income = st.slider("Coapplicant Income", min_value=0, max_value=50000, value=2000, step=500)
loan_amount = st.slider("Loan Amount (in thousands)", min_value=0, max_value=700, value=120, step=10)
loan_amount_term = st.slider("Loan Amount Term (in months)", min_value=12, max_value=480, value=360, step=12)

credit_history_label = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area_label = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert inputs to numeric codes
gender_code = 1 if gender == "Male" else 0
married_code = 1 if married == "Yes" else 0
dependents_code = 3 if dependents == "3+" else int(dependents)
education_code = 0 if education == "Graduate" else 1
self_employed_code = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history_label == "Good (1)" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area_label]

# Create input array
input_data = np.array([[gender_code, married_code, dependents_code, education_code,
                        self_employed_code, applicant_income, coapplicant_income,
                        loan_amount, loan_amount_term, credit_history, property_area]])

# Scale numeric features
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Loan Default"):
    prediction = model.predict(input_scaled)
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error("❌ High Risk: Loan Likely to Default.")
    else:
        st.success("✅ Low Risk: Loan Likely to be Approved.")

    if prob is not None:
        st.info(f"🔍 Probability of Default: **{prob:.2%}**")

    # ---------------- Charts ---------------- #

    st.subheader("📊 Income Distribution")
    st.bar_chart({
        'Income': [applicant_income, coapplicant_income]
    }, use_container_width=True)

    st.subheader("📈 Credit History Pie Chart")
    labels = ['Good Credit', 'Bad Credit']
    sizes = [credit_history, 1 - credit_history]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#F44336"])
    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("🧾 Input Summary")
    st.table(pd.DataFrame({
        "Feature": ["Gender", "Married", "Dependents", "Education", "Self Employed", "Applicant Income",
                    "Coapplicant Income", "Loan Amount", "Loan Term", "Credit History", "Property Area"],
        "Value": [gender, married, dependents, education, self_employed, applicant_income,
                  coapplicant_income, loan_amount, loan_amount_term, credit_history_label, property_area_label]
    }))
