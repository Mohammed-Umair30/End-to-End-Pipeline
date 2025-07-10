# telco_churn_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================
# Load trained pipeline
# ============================
model = joblib.load('telco_churn_rf_pipeline.joblib')

# ============================
# App title and description
# ============================
st.title("🔮 Telco Customer Churn Prediction App")
st.write("""
This app predicts whether a customer is likely to churn (leave) or not, 
based on their subscription and demographic information.  
Upload a CSV file or manually enter customer data below!  
""")

# ============================
# Sidebar - User option
# ============================
option = st.sidebar.radio("Choose input method:", ("Upload CSV", "Manual Input"))

# ============================
# If CSV upload
# ============================
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("### 📄 Uploaded Data Preview", data.head())

        # Predict
        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

        data['Churn_Probability'] = probabilities
        data['Churn_Predicted'] = np.where(predictions == 1, 'Yes', 'No')

        st.write("### ✅ Predictions")
        st.dataframe(data[['Churn_Probability', 'Churn_Predicted']])
else:
    st.write("### ✍️ Manually Input Customer Details")

    # Example manual input fields
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 2500.0)

    # Build dataframe
    input_df = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone],
        'MultipleLines': [multiple],
        'InternetService': [internet],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless],
        'PaymentMethod': [payment],
        'MonthlyCharges': [monthly],
        'TotalCharges': [total]
    })

    # Predict on button click
    if st.button("Predict Churn"):
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        st.write(f"**Churn Probability:** {proba:.2f}")
        st.write(f"**Churn Predicted:** {'Yes' if pred == 1 else 'No'}")

        # Explanation note
        st.info("""
        🔎 **Note**:  
        - Probability closer to 1 indicates higher chance of churn.  
        - Use this to plan retention strategies (e.g., special offers).  
        """)
