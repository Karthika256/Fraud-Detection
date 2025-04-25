import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection App")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details below and click **Predict** to check for fraud.")

# List of top 13 features
features = ['V10', 'V4', 'V14', 'V12', 'V11', 'V17', 'V16', 'V7', 'V3', 'V2', 'V21', 'V9', 'V18']

# Collect user inputs
user_inputs = {}
for feature in features:
    user_inputs[feature] = st.number_input(f"{feature}", value=0.0, format="%.5f")

# Convert to DataFrame
input_df = pd.DataFrame([user_inputs])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("🚨 Fraud Detected!")
    else:
        st.success("✅ Transaction is Safe.")

    st.write(f"**Probability of Fraud:** {proba:.2%}")
