import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts
model = joblib.load('mental_health_model.joblib')
le = joblib.load('label_encoder.joblib')
valid_cols = joblib.load('valid_cols.joblib')
combined_scale = joblib.load('combined_scale.joblib')

st.set_page_config(page_title="Student Mental Health Self-Assessment", layout="centered")

st.title("📝 Student Mental Health Self-Assessment")

# Prepare response options (exclude nan if present)
response_options = sorted([k for k in combined_scale.keys() if pd.notna(k)])

# Create a form for user input
with st.form(key='assessment_form'):
    responses = {}
    for q in valid_cols:
        responses[q] = st.selectbox(q, response_options)

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    # Map responses to numeric scores, default to 0 if unknown
    numeric_responses = [combined_scale.get(res, 0) for res in responses.values()]
    df_input = pd.DataFrame([numeric_responses], columns=valid_cols).fillna(0)

    # Calculate risk score and health index
    risk_score = df_input.sum(axis=1).values[0]
    max_score = len(valid_cols) * 4
    health_index = 100 - (risk_score / max_score * 100)

    # Predict risk category
    predicted_label_encoded = model.predict(df_input)[0]
    predicted_risk_category = le.inverse_transform([predicted_label_encoded])[0]

    # Show results
    st.write(f"### Your Health Index: {health_index:.2f}%")
    st.write(f"### Predicted Risk Category: {predicted_risk_category}")
