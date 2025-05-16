import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts
model = joblib.load('mental_health_model.joblib')
le = joblib.load('label_encoder.joblib')
valid_cols = joblib.load('valid_cols.joblib')
combined_scale = joblib.load('combined_scale.joblib')

# Define question sections
question_sections = {
    "Academic Pressure": [
        "I feel overwhelmed by my coursework.",
        "I struggle to meet academic deadlines."
    ],
    "Social Life": [
        "I find it hard to make new friends.",
        "I feel isolated from my peers."
    ],
    "Family Expectations": [
        "My family's expectations cause me stress.",
        "I fear disappointing my family."
    ],
    "Personal Well-being": [
        "I often feel anxious or depressed.",
        "I have trouble sleeping due to stress."
    ]
}

# Prepare response options (exclude nan if present)
response_options = sorted([k for k in combined_scale.keys() if pd.notna(k)])

st.set_page_config(page_title="Student Mental Health Self-Assessment", layout="centered")
st.title("📝 Student Mental Health Self-Assessment")

# Create a form for user input
with st.form(key='assessment_form'):
    responses = {}

    for section, questions in question_sections.items():
        with st.expander(section):
            for q in questions:
                responses[q] = st.selectbox(q, response_options, key=q)

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    # Map responses to numeric scores, default to 0 if unknown
    numeric_responses = [combined_scale.get(res, 0) for res in responses.values()]
    df_input = pd.DataFrame([numeric_responses], columns=responses.keys()).fillna(0)

    # Calculate risk score and health index
    risk_score = df_input.sum(axis=1).values[0]
    max_score = len(responses) * 4
    health_index = 100 - (risk_score / max_score * 100)

    # Predict risk category
    predicted_label_encoded = model.predict(df_input)[0]
    predicted_risk_category = le.inverse_transform([predicted_label_encoded])[0]

    # Show results
    st.markdown(f"### 🧠 Your Health Index: `{health_index:.2f}%`")
    st.markdown(f"### 📊 Predicted Risk Category: **{predicted_risk_category}**")

    # Optional: Add visual feedback
    if health_index > 80:
        st.success("Great job! Your mental health appears to be in good shape.")
    elif health_index > 50:
        st.warning("Caution: There are some areas to watch. Consider seeking support.")
    else:
        st.error("Alert: Your responses indicate potential mental health risks. Please consult a professional.")
