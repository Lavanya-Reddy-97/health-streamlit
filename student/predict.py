import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and encoders
model = joblib.load("mental_health_model.joblib")
le = joblib.load("label_encoder.joblib")
combined_scale = joblib.load("combined_scale.joblib")
valid_cols = joblib.load("valid_cols.joblib")

st.set_page_config(page_title="Mental Health Self-Assessment", layout="centered")
st.title("🧠 Student Mental Health Self-Assessment")

# --- Group feature names into logical sections (based on training features)
section_map = {
    "Personal and Social Well-being": [col for col in valid_cols if "personal" in col.lower() or "social" in col.lower()],
    "ADHD Symptoms & Behavior": [col for col in valid_cols if "focus" in col.lower() or "distracted" in col.lower() or "assignment" in col.lower()],
    "Academic Satisfaction": [col for col in valid_cols if "academic" in col.lower()],
    "Emotional Well-being": [col for col in valid_cols if "emotion" in col.lower() or "anxious" in col.lower() or "happy" in col.lower()],
    "Other Factors": [col for col in valid_cols if col not in sum(list(section_map.values()), [])]
}

# --- Begin form
with st.form("assessment_form"):
    responses = {}
    for section, questions in section_map.items():
        with st.expander(section, expanded=False):
            for q in questions:
                options = ["Choose an option"] + [opt for opt in combined_scale.keys() if pd.notna(opt)]
                selected = st.selectbox(q, options, index=0, key=q)
                responses[q] = selected

    submitted = st.form_submit_button("Submit")

# --- Handle submission
if submitted:
    # Validate responses
    if any(r == "Choose an option" for r in responses.values()):
        st.warning("Please answer all questions before submitting.")
        st.stop()

    # Convert to numeric features
    input_scores = [combined_scale.get(r, 0) for r in responses.values()]
    df_input = pd.DataFrame([input_scores], columns=valid_cols).fillna(0)

    # Compute risk score and health index
    risk_score = df_input.sum(axis=1).values[0]
    max_score = len(valid_cols) * 4
    health_index = 100 - (risk_score / max_score * 100)

    # Predict category
    pred_label = model.predict(df_input)[0]
    pred_category = le.inverse_transform([pred_label])[0]

    # Display results
    st.subheader("🧾 Assessment Result")
    st.metric("Health Index", f"{health_index:.2f}%")
    st.metric("Predicted Risk Category", pred_category)

    if health_index > 80:
        st.success("✅ You appear to be in good mental health.")
    elif health_index > 50:
        st.warning("⚠️ Moderate risk detected. Consider talking to someone you trust.")
    else:
        st.error("🚨 High risk detected. Please reach out for professional support.")
