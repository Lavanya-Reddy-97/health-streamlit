import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts
model = joblib.load('mental_health_model.joblib')
le = joblib.load('label_encoder.joblib')
combined_scale = joblib.load('combined_scale.joblib')

# Define sections and questions
question_sections = {
    "Personal and Social Well-being": [
        "I am able to quickly adapt to changes in life",
        "I feel accepted by my peers",
        "I feel comfortable expressing myself with others"
    ],
    "ADHD Symptoms & Behavior": [
        "I have trouble focusing during lectures or study time",
        "I often act impulsively without thinking",
        "I feel the need to constantly move or fidget"
    ],
    "Academic Satisfaction and Anxiety": [
        "I feel overwhelmed by my coursework",
        "I struggle to meet academic deadlines",
        "I am satisfied with my academic performance"
    ],
    "Online Learning Experience": [
        "I find it easy to learn through online classes",
        "I stay engaged during virtual sessions",
        "I miss in-person interaction with teachers and peers"
    ],
    "Emotional Well-being": [
        "I often feel sad or down",
        "I feel anxious without clear reason",
        "I feel hopeful about my future"
    ],
    "Coping and Support Strategies": [
        "I know where to go when I need help",
        "I use healthy ways to cope with stress",
        "I feel supported by my school community"
    ],
    "Eating Habits & Body Image": [
        "I have a healthy relationship with food",
        "I feel confident about my body image",
        "I skip meals regularly"
    ],
    "Mental Health and Emotional Safety": [
        "I have thoughts of hurting myself",
        "I find it difficult to share my feelings with others",
        "I feel emotionally safe at school"
    ],
    "Risky Activities": [
        "I take part in risky activities"
    ]
}

# Define custom options per section
category_options = {
    "Risky Activities": [
        "None",
        "Skipping classes",
        "Bullying or getting into fights",
        "Engaging in unsafe actions (e.g., reckless behavior)",
        "Experimenting with substances"
    ]
}

# Default Likert-style options for other categories
likert_options = ["Always", "Often", "Sometimes", "Rarely", "Never"]

# Build UI
st.set_page_config(page_title="Student Mental Health Self-Assessment", layout="centered")
st.title("📝 Student Mental Health Self-Assessment")

with st.form(key='assessment_form'):
    responses = {}

    for section, questions in question_sections.items():
        with st.expander(f"{section}"):
            for q in questions:
                options = category_options.get(section, likert_options)
                responses[q] = st.selectbox(q, options, key=q)

    submit = st.form_submit_button("Submit")

if submit:
    # Map responses to numeric values using combined_scale
    numeric_responses = [combined_scale.get(val, 0) for val in responses.values()]
    df_input = pd.DataFrame([numeric_responses], columns=responses.keys()).fillna(0)

    # Health calculation
    risk_score = df_input.sum(axis=1).values[0]
    max_score = len(responses) * 4
    health_index = 100 - (risk_score / max_score * 100)

    # Predict risk
    predicted_label_encoded = model.predict(df_input)[0]
    predicted_risk = le.inverse_transform([predicted_label_encoded])[0]

    # Display results
    st.markdown(f"### 🧠 Your Health Index: `{health_index:.2f}%`")
    st.markdown(f"### 📊 Predicted Risk Category: **{predicted_risk}**")

    if health_index > 80:
        st.success("✅ Great job! Your mental health appears to be in good shape.")
    elif health_index > 50:
        st.warning("⚠️ You may be experiencing some challenges. Consider seeking support.")
    else:
        st.error("🚨 Your responses indicate a higher risk. Please reach out to a counselor or mental health professional.")
