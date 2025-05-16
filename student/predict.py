import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model artifacts
model = joblib.load('mental_health_model.joblib')
le = joblib.load('label_encoder.joblib')
combined_scale = joblib.load('combined_scale.joblib')

# --- Define all questions grouped by categories (based on your PDF) ---

question_sections = {
    "Personal and Social Well-being": [
        "I am able to quickly adapt to changes in life",
        "I feel accepted by my peers",
        "I feel comfortable expressing myself with others",
        # ... add all relevant questions here
    ],
    "ADHD Symptoms & Behavior": [
        "I have trouble focusing during lectures or study time",
        "I often act impulsively without thinking",
        "I feel the need to constantly move or fidget",
        # ... add all relevant questions here
    ],
    "Academic Satisfaction and Anxiety": [
        "I feel overwhelmed by my coursework",
        "I struggle to meet academic deadlines",
        "I am satisfied with my academic performance",
        # ... add all relevant questions here
    ],
    "Online Learning Experience": [
        "I find it easy to learn through online classes",
        "I stay engaged during virtual sessions",
        "I miss in-person interaction with teachers and peers",
        # ... add all relevant questions here
    ],
    "Emotional Well-being": [
        "I often feel sad or down",
        "I feel anxious without clear reason",
        "I feel hopeful about my future",
        # ... add all relevant questions here
    ],
    "Coping and Support Strategies": [
        "I know where to go when I need help",
        "I use healthy ways to cope with stress",
        "I feel supported by my school community",
        # ... add all relevant questions here
    ],
    "Eating Habits & Body Image": [
        "I have a healthy relationship with food",
        "I feel confident about my body image",
        "I skip meals regularly",
        # ... add all relevant questions here
    ],
    "Mental Health and Emotional Safety": [
        "I have thoughts of hurting myself",
        "I find it difficult to share my feelings with others",
        "I feel emotionally safe at school",
        # ... add all relevant questions here
    ],
    "Risky Activities": [
        "I take part in risky activities",
        # You can add specific risky activities here if needed
    ]
}

# --- Define answer options per category ---
default_options = ["Always", "Often", "Sometimes", "Rarely", "Never"]
yes_no_options = ["Yes", "No", "Maybe", "Neither", "Not Sure"]
risky_activities_options = [
    "None",
    "Skipping classes",
    "Bullying or getting into fights",
    "Engaging in unsafe actions (e.g., reckless behavior)",
    "Experimenting with substances"
]

category_options = {
    "Risky Activities": risky_activities_options,
    # Add any other categories with specific options here if needed
}

# --- Streamlit UI ---
st.set_page_config(page_title="Student Mental Health Self-Assessment", layout="centered")
st.title("📝 Student Mental Health Self-Assessment")

with st.form(key="assessment_form"):
    responses = {}
    for section, questions in question_sections.items():
        with st.expander(section, expanded=True):
            options = category_options.get(section, default_options)
            for q in questions:
                responses[q] = st.selectbox(q, options, key=q)

    submit = st.form_submit_button("Submit")

if submit:
    # Convert responses to numeric using combined_scale
    numeric_responses = [combined_scale.get(val, 0) for val in responses.values()]
    df_input = pd.DataFrame([numeric_responses], columns=responses.keys()).fillna(0)

    # Calculate health index
    risk_score = df_input.sum(axis=1).values[0]
    max_score = len(responses) * 4
    health_index = 100 - (risk_score / max_score * 100)

    # Predict risk category
    pred_label_enc = model.predict(df_input)[0]
    pred_risk_cat = le.inverse_transform([pred_label_enc])[0]

    # Show results
    st.markdown(f"### 🧠 Your Health Index: `{health_index:.2f}%`")
    st.markdown(f"### 📊 Predicted Risk Category: **{pred_risk_cat}**")

    # Visual feedback
    if health_index > 80:
        st.success("✅ Great job! Your mental health appears to be in good shape.")
    elif health_index > 50:
        st.warning("⚠️ You may be experiencing some challenges. Consider seeking support.")
    else:
        st.error("🚨 Your responses indicate a higher risk. Please seek help from a professional.")
