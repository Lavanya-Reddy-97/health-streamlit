import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
model = joblib.load('mental_health_model.joblib')
le = joblib.load('label_encoder.joblib')
valid_cols = joblib.load('valid_cols.joblib')
combined_scale = joblib.load('combined_scale.joblib')

st.set_page_config(page_title="Student Mental Health Self-Assessment", layout="centered")
st.title("📝 Student Mental Health Self-Assessment")

# Define question sections
question_sections = {
    "Personal and Social Well-being": [
        "I am able to quickly adapt to changes in life",
        "I trust others easily",
        "I feel satisfied with my personal life",
        "I feel satisfied with my school life",
        "I feel responsible for doing well in my life",
        "I feel confident about my body image",
        "I can understand and respect others' viewpoints",
        "I hesitate to ask questions in class",
        "I find it difficult to initiate conversations",
        "People in my life see me as a happy person",
        "My teachers see me as a good leader",
        "My friends consider me trustworthy",
        "I give in to peer pressure to fit in",
        "I worry that people don't like me"
    ],
    "ADHD Symptoms & Behavior": [
        "I often find it difficult to stay focused during class",
        "I have trouble finishing assignments or tasks on time",
        "I frequently forget things like assignments, books, or my personal items",
        "I get distracted easily, even by things that aren't part of the task at hand",
        "I often find myself daydreaming or thinking about unrelated things when I should be focusing",
        "I have trouble sitting still or staying in one place for a long time",
        "I talk or move around excessively, even when it's not appropriate",
        "I tend to act impulsively, without thinking about the consequences",
        "I often interrupt others or have trouble waiting for my turn during conversations",
        "I struggle with organizing tasks or managing my time effectively",
        "I often feel restless or find it hard to relax, even when I should be resting",
        "When I'm asked to complete a task, I often start it but don't finish it",
        "Do you often lose things necessary for tasks (e.g., books, pencils, assignments)?"
    ],
    "Academic Satisfaction and Anxiety": [
        "I am satisfied with my academic performance",
        "I am able to submit my assignments on time",
        "I feel valued when I perform well academically",
        "I get bothered when teachers don't notice my efforts",
        "I feel like I don't belong or fit in at school",
        "I feel jealous of others who are more popular or successful",
        "I don't enjoy group activities or school events"
    ],
    "Emotional Well-being": [
        "I often feel happy",
        "I often feel anxious",
        "I often feel lonely or tearful",
        "I feel hopeful during stressful situations",
        "I can understand others feelings and respond accordingly",
        "I get into fights with my classmates or friends",
        "I skip school or classes without a good reason",
        "I tend to lie or hide the truth to avoid trouble",
        "I have trouble following rules or instructions"
    ],
    "Coping and Support Strategies": [
        "I believe I can solve challenging tasks",
        "I have people I can talk to about my feelings"
    ],
    "Eating Habits & Body Image": [
        "I feel satisfied with my eating habits",
        "I skip meals intentionally",
        "I eat even when I'm not hungry due to stress or emotions",
        "I feel guilty after eating",
        "I avoid eating in front of others",
        "I worry excessively about gaining weight",
        "I feel pressure to look a certain way because of social media or peers",
        "I restrict food intake to control my weight",
        "Do you think your eating habits affect your emotional or physical well-being?"
    ],
    "Mental Health and Emotional Safety": [
        "I feel overwhelmed by my emotions",
        "I have felt hopeless or helpless recently",
        "I feel like life is not worth living",
        "I have thoughts of hurting myself",
        "I find it difficult to share my feelings with others",
        "I feel emotionally safe at school"
    ]
}

# --- Begin Form UI ---
with st.form("assessment_form"):
    st.markdown("Please complete the assessment by selecting the most accurate response for each statement.")
    responses = {}
    for section, questions in question_sections.items():
        with st.expander(section):
            for q in questions:
                options = ["Choose an option"] + list(combined_scale.keys())
                response = st.selectbox(q, options, key=q)
                responses[q] = response

    submitted = st.form_submit_button("Submit")

# --- On Submit ---
if submitted:
    unanswered = [q for q, a in responses.items() if a == "Choose an option"]
    if unanswered:
        st.error("⚠️ Please answer all questions before submitting.")
        st.stop()

    # Convert answers to numeric scores
    scores = [combined_scale.get(responses[q], 0) for q in valid_cols]
    df_input = pd.DataFrame([scores], columns=valid_cols).fillna(0)

    # Calculate index
    risk_score = df_input.sum(axis=1).values[0]
    max_score = len(valid_cols) * 4
    health_index = 100 - (risk_score / max_score * 100)

    # Predict risk category
    pred_encoded = model.predict(df_input)[0]
    pred_risk = le.inverse_transform([pred_encoded])[0]

    # Display
    st.markdown(f"### 🧠 Health Index: `{health_index:.2f}%`")
    st.markdown(f"### 📊 Predicted Risk Category: **{pred_risk}**")

    if health_index > 80:
        st.success("✅ Great job! Your mental health appears to be in good shape.")
    elif health_index > 50:
        st.warning("⚠️ You may be experiencing some challenges. Consider seeking support.")
    else:
        st.error("🚨 Your responses indicate a higher risk. Please seek help from a professional.")
