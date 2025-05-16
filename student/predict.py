import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and label encoder, plus combined_scale dict from your training session
model = joblib.load('mental_health_model.joblib')
le = joblib.load('label_encoder.joblib')
combined_scale = joblib.load('combined_scale.joblib')

st.set_page_config(page_title="Student Mental Health Self-Assessment", layout="centered")
st.title("📝 Student Mental Health Self-Assessment")

# Define all questions with their appropriate options grouped by section

question_sections = {
    "Personal and Social Well-being": [
        ("I am able to quickly adapt to changes in life", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I trust others easily", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I feel satisfied with my personal life", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I feel satisfied with my school life", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I feel responsible for doing well in my life", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I feel confident about my body image", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I can understand and respect others’ viewpoints", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I hesitate to ask questions in class", ["Yes", "No", "Sometimes"]),
        ("I find it difficult to initiate conversations", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("People in my life see me as a happy person", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("My teachers see me as a good leader", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("My friends consider me trustworthy", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I give in to peer pressure to fit in", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I worry that people don’t like me", ["Always", "Often", "Sometimes", "Rarely", "Never"])
    ],
    "ADHD Symptoms & Behavior": [
        ("I often find it difficult to stay focused during class", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I have trouble finishing assignments or tasks on time", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I frequently forget things like assignments, books, or my personal items", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I get distracted easily, even by things that aren’t part of the task at hand", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I often find myself daydreaming or thinking about unrelated things when I should be focusing", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I have trouble sitting still or staying in one place for a long time", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I talk or move around excessively, even when it's not appropriate", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I tend to act impulsively, without thinking about the consequences", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I often interrupt others or have trouble waiting for my turn during conversations", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I struggle with organizing tasks or managing my time effectively", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I often feel restless or find it hard to relax, even when I should be resting", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("When I’m asked to complete a task, I often start it but don’t finish it", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("Do you often lose things necessary for tasks (e.g., books, pencils, assignments)?", ["Occasionally", "Sometimes", "Often", "Always", "Never"])
    ],
    "Academic Satisfaction and Anxiety": [
        ("I am satisfied with my academic performance", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I am able to submit my assignments on time", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I feel valued when I perform well academically", ["Yes", "No"]),
        ("I get bothered when teachers don’t notice my efforts", ["Yes", "No", "Neither"]),
        ("I feel like I don’t belong or fit in at school", ["Yes", "No", "Neither"]),
        ("I feel jealous of others who are more popular or successful", ["Yes", "No", "Neither"]),
        ("I don’t enjoy group activities or school events", ["Yes", "No", "Neither"])
    ],
    "Online Learning Experience": [
        ("I find online classes better than offline classes", ["Yes", "No", "Maybe"]),
        ("I find it difficult to focus during online classes", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ],
    "Emotional Well-being": [
        ("I often feel happy", ["Always", "Sometimes", "Often", "Rarely", "Never"]),
        ("I often feel anxious", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I often feel lonely or tearful", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I feel hopeful during stressful situations", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I can understand others feelings and respond accordingly", ["Yes", "No"]),
        ("I get into fights with my classmates or friends", ["Yes", "No"]),
        ("I skip school or classes without a good reason", ["Yes", "No"]),
        ("I tend to lie or hide the truth to avoid trouble", ["Yes", "No"]),
        ("I have trouble following rules or instructions", ["Yes", "No"])
    ],
    "Coping and Support Strategies": [
        ("I believe I can solve challenging tasks", ["Yes", "No"]),
        ("I have people I can talk to about my feelings", ["Yes", "No"]),
    ],
    "Eating Habits & Body Image": [
        ("I feel satisfied with my eating habits", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I skip meals intentionally", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I eat even when I’m not hungry due to stress or emotions", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I feel guilty after eating", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I avoid eating in front of others", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I worry excessively about gaining weight", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I feel pressure to look a certain way because of social media or peers", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I restrict food intake to control my weight", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("Do you think your eating habits affect your emotional or physical well-being?", ["Yes", "No", "Not Sure"]),
        # Note: Follow-up multi-select question omitted as it needs complex UI
    ],
    "Mental Health and Emotional Safety": [
        ("I feel overwhelmed by my emotions", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"]),
        ("I have felt hopeless or helpless recently", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"]),
        ("I feel like life is not worth living", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"]),
        ("I have thoughts of hurting myself", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"]),
        ("I find it difficult to share my feelings with others", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"]),
        ("I feel emotionally safe at school", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"])
    ],   
}

# --- Begin UI ---

with st.form("assessment_form"):
    responses = {}
    for section, qs in question_sections.items():
        with st.expander(section, expanded=False):
            for q, opts in qs:
                if "Select all that apply" in q or len(opts) > 5 and section in ["Coping and Support Strategies", "Academic Satisfaction and Anxiety", "Online Learning Experience"]:
                    # Multi-select special cases (skip for scoring or handle differently)
                    responses[q] = st.multiselect(q, opts, key=q)
                elif section == "Risky Activities":
                    responses[q] = st.multiselect(q, opts, key=q)
                else:
                    responses[q] = st.selectbox(q, opts, key=q)

    submitted = st.form_submit_button("Submit")

if submitted:
    # For multi-select responses, skip or encode as zero (could be improved)
    numeric_vals = []
    for question, answer in responses.items():
        if isinstance(answer, list):
            # Multi-select: count number of selections as risk factor (or zero if none)
            numeric_vals.append(len(answer) if answer else 0)
        else:
            # Map to combined_scale; fallback 0
            numeric_vals.append(combined_scale.get(answer, 0))

    df_input = pd.DataFrame([numeric_vals], columns=responses.keys()).fillna(0)

    risk_score = df_input.sum(axis=1).values[0]
    max_score = len(responses) * 4
    health_index = 100 - (risk_score / max_score * 100)

    pred_label_enc = model.predict(df_input)[0]
    pred_risk_cat = le.inverse_transform([pred_label_enc])[0]

    st.markdown(f"### 🧠 Your Health Index: `{health_index:.2f}%`")
    st.markdown(f"### 📊 Predicted Risk Category: **{pred_risk_cat}**")

    if health_index > 80:
        st.success("✅ Great job! Your mental health appears to be in good shape.")
    elif health_index > 50:
        st.warning("⚠️ You may be experiencing some challenges. Consider seeking support.")
    else:
        st.error("🚨 Your responses indicate a higher risk. Please seek help from a professional.")
