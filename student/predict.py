import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your saved model and artifacts (make sure these files are in the same folder)
model = joblib.load('mental_health_model.joblib')
le = joblib.load('label_encoder.joblib')
combined_scale = joblib.load('combined_scale.joblib')

st.set_page_config(page_title="Student Mental Health Self-Assessment", layout="centered")
st.title("📝 Student Mental Health Self-Assessment")

# All questions grouped by category with options
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
        ("I worry that people don’t like me", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
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
        ("Do you often lose things necessary for tasks (e.g., books, pencils, assignments)?", ["Occasionally", "Sometimes", "Often", "Always", "Never"]),
    ],
    "Academic Satisfaction and Anxiety": [
        ("I am satisfied with my academic performance", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("I am able to submit my assignments on time", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
        ("What prevents you from doing well academically? (Select all that apply)", [
            "Lack of concentration", "Poor study habits", "Difficulty managing time", "Distractions", "Difficulty understanding content", "Other"
        ]),
        ("I feel anxious due to: (Select all that apply)", [
            "Studies", "Exams", "Results", "College admissions", "Career", "I don’t feel anxious", "Other"
        ]),
        ("I feel valued when I perform well academically", ["Yes", "No"]),
        ("I get bothered when teachers don’t notice my efforts", ["Yes", "No", "Neither"]),
        ("I feel like I don’t belong or fit in at school", ["Yes", "No", "Neither"]),
        ("I feel jealous of others who are more popular or successful", ["Yes", "No", "Neither"]),
        ("I don’t enjoy group activities or school events", ["Yes", "No", "Neither"]),
    ],
    "Online Learning Experience": [
        ("I find online classes better than offline classes", ["Yes", "No", "Maybe"]),
        ("What challenges did you face during online classes? (Select all that apply)", [
            "Difficulty understanding content", "Technical issues", "Lack of time management", "Electricity/internet problems",
            "Personal reasons", "Lack of social interaction", "Other"
        ]),
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
        ("I have trouble following rules or instructions", ["Yes", "No"]),
    ],
    "Coping and Support Strategies": [
        ("When feeling low, I prefer: (Select all that apply)", [
            "Talking to friends", "Talking to parents", "Talking to teachers", "Solving on my own",
            "Ignoring the feelings", "Waiting for things to improve", "Other"
        ]),
        ("I believe I can solve challenging tasks", ["Yes", "No"]),
        ("I have people I can talk to about my feelings", ["Yes", "No"]),
        ("I usually cope with stress by: (Select all that apply)", [
            "Yoga or meditation", "Changing the way I think", "Writing a diary", "Watching TV", "Helping others", "Other"
        ]),
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
    ],
    "Mental Health and Emotional Safety": [
        ("I feel overwhelmed by my emotions", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"]),
        ("I have felt hopeless or helpless recently", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"]),
        ("I feel like life is not worth living", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"]),
        ("I have thoughts of hurting myself", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"]),
        ("I find it difficult to share my feelings with others", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"]),
        ("I feel emotionally safe at school", ["Always", "Often", "Sometimes", "Rarely", "Never", "Skip"]),
    ],
    "Risky Activities": [
        ("I take part in risky activities (Select all that apply)", [
            "Skipping classes",
            "Engaging in unsafe actions (e.g., reckless behavior, unsafe stunts)",
            "Bullying or getting into fights",
            "Experimenting with substances – alcohol, smoking, or other substances",
            "Driving without a license",
            "Stealing or shoplifting",
            "Running away from home",
            "Avoiding homework or assignments intentionally",
            "Using weapons or carrying dangerous objects",
            "None"
        ]),
    ],
}

with st.form("assessment_form"):
    responses = {}
    for section, questions in question_sections.items():
        with st.expander(section, expanded=False):
            for q, opts in questions:
                if "Select all that apply" in q or section in ["Coping and Support Strategies", "Academic Satisfaction and Anxiety", "Online Learning Experience", "Risky Activities"]:
                    # Multi-select questions
                    responses[q] = st.multiselect(q, opts, key=q)
                else:
                    options_with_placeholder = ["Choose an option"] + opts
                    responses[q] = st.selectbox(q, options_with_placeholder, index=0, key=q)

    submitted = st.form_submit_button("Submit")

if submitted:
    numeric_vals = []
    for question, answer in responses.items():
        if isinstance(answer, list):
            numeric_vals.append(len(answer) if answer else 0)
        else:
            if answer == "Choose an option":
                numeric_vals.append(0)
            else:
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
