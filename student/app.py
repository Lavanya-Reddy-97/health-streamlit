import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# --- CONFIG ---
st.set_page_config(page_title="Student Mental Health Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>body { background-color: #f5f7fa; }</style>", unsafe_allow_html=True)

st.title("🧠 Student Mental Health Risk Assessment Dashboard")
st.write("Current working directory:", os.getcwd())  # Debugging path display

# --- FILE UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload student survey CSV", type=["csv"])
if not uploaded_file:
    st.warning("Please upload the `Student_Survey_Responses_300.csv` file.")
    st.stop()

df = pd.read_csv(uploaded_file, encoding='windows-1252')

# --- RESPONSE SCALE MAPPINGS ---
freq_scale = {
    'Never': 0,
    'Rarely': 1,
    'Sometimes': 2,
    'Often': 3,
    'Always': 4,
    'Skip': np.nan
}
yes_no_scale = {
    'Yes': 0,
    'No': 4,
    'Neither': 2,
    'Not Sure': 2,
    'Maybe': 2,
    'Skip': np.nan
}

combined_scale = {**freq_scale, **yes_no_scale}

# --- NON-SCORING COLUMNS ---
non_question_cols = ['Student SNO', 'Gender', 'Current Grade', 'Risk Activity']

# --- VALIDATE AND SELECT QUESTION COLUMNS ---
valid_cols = []
for col in df.columns:
    if col not in non_question_cols:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset(combined_scale.keys()):
            valid_cols.append(col)

# --- REPLACE RESPONSES WITH SCORES ---
df_numeric = df[valid_cols].replace(combined_scale).apply(pd.to_numeric, errors='coerce')

# --- CALCULATE SCORES ---
df['Risk_Score'] = df_numeric.sum(axis=1)
max_score = len(valid_cols) * 4
df['Health_Index_%'] = 100 - (df['Risk_Score'] / max_score * 100)

# --- RISK CATEGORY FUNCTION ---
def risk_category(health_idx):
    if health_idx > 70:
        return 'Low Risk'
    elif health_idx > 50:
        return 'Moderate Risk'
    else:
        return 'High Risk'

df['Risk_Category'] = df['Health_Index_%'].apply(risk_category)

# --- MACHINE LEARNING MODEL ---
X = df_numeric.fillna(0)
y = df['Risk_Category']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = LogisticRegression(multi_class='ovr', max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
df['Predicted_Risk'] = le.inverse_transform(model.predict(X))

# --- SAVE MODEL FILES ---
output_dir = os.path.dirname(__file__)  # Save to same folder as app.py
joblib.dump(model, os.path.join(output_dir, 'mental_health_model.joblib'))
joblib.dump(le, os.path.join(output_dir, 'label_encoder.joblib'))
joblib.dump(valid_cols, os.path.join(output_dir, 'valid_cols.joblib'))
joblib.dump(combined_scale, os.path.join(output_dir, 'combined_scale.joblib'))
st.success("✅ Model and encoders saved successfully to app folder!")

# --- SIDEBAR NAVIGATION ---
view = st.sidebar.radio("Choose View", ["📊 School Portfolio", "🧍 Student Portfolio"])

# --- SCHOOL PORTFOLIO VIEW ---
if view == "📊 School Portfolio":
    st.subheader("🏫 School Health Overview")

    col1, col2 = st.columns(2)
    with col1:
        avg_health = df['Health_Index_%'].mean()
        st.metric("Average School Health Index", f"{avg_health:.2f}%")
    with col2:
        risk_dist = df['Risk_Category'].value_counts(normalize=True) * 100
        st.write("Risk Distribution (%)")
        st.bar_chart(risk_dist)

    st.write("### All Students Summary")
    st.dataframe(df[['Student SNO', 'Gender', 'Current Grade', 'Health_Index_%', 'Risk_Category', 'Predicted_Risk']])

    st.write("### 🔥 Risk Score Heatmap (First 30 Students)")
    plt.figure(figsize=(12, 5))
    sns.heatmap(df_numeric.head(30), cmap='YlOrRd', cbar=True)
    st.pyplot(plt)

# --- STUDENT PORTFOLIO VIEW ---
if view == "🧍 Student Portfolio":
    st.subheader("🔍 Individual Student Review")

    selected_student = st.selectbox("Select Student SNO", df['Student SNO'])
    student_data = df[df['Student SNO'] == selected_student].iloc[0]

    st.write(f"**Gender:** {student_data['Gender']}")
    st.write(f"**Grade:** {student_data['Current Grade']}")
    st.metric("Health Index", f"{student_data['Health_Index_%']:.2f}%")
    st.metric("Risk Category", student_data['Risk_Category'])
    st.metric("Predicted Risk (ML)", student_data['Predicted_Risk'])

    st.write("### Risk Category Filter")
    risk_filter = st.selectbox("Filter Risk Category", ['All', 'Low Risk', 'Moderate Risk', 'High Risk'])
    if risk_filter != 'All':
        filtered_df = df[df['Risk_Category'] == risk_filter]
        st.write(f"Students in {risk_filter}: {len(filtered_df)}")

    st.write("### Response Breakdown")
    responses = df.loc[df['Student SNO'] == selected_student, valid_cols].T
    responses.columns = ['Response']
    st.dataframe(responses)

    st.write("### Additional Text-Based Responses")
    text_cols = [col for col in non_question_cols if col in df.columns and col not in ['Student SNO', 'Gender', 'Current Grade']]
    if text_cols:
        text_data = df.loc[df['Student SNO'] == selected_student, text_cols].T
        text_data.columns = ['Response']
        st.dataframe(text_data)
    else:
        st.write("No additional text-based responses found.")

# --- FOOTER ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | © 2025 Mental Health Project")
