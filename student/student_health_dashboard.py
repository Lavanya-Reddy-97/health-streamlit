import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- CONFIG ---
st.set_page_config(page_title="Student Mental Health Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>body { background-color: #f5f7fa; }</style>", unsafe_allow_html=True)

st.title("🧠 Student Mental Health Risk Assessment Dashboard")

# --- FILE UPLOAD ---
uploaded_file = st.sidebar.file_uploader("Upload student survey CSV", type=["csv"])
if not uploaded_file:
    st.warning("Please upload the `Student_Survey_Responses_300.csv` file.")
    st.stop()

df = pd.read_csv(uploaded_file, encoding='windows-1252')

# --- CONFIGURE SCALE ---
response_scale = {
    'Never': 0,
    'Rarely': 1,
    'Sometimes': 2,
    'Often': 3,
    'Always': 4,
    'Skip': np.nan
}
non_question_cols = ['Student SNO', 'Gender', 'Current Grade']
standard_responses = set(response_scale.keys())

# --- VALIDATE COLUMNS ---
valid_cols = []
for col in df.columns:
    if col not in non_question_cols:
        vals = set(df[col].dropna().unique())
        if vals.issubset(standard_responses):
            valid_cols.append(col)

# --- CALCULATE SCORES ---
df_numeric = df[valid_cols].replace(response_scale).apply(pd.to_numeric, errors='coerce')
df['Risk_Score'] = df_numeric.sum(axis=1)
max_score = len(valid_cols) * 4
df['Health_Index_%'] = 100 - (df['Risk_Score'] / max_score * 100)

def risk_category(score):
    if score > 50:
        return 'Low Risk'
    elif score > 40:
        return 'Moderate Risk'
    else:
        return 'High Risk'

df['Risk_Category'] = df['Health_Index_%'].apply(risk_category)

# --- ML MODEL ---
X = df_numeric.fillna(0)
y = df['Risk_Category']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = LogisticRegression(multi_class='ovr', max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
df['Predicted_Risk'] = le.inverse_transform(model.predict(X))

# --- SIDEBAR NAVIGATION ---
view = st.sidebar.radio("Choose View", ["📊 School Portfolio", "🧍 Student Portfolio"])

# --- SCHOOL PORTFOLIO ---
if view == "📊 School Portfolio":
    st.subheader("🏫 School Health Overview")

    col1, col2 = st.columns(2)
    with col1:
        avg_health = df['Health_Index_%'].mean()
        st.metric("Average School Health Index", f"{avg_health:.2f}%")
    with col2:
        risk_distribution = df['Risk_Category'].value_counts(normalize=True) * 100
        st.write("Risk Distribution:")
        st.bar_chart(risk_distribution)

    # Risk category pie chart
    st.write("### Risk Category Distribution")
    pie_fig, ax = plt.subplots()
    df['Risk_Category'].value_counts().plot.pie(
        autopct='%1.1f%%', ax=ax,
        colors=['#4CAF50', '#FFC107', '#F44336'],
        startangle=140
    )
    ax.set_ylabel('')
    st.pyplot(pie_fig)

    # Risk category count by grade
    st.write("### Risk Category Count by Grade")
    grade_risk_counts = df.groupby(['Current Grade', 'Risk_Category']).size().unstack(fill_value=0)
    st.bar_chart(grade_risk_counts)

    st.write("### All Students Summary")
    st.dataframe(df[['Student SNO', 'Gender', 'Current Grade', 'Health_Index_%', 'Risk_Category', 'Predicted_Risk']])

    # Heatmap of first 30 students' responses
    st.write("### 🔥 Risk Score Heatmap (First 30 Students)")
    plt.figure(figsize=(12, 4))
    sns.heatmap(df_numeric.head(30), cmap='YlOrRd', cbar=True)
    st.pyplot(plt)

    # Correlation heatmap of survey questions
    st.write("### Question Response Correlation Heatmap")
    corr = df_numeric.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    st.pyplot(plt)

    # Confusion matrix for ML model
    st.write("### Model Performance: Predicted vs Actual Risk Category")
    y_pred = model.predict(X_test)
    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    st.pyplot(fig)

# --- STUDENT PORTFOLIO ---
if view == "🧍 Student Portfolio":
    st.subheader("🔍 Individual Student Review")

    selected_student = st.selectbox("Select Student SNO", df['Student SNO'])
    student_data = df[df['Student SNO'] == selected_student].iloc[0]

    st.write(f"**Gender**: {student_data['Gender']}")
    st.write(f"**Grade**: {student_data['Current Grade']}")
    st.metric("Health Index", f"{student_data['Health_Index_%']:.2f}%")
    st.metric("Risk Category", student_data['Risk_Category'])
    st.metric("Predicted Risk (ML)", student_data['Predicted_Risk'])

    st.write("### Response Breakdown")
    st.dataframe(
        df.loc[df['Student SNO'] == selected_student, valid_cols]
        .T.rename(columns={df.loc[df['Student SNO'] == selected_student].index[0]: 'Response'})
    )

# --- FOOTER ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | © 2025 Mental Health Project")
