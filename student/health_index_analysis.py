import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file with correct encoding
df = pd.read_csv('Student_Survey_Responses_300.csv', encoding='windows-1252')

# STEP 1: Define scoring scale
response_scale = {
    'Never': 0,
    'Rarely': 1,
    'Sometimes': 2,
    'Often': 3,
    'Always': 4,
    'Skip': np.nan
}

# STEP 2: Define non-question (meta) columns
non_question_cols = ['Student SNO', 'Gender', 'Current Grade']

# STEP 3: Identify valid response columns (only use scale-based responses)
standard_responses = set(response_scale.keys())
valid_response_columns = []

for col in df.columns:
    if col not in non_question_cols:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset(standard_responses):
            valid_response_columns.append(col)

print(f"✅ Found {len(valid_response_columns)} valid survey question columns for analysis.")

# STEP 4: Convert responses to numeric scores
df_numeric = df[valid_response_columns].replace(response_scale)
df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')  # Ensure numeric conversion

# STEP 5: Calculate individual risk score
df['Risk_Score'] = df_numeric.sum(axis=1, skipna=True)

# STEP 6: Normalize to health index %
max_possible_score = len(valid_response_columns) * 4  # Max score if all answers are "Always"
df['Health_Index_%'] = 100 - (df['Risk_Score'] / max_possible_score * 100)

# STEP 7: Categorize into risk levels
def categorize_risk(health_index):
    if health_index > 50:
        return 'Low Risk'
    elif health_index > 25:
        return 'Moderate Risk'
    else:
        return 'High Risk'

df['Risk_Category'] = df['Health_Index_%'].apply(categorize_risk)

# STEP 8: Print and save category summary
category_summary = df['Risk_Category'].value_counts(normalize=True) * 100
print("\n📊 Category-wise Health Risk Percentage:")
print(category_summary.round(2))

# STEP 9: Compute overall school health index
overall_index = df['Health_Index_%'].mean()
print(f"\n🏫 Overall School Health Index: {overall_index:.2f}%")

# STEP 10: Export full report
df.to_csv('Student_Health_Index_Report.csv', index=False)
print("\n📁 Report saved as: Student_Health_Index_Report.csv")

# STEP 11: Plot risk category distribution
sns.set(style='whitegrid')
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Risk_Category', order=['Low Risk', 'Moderate Risk', 'High Risk'], palette='Set2')
plt.title("Student Health Risk Distribution")
plt.xlabel("Risk Category")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("health_risk_distribution.png")
plt.show()
print("📈 Chart saved as: health_risk_distribution.png")
