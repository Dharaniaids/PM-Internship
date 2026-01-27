
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine

# ---------------- CONFIG ----------------
STUDENTS_CSV = r"students_300_samples.csv"
INTERNSHIPS_CSV = r"internships_300_samples.csv"



# ------------------ Constants ------------------
ALLOWED_EDUCATION = {
    "Class 10", "Class 12", "ITI", "Diploma",
    "BA", "B.Sc", "B.Com", "BBA", "BCA", "B.Pharma"
}


TOP_K = 5

# ------------------ Eligibility Logic ------------------
def compute_eligibility(df):
    def is_eligible(row):
        try:
            age = int(row.get("Age", np.nan))
            job = row.get("Job_Status", "")
            enrolled = row.get("Enrolled", "")
            income = float(row.get("Family_Income (LPA)", 1e9))
            govt = row.get("Govt_Job", "")
            edu = row.get("Education", "")
        except Exception:
            return "No"

        if not (21 <= age <= 24): return "No"
        if job == "Employed Full-Time": return "No"
        if enrolled == "Yes": return "No"
        if income >= 8.0: return "No"
        if govt == "Yes": return "No"
        if edu not in ALLOWED_EDUCATION: return "No"

        return "Yes"

    return df.apply(is_eligible, axis=1)

# ------------------ Internship Ranking ------------------
def rank_internships(student, internships_df, top_k=5):
    student_skills = [s.strip().lower() for s in student['Skills']]
    student_edu = student['Education']
    student_loc = student['Preferred_Location']

    all_skills = set()
    for skills in internships_df['Skills_Required'].dropna():
        all_skills.update([s.strip().lower() for s in skills.split(",")])
    all_skills = sorted(list(all_skills))

    scores = []
    for _, row in internships_df.iterrows():
        internship_skills = [
            s.strip().lower()
            for s in (row['Skills_Required'].split(",")
                      if pd.notna(row['Skills_Required']) else [])
        ]

        skill_vec_student = np.array([1 if s in student_skills else 0 for s in all_skills])
        skill_vec_intern = np.array([1 if s in internship_skills else 0 for s in all_skills])

        cosine_sim = (
            1 - cosine(skill_vec_student, skill_vec_intern)
            if np.any(skill_vec_student) and np.any(skill_vec_intern)
            else 0.0
        )

        matched_skills = list(set(student_skills) & set(internship_skills))
        edu_match = student_edu in str(row.get("Eligibility", ""))
        loc_match = student_loc.lower() == str(row['Location']).lower()

        final_score = (0.70 * cosine_sim) + (0.15 * int(edu_match)) + (0.15 * int(loc_match))

        scores.append({
            "Title": row.get("Title", ""),
            "Company": row.get("Company_Name", ""),
            "Location": row.get("Location", ""),
            "CosineSimilarity": round(cosine_sim, 3),
            "MatchedSkills": matched_skills,
            "EducationMatch": edu_match,
            "LocationMatch": loc_match,
            "FinalScore": round(final_score, 3)
        })

    ranked = sorted(scores, key=lambda x: x['FinalScore'], reverse=True)
    return ranked[:top_k]

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Internship Recommendation System", layout="centered")

st.title("🎓 Internship Eligibility & Recommendation System")

# Load data
internships = pd.read_csv(INTERNSHIPS_CSV)

st.subheader("👤 Enter Student Details")

age = st.number_input("Age", min_value=15, max_value=40, step=1)
edu = st.selectbox("Education", sorted(ALLOWED_EDUCATION))
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)
income = st.number_input("Family Income (LPA)", min_value=0.0, step=0.1)

job = st.selectbox("Job Status", ["Unemployed", "Employed Full-Time"])
enrolled = st.selectbox("Currently Enrolled in other course?", ["No", "Yes"])
govt = st.selectbox("Any Govt Job holder in family?", ["No", "Yes"])
tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"])
location = st.text_input("Preferred Location")
skills = st.text_input("Skills (comma separated)")

if st.button("🔍 Check Eligibility & Recommend"):
    student = {
        "Age": age,
        "Education": edu,
        "CGPA": cgpa,
        "Family_Income (LPA)": income,
        "Job_Status": job,
        "Enrolled": enrolled,
        "Govt_Job": govt,
        "College_Tier": tier,
        "Preferred_Location": location,
        "Skills": [s.strip() for s in skills.split(",") if s.strip()]
    }

    df = pd.DataFrame([student])
    df["Eligibility_Status"] = compute_eligibility(df)

    if df["Eligibility_Status"].iloc[0] == "Yes":
        st.success("✅ You are ELIGIBLE for internships")

        recommendations = rank_internships(student, internships, TOP_K)

        st.subheader("🎯 Top Internship Recommendations")
        for rec in recommendations:
            st.markdown(f"""
            **{rec['Title']}**  
            🏢 {rec['Company']}  
            📍 {rec['Location']}  
            ⭐ Score: `{rec['FinalScore']}`  
            🛠 Skills Match: `{', '.join(rec['MatchedSkills'])}`
            ---
            """)
    else:
        st.error("❌ You are NOT eligible for internships based on the criteria.")
