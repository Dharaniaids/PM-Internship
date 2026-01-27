# ================== STREAMLIT INTERNSHIP RECOMMENDER ==================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cosine

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="PM Internship Recommender",
    layout="wide"
)

st.title("🎓 PM Internship Eligibility & Recommendation System")

# ------------------ CONFIG ------------------
STUDENTS_CSV = r"students_300_samples.csv"
INTERNSHIPS_CSV = r"internships_300_samples.csv"
RANDOM_STATE = 42
TOP_K = 5

# ------------------ ELIGIBILITY RULES ------------------
ALLOWED_EDUCATION = {
    "Class 10", "Class 12", "ITI", "Diploma",
    "BA", "B.Sc", "B.Com", "BBA", "BCA", "B.Pharma"
}

def compute_eligibility(df):
    def is_eligible(row):
        try:
            age = int(row["Age"])
            income = float(row["Family_Income_LPA"])
        except:
            return "No"

        if not (21 <= age <= 24): return "No"
        if row["Job_Status"] == "Employed Full-Time": return "No"
        if row["Enrolled"] == "Yes": return "No"
        if income >= 8.0: return "No"
        if row["Govt_Job"] == "Yes": return "No"
        if row["Education"] not in ALLOWED_EDUCATION: return "No"

        return "Yes"

    return df.apply(is_eligible, axis=1)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    students = pd.read_csv(STUDENTS_CSV, encoding="latin1")
    internships = pd.read_csv(INTERNSHIPS_CSV)
    return students, internships

students, internships = load_data()

students["Eligibility_Status"] = compute_eligibility(students)

st.subheader("📊 Student Dataset Preview")
st.dataframe(students.head())

# ------------------ ML PREPARATION ------------------
students_ml = students.copy()

students_ml["Skills"] = students_ml["Skills"].fillna("").apply(
    lambda x: [s.strip() for s in x.split(",") if s.strip()]
)

mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(students_ml["Skills"])
skills_df = pd.DataFrame(skills_encoded, columns=[f"skill_{s}" for s in mlb.classes_])

num_cols = ["Age", "Family_Income_LPA", "CGPA"]
cat_cols = ["Education", "Job_Status", "Enrolled", "Govt_Job",
            "College_Tier", "Preferred_Location"]

students_ml[num_cols] = students_ml[num_cols].fillna(students_ml[num_cols].median())

X = pd.concat(
    [students_ml[num_cols + cat_cols].reset_index(drop=True),
     skills_df.reset_index(drop=True)],
    axis=1
)

y = (students_ml["Eligibility_Status"] == "Yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

clf = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ))
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ------------------ MODEL RESULTS ------------------
st.subheader("🤖 Eligibility Model Performance")

st.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig)

# ------------------ TF-IDF PROFILES ------------------
def build_student_text(row):
    return f"{row['Education']} | {row['Skills']} | {row['Preferred_Location']} | {row['College_Tier']} | CGPA {row['CGPA']}"

def build_intern_text(row):
    return f"{row['Title']} | {row['Domain']} | {row['Skills_Required']} | {row['Location']}"

internships["text_profile"] = internships.apply(build_intern_text, axis=1)

eligible_students = students[students["Eligibility_Status"] == "Yes"].copy()
eligible_students["text_profile"] = eligible_students.apply(build_student_text, axis=1)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
vectorizer.fit(pd.concat([internships["text_profile"], eligible_students["text_profile"]]))

internship_matrix = vectorizer.transform(internships["text_profile"])
student_matrix = vectorizer.transform(eligible_students["text_profile"])

svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
internship_emb = svd.fit_transform(internship_matrix)
student_emb = svd.transform(student_matrix)

# ------------------ RECOMMENDATION ------------------
all_skills = list(mlb.classes_)

def rank_internships(student, internships_df, top_k=5):
    student_skills = student["Skills"]
    scores = []

    for _, row in internships_df.iterrows():
        intern_skills = row["Skills_Required"].split(",") if pd.notna(row["Skills_Required"]) else []

        sv = np.array([1 if s in student_skills else 0 for s in all_skills])
        iv = np.array([1 if s in intern_skills else 0 for s in all_skills])

        sim = 1 - cosine(sv, iv) if np.any(sv) and np.any(iv) else 0
        edu_match = student["Education"] in str(row.get("Eligibility", ""))
        loc_match = student["Preferred_Location"].lower() == row["Location"].lower()

        score = 0.7*sim + 0.15*edu_match + 0.15*loc_match

        scores.append({
            "Title": row["Title"],
            "Company": row["Company_Name"],
            "Location": row["Location"],
            "Score": round(score, 3)
        })

    return sorted(scores, key=lambda x: x["Score"], reverse=True)[:top_k]

# ------------------ UI: STUDENT SELECTION ------------------
st.subheader("🎯 Internship Recommendations")

student_name = st.selectbox(
    "Select Eligible Student",
    eligible_students["Name"].tolist()
)

selected_student = eligible_students[
    eligible_students["Name"] == student_name
].iloc[0]

recs = rank_internships(selected_student, internships)

st.table(pd.DataFrame(recs))

# ------------------ BIAS & FAIRNESS ------------------
st.subheader("⚖️ Bias & Fairness Audit")

tier_bias = (
    students.groupby("College_Tier")["Eligibility_Status"]
    .value_counts(normalize=True)
    .unstack()
    .fillna(0)
)

st.write("Eligibility Distribution by College Tier")
st.dataframe(tier_bias)

parity = students.groupby("College_Tier")["Eligibility_Status"] \
    .apply(lambda x: (x == "Yes").mean())

fig2, ax2 = plt.subplots()
parity.plot(kind="bar", ax=ax2)
ax2.set_title("Demographic Parity")
ax2.set_ylabel("Eligibility Rate")
st.pyplot(fig2)

threshold = 0.25
max_diff = parity.max() - parity.min()

if max_diff > threshold:
    st.warning(f"⚠️ Potential bias detected (difference = {round(max_diff,2)})")
else:
    st.success("✅ No significant bias detected")
