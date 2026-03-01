**PM Internship Eligibility & Recommendation System**

An AI-powered Internship Eligibility and Recommendation System developed as part of Smart India Hackathon (SIH).
This system evaluates student eligibility using Machine Learning and recommends the most suitable internships using similarity-based ranking techniques.

 **Problem Statement**

Many students struggle to identify suitable internships based on eligibility criteria, skills, and location preferences.
This project provides an automated system that:

Determines eligibility using defined rules and ML models

Recommends internships based on skill matching

Ensures fairness through bias auditing

** Key Features**
 1. Eligibility Prediction

Rule-based filtering (Age, Income, Education, Employment Status)

Machine Learning model (Random Forest Classifier)

Accuracy evaluation with confusion matrix

 2. Internship Recommendation

Skill-based similarity matching

Cosine similarity scoring

TF-IDF profile vectorization

Weighted ranking (Skills + Education + Location)

Top-K internship recommendations

 3. Bias & Fairness Audit

Demographic parity analysis

Eligibility distribution by college tier

Automatic bias detection mechanism

 Tech Stack

Python

Streamlit

Pandas & NumPy

Scikit-learn

TF-IDF Vectorizer

Random Forest

Truncated SVD

Matplotlib

 Machine Learning Workflow

Data Preprocessing

Feature Engineering

MultiLabel Skill Encoding

Train-Test Split

Random Forest Model Training

Model Evaluation (Accuracy, Classification Report)

Cosine Similarity-Based Ranking

Fairness Evaluation

 Project Structure
├── SIH3.py
├── students_300_samples.csv
├── internships_300_samples.csv
└── README.md
How to Run
pip install -r requirements.txt
streamlit run SIH3.py
 Learning Outcomes

Built an end-to-end ML pipeline

Implemented a recommendation system

Applied NLP-based similarity matching

Performed fairness and bias auditing

Developed an interactive ML web app
