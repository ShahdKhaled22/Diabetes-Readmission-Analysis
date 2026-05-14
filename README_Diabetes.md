# 🏥 Diabetes Patient Readmission Analysis

> Exploratory Data Analysis on 100,000+ diabetic patient records to identify key factors influencing hospital readmission rates.

---

## 📌 Project Overview

Hospital readmission is a critical challenge in modern healthcare — it increases costs, strains resources, and often signals gaps in post-discharge care. This project analyzes the **UCI Diabetes 130-US Hospitals dataset (1999–2008)** to uncover patterns and risk factors associated with patient readmission among diabetic patients.

The insights from this analysis can directly inform **Digital Health system design**, helping hospitals build smarter early-warning tools and personalized discharge planning.

---

## 🎯 Key Questions Answered

| # | Question | Finding |
|---|----------|---------|
| 1 | What is the overall distribution of readmissions? | Dataset is imbalanced — majority NOT readmitted |
| 2 | Does hospital stay increase with patient age? | ✅ Older patients spend significantly more days |
| 3 | Which diagnoses have highest readmission rates? | Circulatory & Respiratory conditions |
| 4 | Do lab procedures predict readmission? | ✅ Readmitted patients have more procedures & medications |
| 5 | What features correlate most with readmission? | `had_inpatient` & `num_medications` are strongest predictors |
| 6 | Does medication change affect readmission? | ✅ Medication changes linked to higher readmission rates |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| Dataset | Diabetes 130-US Hospitals (1999–2008) |
| Records | ~101,000 patient encounters |
| Features | 50 clinical attributes |
| Target | Readmission status (`<30`, `>30`, `NO`) |

---

## 🛠️ Tools & Technologies

```
Python 3.12
├── pandas        — Data manipulation
├── numpy         — Numerical computing
├── matplotlib    — Data visualization
└── seaborn       — Statistical plotting
```

---

## 📈 Key Findings

**1. Readmission Imbalance**
The majority of patients were not readmitted, highlighting the class imbalance challenge common in healthcare datasets.

**2. Age & Hospital Stay**
A clear positive correlation exists between patient age and length of hospital stay — older patients require more resources and extended care.

**3. High-Risk Diagnoses**
Circulatory and Respiratory conditions show the highest readmission counts, suggesting these patients need enhanced post-discharge monitoring.

**4. Strongest Predictors**
- `had_inpatient` (prior inpatient visits) — strongest single predictor
- `num_medications` — patients on more medications are more likely to be readmitted
- `num_lab_procedures` — reflects medical complexity linked to readmission risk

**5. Medication Changes Matter**
Patients whose medication dosage was changed during their stay had a measurably higher readmission probability — suggesting medication stability as a key discharge factor.

---

## 💡 Healthcare Implications

These findings have direct applications in **Digital Health system design**:

- 🔔 **Early Warning Systems** — flag high-risk patients before discharge
- 📋 **Personalized Care Plans** — tailor post-discharge follow-up based on medication complexity
- 📊 **Resource Allocation** — prioritize monitoring for circulatory/respiratory patients
- 🏥 **EHR Integration** — embed readmission risk scores into Electronic Health Records

---

## 🗂️ Repository Structure

```
Diabetes-Readmission-Analysis/
│
├── Diabetes_Analysis.ipynb    # Main analysis notebook
├── README.md                  # Project documentation
└── requirements.txt           # Dependencies
```

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/ShahdKhaled22/Diabetes-Readmission-Analysis.git

# Navigate to directory
cd Diabetes-Readmission-Analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn

# Open notebook
jupyter notebook Diabetes_Analysis.ipynb
```

---

## 👩‍💻 Author

**Shahd Khaled**
B.Sc. Computer Science — The Open University, UK
Data Science & Machine Learning Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-ShahdKhaled22-black?style=flat&logo=github)](https://github.com/ShahdKhaled22)

---

## 🔗 Related Projects

- [Heart Disease Prediction System](https://github.com/ShahdKhaled22/Heart-Disease-Prediction) — ML classification using Random Forest, Logistic Regression & Gradient Boosting
