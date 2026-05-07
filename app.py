import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Analysis",
    page_icon="🏥",
    layout="wide"
)

# ── Load Data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('diabetic_data.csv', na_values='?')

    # Drop high-null columns
    df.drop(columns=['weight', 'payer_code', 'medical_specialty'], inplace=True)

    # Fill missing lab test values with 'None'
    df['max_glu_serum'] = df['max_glu_serum'].fillna('None')
    df['A1Cresult']     = df['A1Cresult'].fillna('None')

    # Drop rows with few missing values
    df.dropna(subset=['diag_1', 'diag_2', 'diag_3', 'race'], inplace=True)

    # Remove invalid gender
    df = df[df['gender'] != 'Unknown/Invalid']

    # Keep first visit per patient only
    df.drop_duplicates(subset=['patient_nbr'], keep='first', inplace=True)

    # Convert age to numeric midpoints
    age_map = {
        '[0-10)': 5,  '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df['age_numeric'] = df['age'].map(age_map)

    # Binary visit flags
    df['had_emergency']  = (df['number_emergency']  > 0).astype(int)
    df['had_outpatient'] = (df['number_outpatient'] > 0).astype(int)
    df['had_inpatient']  = (df['number_inpatient']  > 0).astype(int)

    # Binary target variable
    df['readmitted_binary'] = df['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)

    # Map diagnosis codes
    def map_diagnosis(code):
        try:
            c = str(code).strip().upper()
            if c.startswith(('V', 'E')) or c in ('?', 'OTHER', 'NAN'):
                return 'Other'
            val = float(c)
            if 390 <= val <= 459 or val == 785:   return 'Circulatory'
            elif 460 <= val <= 519 or val == 786: return 'Respiratory'
            elif 520 <= val <= 579 or val == 787: return 'Digestive'
            elif 250.0 <= val < 251.0:            return 'Diabetes'
            elif 800 <= val <= 999:               return 'Injury'
            elif 710 <= val <= 739:               return 'Musculoskeletal'
            elif 580 <= val <= 629 or val == 788: return 'Genitourinary'
            elif 140 <= val <= 239:               return 'Neoplasms'
            else:                                 return 'Other'
        except:
            return 'Other'

    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].apply(map_diagnosis)

    return df

df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=80)
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🧹 Data Cleaning",
    "📈 Univariate Analysis",
    "🔗 Bivariate Analysis",
    "📋 Summary"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Patients:** {len(df):,}")
st.sidebar.markdown(f"**Features:** {df.shape[1]}")

# ── Page 1: Overview ───────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("🏥 Diabetes Patient Readmission Analysis")
    st.markdown("**Dataset:** UCI Diabetes 130-US Hospitals (1999–2008)")
    st.markdown("---")

    st.subheader("🔑 Key Research Questions")
    questions = [
        "What is the overall distribution of hospital readmissions?",
        "Does the average time in hospital increase with patient age?",
        "Which diagnosis categories have the highest readmission rates?",
        "How do lab procedures and medications relate to readmission?",
        "What are the most influential features correlated with readmission?",
        "Does changing medication dosage affect readmission probability?"
    ]
    for i, q in enumerate(questions, 1):
        st.markdown(f"**Q{i}:** {q}")

    st.markdown("---")
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head(10))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", f"{len(df):,}")
    col2.metric("Readmitted", f"{df['readmitted_binary'].sum():,}")
    col3.metric("Not Readmitted", f"{(df['readmitted_binary'] == 0).sum():,}")

# ── Page 2: Data Cleaning ──────────────────────────────────────────────────
elif page == "🧹 Data Cleaning":
    st.title("🧹 Data Cleaning Documentation")
    st.markdown("---")

    cleaning_steps = {
        "1. Dropped high-null columns": "weight (~97% missing), payer_code (~40%), medical_specialty (~49%)",
        "2. Filled missing lab values": "max_glu_serum and A1Cresult filled with 'None' (test not done)",
        "3. Dropped sparse rows": "Rows missing diag_1/2/3 or race were removed",
        "4. Removed invalid gender": "Rows with 'Unknown/Invalid' gender removed",
        "5. Removed duplicate patients": "Kept only first visit per patient",
        "6. Converted age to numeric": "Age ranges mapped to midpoint values (e.g. [70-80) → 75)",
        "7. Engineered visit flags": "Created binary flags: had_emergency, had_outpatient, had_inpatient",
        "8. Created binary target": "readmitted → 0 (No) or 1 (Yes)",
        "9. Mapped diagnosis codes": "ICD-9 codes mapped to clinical categories (Circulatory, Diabetes, etc.)"
    }

    for step, detail in cleaning_steps.items():
        with st.expander(step):
            st.write(detail)

    st.markdown("---")
    st.subheader("✅ Final Dataset Info")
    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]:,}")
    st.write("**Missing values remaining:**", df.isnull().sum().sum())

# ── Page 3: Univariate Analysis ────────────────────────────────────────────
elif page == "📈 Univariate Analysis":
    st.title("📈 Univariate Analysis")
    st.markdown("Examining the distribution of each variable independently.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Readmission Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='readmitted_binary', data=df, palette='viridis', ax=ax)
        ax.set_xlabel("Readmission (0=No, 1=Yes)")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots()
        gender_counts = df['gender'].value_counts()
        ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
               colors=sns.color_palette('pastel'), startangle=90)
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Days in Hospital")
        fig, ax = plt.subplots()
        sns.histplot(df['time_in_hospital'], bins=14, kde=True, color='teal', ax=ax)
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Number of Medications")
        fig, ax = plt.subplots()
        sns.histplot(df['num_medications'], bins=30, kde=True, color='coral', ax=ax)
        st.pyplot(fig)
        plt.close()

    st.subheader("Outlier Detection — Box Plots")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color in zip(axes,
        ['time_in_hospital', 'num_medications', 'num_lab_procedures'],
        ['teal', 'coral', 'steelblue']):
        sns.boxplot(x=df[col], ax=ax, color=color)
        ax.set_title(col)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Page 4: Bivariate Analysis ─────────────────────────────────────────────
elif page == "🔗 Bivariate Analysis":
    st.title("🔗 Bivariate & Multivariate Analysis")
    st.markdown("---")

    st.subheader("Q2: Average Hospital Stay by Age Group")
    age_trend = df.groupby('age_numeric')['time_in_hospital'].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(age_trend.index, age_trend.values, marker='o', color='red', linewidth=2)
    ax.set_xlabel("Age (Numeric Midpoint)")
    ax.set_ylabel("Avg Days in Hospital")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    plt.close()

    st.subheader("Q3: Readmission by Primary Diagnosis")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(y='diag_1', hue='readmitted_binary', data=df,
                  order=df['diag_1'].value_counts().index, palette='magma', ax=ax)
    ax.legend(title='Readmitted', labels=['No', 'Yes'])
    st.pyplot(fig)
    plt.close()

    st.subheader("Q5: Correlation Heatmap")
    numerical_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                      'num_medications', 'number_diagnoses', 'age_numeric',
                      'had_emergency', 'had_outpatient', 'had_inpatient', 'readmitted_binary']
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df[numerical_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    plt.close()

    st.subheader("Q6: Readmission Rate by Medication Change")
    change_readmit = df.groupby('change')['readmitted_binary'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='change', y='readmitted_binary', data=change_readmit, palette='coolwarm', ax=ax)
    ax.set_ylabel("Readmission Rate")
    st.pyplot(fig)
    plt.close()

# ── Page 5: Summary ────────────────────────────────────────────────────────
elif page == "📋 Summary":
    st.title("📋 Summary of Findings")
    st.markdown("---")

    findings = {
        "Q1: Readmission Distribution": "Dataset is imbalanced — majority of patients were NOT readmitted",
        "Q2: Age vs Hospital Stay": "Older patients tend to spend more days in the hospital",
        "Q3: Diagnosis & Readmission": "Circulatory and Respiratory conditions show highest readmission counts",
        "Q4: Lab Procedures & Meds": "Readmitted patients tend to have more lab procedures and medications",
        "Q5: Feature Correlations": "had_inpatient and num_medications are the strongest predictors",
        "Q6: Medication Change": "Patients whose medications were changed have a higher readmission rate"
    }

    for question, finding in findings.items():
        st.success(f"**{question}:** {finding}")

    st.markdown("---")
    st.subheader("🛠️ Technologies Used")
    st.markdown("Python | Pandas | NumPy | Matplotlib | Seaborn | Streamlit")
