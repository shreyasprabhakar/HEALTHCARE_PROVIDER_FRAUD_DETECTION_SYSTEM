# Healthcare Insurance Fraud Detection using Machine Learning

## 📌 Project Overview
This project focuses on detecting fraudulent activities in healthcare insurance claims using machine learning techniques. The system analyzes multiple healthcare-related datasets to identify common fraud patterns such as **upcoding, phantom billing, unbundling, and kickback schemes**. The objective is to assign fraud risk scores to claims and providers, helping insurers prioritize suspicious cases for further investigation.

---

## 🎯 Problem Statement
Healthcare insurance fraud leads to significant financial losses for insurers, higher premiums for policyholders, delayed payments for legitimate providers, and misuse of public funds. Traditional rule-based systems are insufficient to detect evolving fraud patterns. This project demonstrates a **data-driven, ML-based fraud detection framework** that enhances claim verification and risk assessment.

---

## 📊 Data Sources
- Multiple **synthetic healthcare insurance datasets** sourced from **Kaggle**
- Datasets simulate real-world healthcare claim behavior while preserving privacy
- Each dataset captures different aspects such as claims, providers, billing patterns, and referrals

---

## 🧠 Fraud Types Covered
- **Upcoding** – Billing higher-cost services than actually provided  
- **Phantom Billing** – Billing for services that never occurred  
- **Unbundling** – Splitting bundled procedures into multiple charges  
- **Kickback Schemes** – Incentivized referrals between providers  

---

## ⚙️ Methodology
1. Exploratory Data Analysis (EDA) and feature understanding  
2. Fraud mapping across datasets based on domain knowledge  
3. Feature engineering and peer-group behavior analysis  
4. Model training using:
   - Logistic Regression
   - XGBoost
   - Isolation Forest (anomaly detection)
5. Risk score generation and normalization  
6. Model explainability using SHAP  
7. Visualization and dashboard creation  

---

## 📈 Evaluation Metrics
- Precision
- Recall
- ROC-AUC
- Top-K risk analysis  
(Accuracy is avoided due to class imbalance in fraud detection.)

---

## 📊 Visualization & Dashboard
- Fraud trends and risk distributions visualized using **Matplotlib, Seaborn, and Plotly**
- Interactive fraud risk dashboard built using **Streamlit**

---

## 🛠️ Tech Stack
- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Dashboard:** Streamlit  
- **Database:** SQL  
- **Tools:** Jupyter Notebook, VS Code  
- **Version Control:** Git, GitHub  

---

## 🚀 How to Run the Project
Create a virtual environment and activate it.
Then,
```bash
pip install -r requirements.txt
streamlit run app.py
