# End-to-End-Pipeline
# 🔮 Telco Customer Churn Prediction - End-to-End ML Pipeline with Streamlit App

![Churn Prediction](https://img.shields.io/badge/Churn%20Prediction-Streamlit%20App-green)

## 💡 Overview
This project demonstrates an **end-to-end machine learning pipeline** to predict customer churn using the **Telco Customer Churn dataset**. It includes:
- Detailed exploratory data analysis (EDA)
- Data preprocessing (handling missing values, outliers, encoding, scaling)
- Model building with Scikit-learn Pipelines
- Hyperparameter tuning with GridSearchCV
- Model evaluation (confusion matrix, classification report, ROC AUC curve, feature importance)
- Exporting trained pipeline with `joblib`
- Deploying an interactive **Streamlit app** for live predictions

---

## 🚀 Project Structure
📁 project/
├── telco_churn_notebook.ipynb # Jupyter notebook with EDA & modeling
├── telco_churn_app.py # Streamlit app code
├── telco_churn_rf_pipeline.joblib # Saved trained pipeline
├── requirements.txt # Python dependencies
├── README.md # Project overview and instructions

---

## 📊 Dataset
**Telco Customer Churn dataset**
- Source: [Kaggle Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Contains customer demographics, services, and account information
- Target variable: `Churn` (Yes/No)

---

## 🏗️ Features of the ML Pipeline
- **Automated preprocessing**: scaling numeric columns, encoding categoricals
- **Hyperparameter tuning**: Random Forest with GridSearchCV to improve recall for churn class
- **Evaluation metrics**: precision, recall, f1-score, ROC AUC, feature importance
- **Exportable**: model saved using `joblib` for reusability and deployment

---

## 💻 Streamlit App Features
- Upload CSV file with customer data or manually input details
- Predict churn probability and churn decision (Yes/No)
- User-friendly interface with explanation and guidance
- Ready for business or operational teams to use

---

## ⚙️ Setup Instructions
### 1️⃣ Clone the repository
```bash
git clone https://github.com/Mohammed-Umair30/End-to-End-Pipeline.git
cd telco-churn-pipeline

