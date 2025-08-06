# 💳 Fraud Detection System

An end-to-end machine learning project to detect fraudulent credit card transactions using real-world data. The system includes data preprocessing, model training, evaluation, and a Streamlit-powered interactive dashboard for predictions.

---

## 📌 Project Overview

Credit card fraud poses a significant risk to financial institutions and customers. This project builds a classification system using machine learning to identify fraudulent transactions from a highly imbalanced dataset.

---

## 📂 Project Structure

fraud_detection_project/
├── data/ # Folder for raw and processed data
├── dashboard/ # Streamlit web app
│ └── app.py
├── src/ # Source code for preprocessing & modeling
│ ├── data_preprocessing.py
│ └── model.py
├── train_model.py # Script to train & evaluate the model
├── model.joblib # Saved model after training
├── requirements.txt # Required Python packages
└── README.md # Project documentation

markdown
Copy
Edit

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Samples**: 284,807 transactions
- **Features**: 30 anonymized numerical features + `Class` label (0: genuine, 1: fraud)
- **Imbalance**: Only ~0.17% of transactions are fraudulent

---

## 🧠 ML Model

- **Algorithms Used**: Random Forest Classifier
- **Preprocessing**:
  - Feature scaling with `StandardScaler`
  - Train-test split (80-20)
- **Evaluation Metrics**:
  - Confusion Matrix
  - Classification Report
  - ROC-AUC Score

**Sample Result:**

Accuracy: 99.93%
Precision (Fraud): 0.85
Recall (Fraud): 0.84
ROC-AUC Score: 0.973

yaml
Copy
Edit

---

## 🖥️ Streamlit Dashboard

The project includes an interactive frontend using [Streamlit](https://streamlit.io) to:

- Upload a CSV file of new transactions
- Predict fraud/genuine in real-time
- Show prediction results clearly


🚀 Future Enhancements
Add data visualizations (e.g., fraud heatmap, time series trends)

Experiment with advanced models (XGBoost, Isolation Forest)

Deploy on Streamlit Cloud

Add model retraining pipeline

Send fraud alerts via email/SMS 

<img width="625" height="492" alt="Screenshot 2025-07-22 200816" src="https://github.com/user-attachments/assets/797c383b-0b80-480b-b2ad-fff2e1ea5ca5" />
<img width="1247" height="437" alt="Screenshot 2025-07-22 201516" src="https://github.com/user-attachments/assets/1dddafa6-5d44-4a5d-a1db-4502b9efcdf8" />
<img width="1269" height="665" alt="Screenshot 2025-07-22 202244" src="https://github.com/user-attachments/assets/ea93e83a-d90b-44d4-a92a-56a608dfeb89" />

