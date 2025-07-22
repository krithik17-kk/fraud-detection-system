import streamlit as st
import pandas as pd
from src.model import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load trained model and scaler
model = load_model('model.joblib')

st.set_page_config(page_title="Financial Fraud Detector", layout="wide")
st.title("💳 Financial Fraud Detection System")

uploaded_file = st.file_uploader("Upload transaction data (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Basic checks
    if 'Time' not in df.columns or 'Amount' not in df.columns:
        st.error("Missing required columns: 'Time' and 'Amount'")
    else:
        # Preprocess input data
        scaler = StandardScaler()
        df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
        if 'Class' in df.columns:
         df = df.drop('Class', axis=1)
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        df['Fraud_Prediction'] = predictions
        df['Fraud_Probability'] = probabilities

        st.subheader("🔍 Results")
        st.dataframe(df[['Amount', 'Time', 'Fraud_Prediction', 'Fraud_Probability']])

        st.download_button("📥 Download Results", data=df.to_csv(index=False), file_name="fraud_predictions.csv")
 
