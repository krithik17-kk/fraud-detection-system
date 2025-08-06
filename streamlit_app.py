import streamlit as st
import pandas as pd
import joblib
import os

from src import preprocessing, model



st.set_page_config(page_title="Financial Fraud Detection System", layout="wide", initial_sidebar_state="expanded")


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/bank-card-back-side.png", width=80)
    st.markdown("""
    ## Financial Fraud Detection
    
    Upload your credit card transaction data (.csv) to detect fraudulent transactions in bulk.
    
    **Instructions:**
    - The CSV should contain columns like `Time`, `Amount`, `V1`-`V28`.
    - Download the results with predictions and fraud probabilities.
    
    ---
    **Made with ❤️ using Streamlit**
    """)

# --- Main Area ---
st.markdown("""
<div style='display: flex; align-items: center; justify-content: center;'>
    <img src="https://img.icons8.com/color/96/000000/bank-card-back-side.png" width="60" style="margin-right: 16px;"/>
    <h1 style='margin-bottom: 0;'>Financial Fraud Detection System</h1>
</div>
""", unsafe_allow_html=True)


# --- File Uploader Section ---
st.markdown("""
<div style='background-color: #22232b; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
    <h4 style='color: #fff;'>Upload transaction data (.csv)</h4>
    <p style='color: #bbb;'>Drag and drop file here (Limit 200MB per file • CSV)</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["csv"],
    help="Limit 200MB per file • CSV"
)

if uploaded_file is not None:
    st.markdown(f"<span style='color:#00bfff;font-size:1.1rem;'>📁 {uploaded_file.name} ({round(uploaded_file.size/1024/1024,1)}MB)</span>", unsafe_allow_html=True)

# Load model
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    return joblib.load(model_path)

clf = load_model()


# --- Results Section ---
st.markdown("""
<br>
<h2 style='display: flex; align-items: center; color: #00bfff;'><span style='font-size:2rem;margin-right:10px;'>🔍</span>Results</h2>
<hr style='border: 1px solid #22232b; margin-top: -10px; margin-bottom: 20px;'>
""", unsafe_allow_html=True)

# Batch prediction only
if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    # Drop 'Class' column if present
    if 'Class' in batch_df.columns:
        batch_df = batch_df.drop('Class', axis=1)
    # batch_df = preprocessing.preprocess(batch_df)  # Uncomment if you have a preprocess function
    batch_pred = clf.predict(batch_df)
    batch_pred_proba = clf.predict_proba(batch_df)
    batch_df['Fraud_Prediction'] = batch_pred
    batch_df['Fraud_Probability'] = batch_pred_proba[:,1]
    # Show only Amount, Time, Fraud_Prediction, Fraud_Probability columns in that order
    display_cols = []
    if 'Amount' in batch_df.columns: display_cols.append('Amount')
    if 'Time' in batch_df.columns: display_cols.append('Time')
    display_cols += ['Fraud_Prediction', 'Fraud_Probability']
    st.dataframe(batch_df[display_cols], use_container_width=True, height=400)
    st.download_button("Download Results", batch_df[display_cols].to_csv(index=False), file_name="predictions.csv")

# --- Footer ---
st.markdown("""
<hr style='border: 1px solid #22232b; margin-top: 40px;'>
<div style='text-align:center; color:#888; font-size:0.95rem;'>
    &copy; 2025 Financial Fraud Detection System &mdash; Powered by Streamlit
</div>
""", unsafe_allow_html=True)
