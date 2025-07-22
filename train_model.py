import pandas as pd
from src.load_data import load_raw_data, split_data
from src.preprocessing import preprocess_data, apply_smote
from src.model import train_model, evaluate_model, save_model
from src.evaluation import plot_roc_curve

def main():
    # 1. Load Data
    df = load_raw_data('data/raw/creditcard.csv')
    print(f"Dataset loaded with shape: {df.shape}")

    # 2. Split Data
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Preprocess: Scaling
    X_train, X_test, scaler = preprocess_data(X_train, X_test)

    # 4. Handle Imbalance: SMOTE
    X_resampled, y_resampled = apply_smote(X_train, y_train)

    # 5. Train Model
    model = train_model(X_resampled, y_resampled, model_type='random_forest')

    # 6. Evaluate
    roc_auc = evaluate_model(model, X_test, y_test)
    plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    # 7. Save Model
    save_model(model, path='model.joblib')
    print("✅ Model training complete and saved as 'model.joblib'.")

if __name__ == "__main__":
    main()

