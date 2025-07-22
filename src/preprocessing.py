import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def preprocess_data(X_train, X_test):
    """
    Standardize the 'Amount' and 'Time' features and apply SMOTE to training data.
    """
    # Scale 'Amount' and 'Time'
    scaler = StandardScaler()
    X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
    X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])

    return X_train, X_test, scaler

def apply_smote(X_train, y_train):
    """
    Handle class imbalance using SMOTE on training set only.
    """
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    return X_resampled, y_resampled 
