import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_artifacts():
    """Load all preprocessing artifacts"""
    scaler = joblib.load('models/scaler.pkl')
    expected_columns = joblib.load('models/expected_columns.pkl')
    median_pdays = joblib.load('models/median_pdays.pkl')
    return scaler, expected_columns, median_pdays

scaler, expected_columns, median_pdays = load_artifacts()

def preprocess_input(input_data):
    # Create DataFrame from input
    df = pd.DataFrame([input_data])

    # Encoding (same as before)
    nominal_features = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    ordinal_features = {'default': {'no': 0, 'yes': 1},
                        'housing': {'no': 0, 'yes': 1},
                        'loan': {'no': 0, 'yes': 1}}

    # One-hot encoding
    for feature in nominal_features:
        dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df.drop(feature, axis=1, inplace=True)

    # Ordinal encoding
    for feature, mapping in ordinal_features.items():
        df[feature] = df[feature].map(mapping)

    # Numerical features
    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    # Special handling for pdays
    df['pdays'] = df['pdays'].replace(-1, np.nan)
    df['pdays'] = df['pdays'].fillna(median_pdays)

    # Scale numerical features
    df[numerical_features] = scaler.transform(df[numerical_features])

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    return df[expected_columns]