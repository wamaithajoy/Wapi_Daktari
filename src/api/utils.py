# src/api/utils.py

import pandas as pd
import joblib

# Load the preprocessor and label encoder
preprocessor = joblib.load('src/api/preprocessor.pkl')
label_encoder = joblib.load('src/api/label_encoder.pkl')

def preprocess_features(features):
    # Convert features to DataFrame
    df = pd.DataFrame([features])

    # Preprocess the features
    df_processed = preprocessor.transform(df)

    return df_processed