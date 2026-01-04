from sklearn.preprocessing import LabelEncoder
import pandas as pd


def prepare_features(df):
    """Encode categorical columns and return feature matrix X and labels y."""
    df_model = df.copy()
    le = LabelEncoder()
    df_model['transaction_type'] = le.fit_transform(df_model['transaction_type'])

    X = df_model[['amount', 'time_of_day', 'transaction_type']]
    y = df_model['is_fraud']
    return X, y
