"""
classical_lin_reg.py

This module uses Classical Linear Regression models with polynomial features to fit curves for different diseases and saves the models to disk.
"""

import pickle
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """
    Load the dataset from a parquet file.

    Parameters:
    filepath (str): Path to the parquet file.

    Returns:
    pd.DataFrame: Loaded dataset.
    """
    try:
        return pd.read_parquet(filepath)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def split_data(df, cutoff_week):
    """
    Split the dataset into training and test sets based on the cutoff week.

    Parameters:
    df (pd.DataFrame): The dataset to split.
    cutoff_week (str): The week to use as the cutoff for splitting the data.

    Returns:
    tuple: (train_df, test_df)
    """
    train = df[df["epi_week"] <= cutoff_week]
    test = df[df["epi_week"] > cutoff_week]
    return train, test

def train_and_save_model(train_df, test_df, model_filename, degree):
    """
    Train a Polynomial Linear Regression model and save it to a file.

    Parameters:
    train_df (pd.DataFrame): Training dataset.
    test_df (pd.DataFrame): Test dataset.
    model_filename (str): Path to save the trained model.
    degree (int): Degree of the polynomial features.
    """
    try:
        # Prepare data
        X_train = np.array(train_df.index).reshape(-1, 1)
        y_train = train_df["no._of_cases"]
        X_test = np.array(test_df.index).reshape(-1, 1)
        y_test = test_df["no._of_cases"]

        # Polynomial feature transformation
        pr = PolynomialFeatures(degree=degree)
        X_train_poly = pr.fit_transform(X_train)
        X_test_poly = pr.transform(X_test)

        # Train the model
        lr = LinearRegression()
        lr.fit(X_train_poly, y_train)

        # Save the model
        with open(model_filename, 'wb') as file:
            pickle.dump(lr, file)
        logging.info(f"Model saved successfully: {model_filename}")
    except Exception as e:
        logging.error(f"Error training or saving model: {e}")
        raise

def main():
    # File paths and configurations
    data_filepath = "data/AverageDailyPolyclinicAttendancesforSelectedDiseases.parquet"
    diseases = {
        "Acute Conjunctivitis": ("models/reg_model_acute_conjunctivitis.pkl", 16),
        "Acute Diarrhoea": ("models/reg_model_acute_diarrhoea.pkl", 9),
        "Acute Upper Respiratory Tract infections": ("models/reg_model_acute_urti.pkl", 9)
    }
    cutoff_week = "2022-W26"

    # Load data
    data = load_data(data_filepath)

    # Train and save models for each disease
    for disease, (model_filename, degree) in diseases.items():
        disease_df = data[data["disease"] == disease].reset_index(drop=True)
        train_df, test_df = split_data(disease_df, cutoff_week)
        train_and_save_model(train_df, test_df, model_filename, degree)

if __name__ == "__main__":
    main()
