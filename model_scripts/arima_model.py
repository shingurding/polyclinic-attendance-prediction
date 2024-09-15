"""
arima_model.py

This module trains SARIMAX models to forecast the average number of cases for different diseases in a polyclinic and saves the models to disk.
"""
import pickle
import pandas as pd
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """
    Load the dataset from a parquet file.

    Args:
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

    Args:
    df (pd.DataFrame): The dataset to split.
    cutoff_week (str): The week to use as the cutoff for splitting the data.

    Returns:
    tuple: (train_df, test_df)
    """
    train = df[df["epi_week"] <= cutoff_week]
    test = df[df["epi_week"] > cutoff_week]
    return train, test

def train_and_save_model(train_df, model_filename, order, seasonal_order):
    """
    Train an SARIMAX model and save it to a file.

    Args:
    train_df (pd.DataFrame): Training dataset.
    model_filename (str): Path to save the trained model.
    order (tuple): ARIMA order parameters (p, d, q).
    seasonal_order (tuple): Seasonal order parameters (P, D, Q, s).
    """
    try:
        model = SARIMAX(train_df["no._of_cases"].values,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        model_fit = model.fit(disp=0)
        with open(model_filename, 'wb') as file:
            pickle.dump(model_fit, file)
        logging.info(f"Model saved successfully: {model_filename}")
    except Exception as e:
        logging.error(f"Error training or saving model: {e}")
        raise

def main():
    # Read data
    data_filepath = "data/AverageDailyPolyclinicAttendancesforSelectedDiseases.parquet"
    diseases = {
        "Acute Conjunctivitis": ("models/arima_acute_conjunctivitis.pkl", (1, 1, 2), (1, 1, 1, 26)),
        "Acute Diarrhoea": ("models/arima_acute_diarrhoea.pkl", (1, 1, 2), (1, 1, 1, 13)),
        "Acute Upper Respiratory Tract infections": ("models/arima_acute_urti.pkl", (1, 2, 2), (1, 1, 1, 13))
    }
    cutoff_week = "2022-W26"

    # Load data
    data = load_data(data_filepath)

    # Train and save models for each disease
    for disease, (model_filename, order, seasonal_order) in diseases.items():
        disease_df = data[data["disease"] == disease].reset_index(drop=True)
        train_df, _ = split_data(disease_df, cutoff_week)
        train_and_save_model(train_df, model_filename, order, seasonal_order)

if __name__ == "__main__":
    main()
