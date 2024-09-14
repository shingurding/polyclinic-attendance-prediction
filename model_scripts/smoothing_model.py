"""
smoothing.py

This module applies Holt-Winters Exponential Smoothing methods for forecasting and saves the models to disk.
"""
import pickle
import pandas as pd
import logging
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """
    Load the dataset from a parquet file.

    Atgs:
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

    Atgs:
    df (pd.DataFrame): The dataset to split.
    cutoff_week (str): The week to use as the cutoff for splitting the data.

    Returns:
    tuple: (train_df, test_df)
    """
    train = df[df["epi_week"] <= cutoff_week]
    test = df[df["epi_week"] > cutoff_week]
    return train, test

def train_and_save_model(train_df, model_filename, seasonal_period, smoothing_params):
    """
    Train an Exponential Smoothing model and save it to a file.

    Args:
    train_df (pd.DataFrame): Training dataset.
    model_filename (str): Path to save the trained model.
    seasonal_period (int): Seasonal period for the model.
    smoothing_params (dict): Dictionary of smoothing parameters.
    """
    try:
        if smoothing_params:
            hw_model = ExponentialSmoothing(
                train_df["no._of_cases"],
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_period
            )
            hw_fit = hw_model.fit(
                smoothing_level=smoothing_params['smoothing_level'],
                smoothing_trend=smoothing_params['smoothing_trend'],
                smoothing_seasonal=smoothing_params['smoothing_seasonal']
            )
        else:
            # Use default parameters if smoothing_params is None
            hw_model = ExponentialSmoothing(
                train_df["no._of_cases"],
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_period
            )
            hw_fit = hw_model.fit()

        with open(model_filename, 'wb') as file:
            pickle.dump(hw_fit, file)
        logging.info(f"Model saved successfully: {model_filename}")
    except Exception as e:
        logging.error(f"Error training or saving model: {e}")
        raise

def main():
    # File paths and configurations
    data_filepath = "data/AverageDailyPolyclinicAttendancesforSelectedDiseases.parquet"
    diseases = {
        "Acute Conjunctivitis": ("models/smoothing_acute_conjunctivitis.pkl", 26, {'smoothing_level': 0.5, 'smoothing_trend': 0.3, 'smoothing_seasonal': 0.5}),
        "Acute Diarrhoea": ("models/smoothing_acute_diarrhoea.pkl", 13, {'smoothing_level': 0.5, 'smoothing_trend': 0.4, 'smoothing_seasonal': 0.3}),
        "Acute Upper Respiratory Tract infections": ("models/smoothing_acute_urti.pkl", 13, None)
    }
    cutoff_week = "2022-W26"

    # Load data
    data = load_data(data_filepath)

    # Train and save models for each disease
    for disease, (model_filename, seasonal_period, smoothing_params) in diseases.items():
        disease_df = data[data["disease"] == disease].reset_index(drop=True)
        train_df, _ = split_data(disease_df, cutoff_week)
        train_and_save_model(train_df, model_filename, seasonal_period, smoothing_params)

if __name__ == "__main__":
    main()
