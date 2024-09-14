# Polyclinic Attendance Prediction

This repository includes various prediction models to forecast the average daily number of patients in a polyclinic.

## Project Overview
This project aims to predict polyclinic attendance using different time series forecasting models. The goal is to help in planning and resource allocation for healthcare services.

## Features
- Implements multiple prediction models including classical linear regression, ARIMA, and smoothing techniques.
- Provides tools for evaluating model performance and forecasting future attendance.

## Setup
1. Clone this repository:
    ```bash
    git clone git@github.com:shingurding/polyclinic-attendance-prediction.git
    ```
2. Create a virtual environment:
    ```bash
    python -m venv .venv

    # On macOS or Linux
    source .venv/bin/activate

    # On Windows
    .venv\Scripts\activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the models
To run the models, execute the following commands:
```bash
python model_scripts/classical_lin_reg.py
python model_scripts/arima_model.py
python model/smoothing_model.py
```

## Repository Structure
```scss
.
├── data/
│    └── AverageDailyPolyclinicAttendancesforSelectedDiseases.parquet
├── models/
│    └── ... (saved models)
├── models.ipynb
├── model_scripts/
│    ├── classical_lin_reg.py
│    ├── arima_model.py
│    └── smoothing_model.py
```
