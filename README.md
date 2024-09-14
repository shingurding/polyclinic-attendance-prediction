# Polyclinic Attendance Prediction

This repository includes different prediction models to predict the average daily number of patients in a polyclinic.

## Set up
1. Clone this repository
    ```
    git clone git@github.com:shingurding/polyclinic-attendance-prediction.git
    ```
2. Create a virtual envrironment:
    ```
    python -m venv .venv

    # On Mac
    source .venv/bin/activate

    # On Windows
    venv\Scripts\activate
    ```
3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Running the models
To run the ARIMA model, execute the following line:
```
python arima_model.py
```

To run the random forest model, execute the following line:
```
python random_forest_model.py
```

## Repository Structure
```
.
├── data
│    └── AverageDailyPolyclinicAttendancesforSelectedDiseases.parquet
├── results
│    └── ...
├── models.ipynb
├── models
│    ├── arima_model.py
│    ├── random_forest_model.py
│    ├── ...
│    └── ... (other methods)
├── Dockerfile
```
