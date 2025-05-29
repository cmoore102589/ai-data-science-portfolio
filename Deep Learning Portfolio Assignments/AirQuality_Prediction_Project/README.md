# Air Quality Prediction Using Deep Learning

## Project Overview
This project focuses on predicting **PM2.5 concentrations** using various environmental and atmospheric indicators. As a data scientist at a climate research company, you are tasked with building, evaluating, and tuning deep learning models using both **TensorFlow** and **PyTorch**.

The dataset includes multiple features such as pollutant concentrations, meteorological variables, and timestamps collected from multiple monitoring stations across Beijing.

## Dataset
- **Source**: [Beijing Multi-site Air Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)
- **Features**:
  - Year, Month, Day, Hour
  - PM10, PM2.5, SO2, NO2, CO, O3
  - TEMP, PRES, DEWP, RAIN, Wind Direction (wd), Wind Speed (WSPM)
- **Target Variable**: PM2.5 (µg/m³)

## Tasks Completed
1. Loaded and combined data from multiple monitoring stations.
2. Handled missing values and applied one-hot encoding for categorical features.
3. Normalized the dataset to [0, 1] using `MinMaxScaler`.
4. Built and evaluated a baseline TensorFlow model with two hidden layers (20, 10 neurons).
5. Applied Keras Tuner to optimize the model’s structure and dropout rates.
6. Built a parallel model using PyTorch with the same architecture.
7. Evaluated overfitting/underfitting through training and validation loss plots.

## Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- TensorFlow / Keras
- PyTorch
- Keras Tuner
- Scikit-learn

## Model Summary

| Framework    | Architecture         | Performance Notes                       |
|--------------|----------------------|-----------------------------------------|
| TensorFlow   | 2 hidden layers (20-10) | Underfitting observed in baseline model |
| TensorFlow + Tuner | Tuned layers & dropout | Improved validation loss after tuning  |
| PyTorch      | 2 hidden layers (20-10) | Well-balanced model, no overfitting     |

## Visualizations
- Missing value heatmap
- Loss curves (training vs validation)
- Comparative plots for tuned and untuned models

## Folder Structure
