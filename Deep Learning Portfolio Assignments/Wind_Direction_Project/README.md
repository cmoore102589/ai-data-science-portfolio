# Wind Direction Prediction Using Deep Learning

## Project Overview
This project focuses on classifying **wind direction (wd)** using a multi-class classification approach. The dataset is derived from the Beijing Multi-Site Air Quality dataset and includes meteorological and environmental features such as pollutant levels, temperature, pressure, and humidity.

Both **TensorFlow** and **PyTorch** models were implemented and evaluated, including hyperparameter tuning using **Keras Tuner** to find the optimal neural network architecture.

## Dataset
- **Source**: [Beijing Multi-site Air Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)
- **Target Variable**: Wind Direction (`wd`)
- **Features Used**:
  - PM2.5, PM10, SO2, NO2, CO, O3
  - TEMP, PRES, DEWP, RAIN, WSPM
- **Dropped**: Time-based columns (year, month, day, hour) and station identifiers

## Tasks Completed
1. Merged multiple CSV files and cleaned the dataset.
2. Imputed missing values using median (numerical) and mode (categorical).
3. One-hot encoded wind direction labels.
4. Normalized features to a range of [0, 1].
5. Built and evaluated a baseline TensorFlow model with two hidden layers (20 and 10 neurons).
6. Tuned the TensorFlow model using Keras Tuner (unit counts and dropout rates).
7. Built and trained a PyTorch neural network with a similar structure.
8. Evaluated model performance and discussed overfitting/underfitting tendencies.

## Technologies Used
- Python, Pandas, NumPy
- TensorFlow / Keras
- PyTorch
- Scikit-learn
- Keras Tuner
- Matplotlib

## Model Summary

| Framework    | Architecture           | Performance Notes                        |
|--------------|------------------------|------------------------------------------|
| TensorFlow   | 2 layers (20-10 units) | Stable training, validation loss aligned |
| TensorFlow + Tuner | Tuned layers + dropout | Improved accuracy, reduced overfitting  |
| PyTorch      | 2 layers (20-10 units) | Underfitting observed, needs complexity  |

## Folder Structure
