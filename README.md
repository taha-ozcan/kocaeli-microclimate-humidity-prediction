# 🌊 Kocaeli Microclimate Relative Humidity Prediction Model

This project analyzes and predicts the impact of microclimatic and topographic variations (altitude, distance to the sea, etc.) on **Relative Humidity** across Kocaeli, Turkey, using Machine Learning (**Random Forest Regressor**).

---

## 📌 Project Overview & Objective

Kocaeli features a complex geography encompassing the Black Sea coast, the Gulf of İzmit (Marmara Sea), the Samanlı Mountains, and dense industrial/urban areas. This topographic diversity creates distinct microclimates within small spatial scales. 

The primary goal of this project is to model and predict localized relative humidity by combining dynamic meteorological data with static geographic features.

---

## 📊 Dataset Specifications

* **Spatial Coverage:** 12 representative locations across Kocaeli (İzmit, Kartepe Summit, Kandıra Coast, Gebze, Gölcük, Karamürsel, Başiskele, Derince, Körfez, Dilovası, Çayırova, Yuvacık Reservoir).
* **Timeframe:** 1 full year of hourly historical data (365 days × 24 hours × 12 locations).
* **Data Volume:** 100,000+ records of time-series and spatial features.
* **Data Source:** Open-Meteo Historical Weather API.

---

## 🛠️ Feature Engineering

| Feature Type | Variable Name | Description |
| :--- | :--- | :--- |
| **Meteorological** | `temperature` | Ambient Air Temperature (°C) |
| **Meteorological** | `wind_speed` | Wind Speed at 10m (km/h) |
| **Meteorological** | `pressure` | Surface Atmospheric Pressure (hPa) |
| **Geographic** | `elevation` | Elevation above sea level (m) |
| **Geographic** | `dist_to_sea` | Distance to the nearest coast/water body (km) |
| **Temporal** | `month` / `hour` | Month of the year (1–12) and Hour of the day (0–23) |
| **Target Variable** | `humidity` | Relative Humidity (%) |

---

## 🚀 Model Performance & Validation

To avoid **Data Leakage** (e.g., excluding derived features like dew point), the model was evaluated using a strict spatial and temporal train-test split:

* **Algorithm:** Random Forest Regressor
* **R² Score (Variance Explained):** `79.68%`
* **MAE (Mean Absolute Error):** `±5.46%` relative humidity

---

## 💻 Getting Started

### Prerequisites

Install the required Python libraries:
```bash
pip install pandas numpy requests scikit-learn matplotlib
