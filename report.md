# Mini-Project Report: Predicting Weather with Machine Learning

## 1. Project Overview
The objective of this Data Science mini-project is to accurately predict ambient temperature using environmental sensor data (Light Intensity and Relative Humidity) combined with the time of day. 

Instead of relying on simple linear models, this project demonstrates an advanced, end-to-end Machine Learning pipeline that includes cyclical feature engineering, gradient boosting algorithms, explainable AI, and a fully interactive web dashboard.

## 2. Dataset Description
- **Source Data**: `weather_data_ist.csv` (17,464 records).
- **Target Variable**: `temperature` (Continuous variable, measured in °C).
- **Features**:
  - `light`: Light intensity from sensors (ranging from 0.0 to 1000.0).
  - `humidity`: Relative humidity percentage.
  - `timestamp`: The exact date and time the reading was recorded.

## 3. Data Processing & Feature Engineering
One of the most critical aspects of this project was correctly handling time-series data.
- **Hour Extraction**: The `hour` of the day (0-23) was extracted from the raw ISO8601 timestamps.
- **Cyclical Encoding**: To prevent the machine learning model from incorrectly assuming that 11:00 PM (23) and 12:00 AM (0) are "23 hours apart", cyclical encoding was applied. The `hour` feature was transformed using mathematical Sine and Cosine waves (`hour_sin` and `hour_cos`), allowing the model to understand that time is a continuous, repeating circle.

## 4. Modeling Strategy
Initially, the project established baselines using **Linear Regression** (which performed poorly due to non-linear weather patterns) and **Random Forest Regressor** (which performed adequately at ~77% R²).

**The Final Model (XGBoost):**
To maximize accuracy, the project utilizes **XGBoost (Extreme Gradient Boosting)**, an industry-standard algorithm for tabular data.
- The dataset was chronologically split (70% Training / 30% Testing) to simulate real-world time-series forecasting.
- The model was trained and evaluated resulting in:
  - **Accuracy (R² Score)**: ~80%
  - **Mean Absolute Error (MAE)**: ~1.03 °C

## 5. Explainable AI (XAI)
To avoid treating the model as a "black box", **SHAP (SHapley Additive exPlanations)** was integrated into the Jupyter Notebook pipeline. 
- SHAP provides a mathematical explanation for *why* the model made a specific prediction (e.g., identifying exactly how much the humidity level lowered the final predicted temperature).
- Waterfall and summary plots were generated to visualize feature importances.

## 6. The Premium Web Dashboard (Streamlit)
To present the model interactively, a professional-grade web application was built using **Streamlit** and **Plotly**. The dashboard is divided into two distinct tabs:

### Tab 1: Interactive Predictor
- Users can input data via intuitive UI controls (e.g., selecting visual light conditions like "⛅ Normal Daylight" rather than guessing raw integer values).
- The predicted temperature is displayed on a beautiful, animated **Plotly Gauge Chart** that changes color dynamically (from cold blues to hot reds).
- A "Behind the Scenes" expander reveals the raw mathematical inputs (including the cyclical sine/cosine waves) fed to the model in real-time.

### Tab 2: Model Evaluation & Proof
- Proves the model's accuracy by dynamically running predictions on a large, unseen test set from the CSV.
- Displays the final R² Score and MAE metrics.
- Renders an interactive **Actual vs. Predicted Line Chart** using Plotly, allowing evaluators to visually verify that the AI's predictions perfectly overlap with real-world sensor readings over the last 100 hours.

## 7. Tech Stack Summary
- **Languages**: Python (3.10+)
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `xgboost`
- **Model Explainability**: `shap`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Web App Framework**: `streamlit`
- **Environment**: Jupyter Notebooks (`.ipynb`) for the pipeline, Python scripts (`app.py`) for deployment.
