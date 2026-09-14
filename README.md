# Predicting Weather with Machine Learning (Advanced Data Science Mini-Project)

A state-of-the-art Data Science mini-project for temperature prediction using weather sensor data (`light`, `humidity`) and time (`hour`).

This project demonstrates a professional-grade Data Science pipeline, including:
- Data Loading and Cleaning
- **Cyclical Feature Engineering** for Time Series
- **XGBoost** Modeling (Gradient Boosting)
- **Explainable AI (SHAP)** to visually interpret the model's decisions
- **Interactive Web UI** using Streamlit

## Project Structure

```text
FAIML-Weather_Prediction/
├── data/
│   ├── weather_data_ist.csv
│   └── weather_data.json
├── Weather_Prediction_Mini_Project.ipynb  # Core Data Science Pipeline
├── app.py                                 # Streamlit Web Application
├── requirements.md
└── README.md
```

## Dataset

The primary dataset is located at `data/weather_data_ist.csv`.
- Features: `light`, `humidity`, `timestamp`
- Target: `temperature`

## Prerequisites

- Python 3.10+

Install the required dependencies using your virtual environment:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter ipykernel xgboost shap streamlit joblib
```

## How to Run the Analysis (Jupyter Notebook)

1. Start the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
2. Open `Weather_Prediction_Mini_Project.ipynb` in your browser.
3. Select "Run All Cells" to execute the analysis, train the XGBoost model, generate the SHAP plots, and save the model to `best_model.pkl`.

## How to Run the Web Application (Streamlit)

Once you have run the notebook (which generates `best_model.pkl`), you can launch the interactive Web UI!

```bash
streamlit run app.py
```
This will open a local web page where you can adjust the sensor values on a slider and see the XGBoost model's temperature prediction update in real-time.

## Notes
- Previous terminal-based pipeline scripts have been archived in the git history under the tag `v1.0-faiml-pipeline`.
