# Predicting Weather with Machine Learning

A Data Science mini-project for temperature prediction using weather sensor data (`light`, `humidity`) and time (`hour`).

This project demonstrates an end-to-end Data Science pipeline, including:
- Data Loading and Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training (Linear Regression vs. Random Forest)
- Evaluation and Visualization

## Project Structure

```text
FAIML-Weather_Prediction/
├── data/
│   ├── weather_data_ist.csv
│   └── weather_data.json
├── Weather_Prediction_Mini_Project.ipynb
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
pip install pandas scikit-learn matplotlib seaborn jupyter
```

## How to Run

Instead of terminal scripts, this project is now fully encapsulated within a Jupyter Notebook. This makes it easy to read the data story, view the plots inline, and understand the model's performance step-by-step.

1. Start the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
2. Open `Weather_Prediction_Mini_Project.ipynb` in your browser.
3. Select "Run All Cells" to execute the entire analysis from start to finish.

## Notes
- Previous terminal-based pipeline scripts (and MongoDB integration) have been archived in the git history under the tag `v1.0-faiml-pipeline`.
