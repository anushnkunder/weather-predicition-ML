# FAIML Weather Prediction

A machine learning pipeline for temperature prediction using weather sensor data (`light`, `humidity`, and timestamp-derived `hour`) with two baseline models:
- Linear Regression
- Random Forest Regressor

The project includes data conversion, MongoDB ingestion/fetch, preprocessing, model training/evaluation, and visualization generation.

## Project Structure

```text
FAIML-Weather_Prediction/
├── data/
│   ├── weather_data_ist.csv
│   └── weather_data.json
├── src/
│   ├── convert.py
│   ├── mongo_upload.py
│   ├── mongo_fetch.py
│   ├── preprocess.py
│   ├── model.py
│   ├── visualize.py
│   ├── run_pipeline.py
│   └── outputs/
│       ├── feature_importance.png
│       ├── actual_vs_predicted.png
│       └── residual_distribution.png
└── README.md
```

## Dataset

Current JSON dataset (`data/weather_data.json`):
- Records: `17464`
- Fields: `temperature`, `humidity`, `light`, `timestamp`

## Prerequisites

- Python 3.10+
- MongoDB running locally at `mongodb://localhost:27017/`

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib pymongo
```

## Workflow

### 1. (Optional) Convert CSV to JSON

```bash
python src/convert.py
```

Converts `data/weather_data_ist.csv` to `data/weather_data.json`.

### 2. Upload JSON data to MongoDB

```bash
python src/mongo_upload.py
```

- Database: `weather_ml`
- Collection: `weather_data`

### 3. Run the end-to-end ML pipeline

Run from the repository root:

```bash
cd src
python run_pipeline.py
```

Pipeline steps:
1. Fetch records from MongoDB
2. Preprocess and engineer `hour` from `timestamp`
3. Chronological train/test split (70/30)
4. Train and evaluate Linear Regression + Random Forest
5. Save plots to `src/outputs/`

## Model Inputs and Target

- Features: `light`, `humidity`, `hour`
- Target: `temperature`

## Output Artifacts

After running the pipeline, generated charts are saved in `src/outputs/`:
- `feature_importance.png`
- `actual_vs_predicted.png`
- `residual_distribution.png`

## Notes

- `run_pipeline.py` uses local module imports (`from mongo_fetch import ...`), so execute it from inside `src/`.
- `mongo_upload.py` currently uses an absolute JSON path in `__main__`. Update it if your local project path differs.
