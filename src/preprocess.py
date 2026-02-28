import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def clean_data(df):
    try:
        logging.info("Starting preprocessing...")

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")

        # Sort chronologically (important for time-series)
        df = df.sort_values("timestamp")

        # Extract hour from timestamp
        df["hour"] = df["timestamp"].dt.hour

        logging.info("Feature engineering completed.")

        # Select only required features
        df_clean = df[["light", "humidity", "hour", "temperature"]]

        logging.info("Preprocessing completed successfully.")

        return df_clean

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")