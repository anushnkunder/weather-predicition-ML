import pandas as pd
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

def convert_csv_to_json(csv_path, output_path):
    try:
        logging.info("Reading CSV file...")
        df = pd.read_csv(csv_path)

        logging.info(f"CSV loaded successfully. Rows: {len(df)}")

        logging.info("Converting to JSON format...")
        df.to_json(output_path, orient="records", indent=4)

        logging.info(f"JSON file saved at: {output_path}")

    except Exception as e:
        logging.error(f"Error during conversion: {e}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(BASE_DIR, "data", "weather_data_ist.csv")
    json_file = os.path.join(BASE_DIR, "data", "weather_data.json")

    convert_csv_to_json(csv_file, json_file)