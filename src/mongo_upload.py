from pymongo import MongoClient
import json
import logging

logging.basicConfig(level=logging.INFO)

def upload_to_mongo(json_path):
    try:
        logging.info("Connecting to MongoDB...")
        client = MongoClient("mongodb://localhost:27017/")

        db = client["weather_ml"]
        collection = db["weather_data"]

        logging.info("Loading JSON file...")
        with open(json_path, "r") as file:
            data = json.load(file)

        logging.info(f"Inserting {len(data)} records into MongoDB...")
        collection.insert_many(data)

        logging.info("Data uploaded successfully!")

    except Exception as e:
        logging.error(f"Error: {e}")

    finally:
        client.close()


if __name__ == "__main__":
    upload_to_mongo("/home/Anush/FAIML-Weather_Prediction/data/weather_data.json")