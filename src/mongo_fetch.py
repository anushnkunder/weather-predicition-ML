from pymongo import MongoClient
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def fetch_from_mongo():
    try:
        logging.info("Connecting to MongoDB...")
        client = MongoClient("mongodb://localhost:27017/")

        db = client["weather_ml"]
        collection = db["weather_data"]

        logging.info("Fetching data from MongoDB...")
        data = list(collection.find())

        logging.info(f"Fetched {len(data)} records")

        df = pd.DataFrame(data)

        # Remove MongoDB internal _id field
        if "_id" in df.columns:
            df.drop("_id", axis=1, inplace=True)

        return df

    except Exception as e:
        logging.error(f"Error: {e}")

    finally:
        client.close()


if __name__ == "__main__":
    df = fetch_from_mongo()
    print(df.head())
    print("\nTotal rows:", len(df))
