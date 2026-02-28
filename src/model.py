import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO)

def train_model(df_clean):
    try:
        logging.info("Preparing features and target...")

        X = df_clean[["light", "humidity", "hour"]]
        y = df_clean["temperature"]

        # Chronological split (70/30)
        split_index = int(len(df_clean) * 0.7)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        # =========================
        # Linear Regression
        # =========================
        logging.info("Training Linear Regression model...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        lr_predictions = lr.predict(X_test)

        lr_mae = mean_absolute_error(y_test, lr_predictions)
        lr_mse = mean_squared_error(y_test, lr_predictions)
        lr_r2 = r2_score(y_test, lr_predictions)

        print("\n===== LINEAR REGRESSION =====")
        print("MAE:", round(lr_mae, 4))
        print("MSE:", round(lr_mse, 4))
        print("R2 Score:", round(lr_r2, 4))
        print("=============================\n")

        # =========================
        # Random Forest
        # =========================
        logging.info("Training Random Forest model...")
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        rf.fit(X_train, y_train)

        rf_predictions = rf.predict(X_test)

        rf_mae = mean_absolute_error(y_test, rf_predictions)
        rf_mse = mean_squared_error(y_test, rf_predictions)
        rf_r2 = r2_score(y_test, rf_predictions)

        print("===== RANDOM FOREST =====")
        print("MAE:", round(rf_mae, 4))
        print("MSE:", round(rf_mse, 4))
        print("R2 Score:", round(rf_r2, 4))
        print("=========================\n")

        return lr, rf, X_test, y_test

    except Exception as e:
        logging.error(f"Error in model training: {e}")
        return None