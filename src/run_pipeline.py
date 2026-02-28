from mongo_fetch import fetch_from_mongo
from preprocess import clean_data
from model import train_model
from visualize import (
    plot_feature_importance,
    plot_actual_vs_predicted,
    plot_residuals
)

print("Starting full pipeline...\n")

df = fetch_from_mongo()
df_clean = clean_data(df)

lr, rf, X_test, y_test = train_model(df_clean)

# Random Forest predictions again for visualization
rf_predictions = rf.predict(X_test)

plot_feature_importance(rf)
plot_actual_vs_predicted(y_test, rf_predictions)
plot_residuals(y_test, rf_predictions)

print("Pipeline executed successfully.")