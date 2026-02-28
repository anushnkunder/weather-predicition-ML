import matplotlib.pyplot as plt
import os


def plot_feature_importance(rf_model):
    features = ["light", "humidity", "hour"]
    importances = rf_model.feature_importances_

    plt.figure()
    plt.bar(features, importances)
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/feature_importance.png")
    plt.close()


def plot_actual_vs_predicted(y_test, predictions):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    plt.figure()
    plt.scatter(y_test, predictions)

    # Fit linear regression line to the scatter
    m, b = np.polyfit(y_test, predictions, 1)
    plt.plot(y_test, m * y_test + b)

    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title("Actual vs Predicted Temperature (With Regression Line)")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/actual_vs_predicted.png")
    plt.close()

def plot_residuals(y_test, predictions):
    residuals = y_test - predictions

    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title("Residual Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/residual_distribution.png")
    plt.close()