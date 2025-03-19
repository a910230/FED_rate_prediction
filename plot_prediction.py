# plot_prediction.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_prediction(full_df, predictions):
    # Plot the predictions against the historical data
    future_dates = pd.date_range(start="2024-01-01", periods=len(predictions), freq="MS")
    plt.figure(figsize=(12, 6))
    plt.plot(full_df.index, full_df["Federal Funds Rate"], label="Historical")
    plt.plot(future_dates, predictions, label="Predicted 2024-2025")
    plt.title("Federal Funds Rate Prediction")
    plt.xlabel("Date")
    plt.ylabel("Rate (%)")
    plt.legend()
    plt.savefig("prediction_plot.png")
    plt.close()

    historical_data = full_df["Federal Funds Rate"].values[-len(predictions):]
    errors_ratio = np.abs(predictions - historical_data) / historical_data
    print(f"Mean Error Ratio: {np.mean(errors_ratio * 100):.2f}%")