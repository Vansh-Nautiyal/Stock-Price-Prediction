import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------------------
# Compute metrics
# -----------------------------------------
def evaluate_model(real, predicted):
    mae = mean_absolute_error(real, predicted)
    mse = mean_squared_error(real, predicted)
    rmse = np.sqrt(mse)

    return mae, mse, rmse


# -----------------------------------------
# Loss Plot
# -----------------------------------------
def plot_loss(history):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history['loss'], label="Loss")
    ax.plot(history.history['val_loss'], label="Val Loss")
    ax.legend()
    ax.set_title("Training Loss")
    return fig   # <-- IMPORTANT


# -----------------------------------------
# Predictions Plot
# -----------------------------------------
def plot_predictions(real, pred):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(real, label="Real")
    ax.plot(pred, label="Predicted")
    ax.legend()
    ax.set_title("Prediction vs Actual")
    return fig

import matplotlib.pyplot as plt

def plot_future_predictions(future_prices):
    """
    Returns a matplotlib figure object for Streamlit.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(range(1, len(future_prices)+1), future_prices, marker='o')
    ax.set_title("Next {} Days Predicted Close Prices".format(len(future_prices)))
    ax.set_xlabel("Days Ahead")
    ax.set_ylabel("Predicted Price")
    
    return fig