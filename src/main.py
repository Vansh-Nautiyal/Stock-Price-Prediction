import numpy as np
from tensorflow.keras.callbacks import Callback

from src.data_loader import download_stock, save_data
from src.feature_engineering import create_features
from src.preprocess import (
    scale_data,
    create_sequence,
    train_test_split,
    get_last_sequence
)
from src.model_lstm import build_lstm_model, predict_future
from src.evaluate import (
    evaluate_model,
    plot_loss,
    plot_predictions,
    plot_future_predictions
)

# -------------------------------------------------------
# Streamlit Callback (Epoch Progress)
# -------------------------------------------------------
class StreamlitTrainingCallback(Callback):
    def __init__(self, placeholder, total_epochs):
        super().__init__()
        self.placeholder = placeholder
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        self.placeholder.write(
            f"**Epoch {epoch + 1}/{self.total_epochs} completed** — "
            f"loss: `{logs['loss']:.4f}`, val_loss: `{logs['val_loss']:.4f}`"
        )

# -------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------
def load_data(symbol, start, end):
    df = download_stock(symbol, start, end)
    save_data(df, f"{symbol}.csv")
    return df

# -------------------------------------------------------
# 2. Feature Engineering
# -------------------------------------------------------
def feature_creation(df):
    df = create_features(df)

    feature_cols = [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'return', 'log_return',
        'sma_10', 'sma_50',
        'ema_12', 'ema_26',
        'macd', 'macd_signal', 'macd_hist',
        'rsi',
        'close_lag1', 'close_lag2'
    ]

    return df[feature_cols].values, feature_cols

# -------------------------------------------------------
# 3. Scaling
# -------------------------------------------------------
def data_scaling(feature_data):
    return scale_data(feature_data)

# -------------------------------------------------------
# 4. Create Sequences
# -------------------------------------------------------
def sequence_creation(scaled_data):
    return create_sequence(scaled_data, seq_length=60)

# -------------------------------------------------------
# 5. Train-test split
# -------------------------------------------------------
def split_data(X, y):
    return train_test_split(X, y)

# -------------------------------------------------------
# 6. Train Model (LSTM)
# -------------------------------------------------------
def train_model(X_train, y_train, X_test, y_test,
                num_features, status_placeholder, epochs):

    model = build_lstm_model(input_shape=(60, num_features))
    callback = StreamlitTrainingCallback(status_placeholder, epochs)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=[callback],
        verbose=0
    )

    return model, history

# -------------------------------------------------------
# 7. Inverse Scaling
# -------------------------------------------------------
def inverse_scale(predicted_scaled, scaler, feature_cols, y_test):
    num_features = len(feature_cols)

    pred_dummy = np.zeros((predicted_scaled.shape[0], num_features))
    pred_dummy[:, 0] = predicted_scaled[:, 0]
    predicted_prices = scaler.inverse_transform(pred_dummy)[:, 0]

    real_dummy = np.zeros((y_test.shape[0], num_features))
    real_dummy[:, 0] = y_test
    real_prices = scaler.inverse_transform(real_dummy)[:, 0]

    return real_prices, predicted_prices

# -------------------------------------------------------
# 8. Full Pipeline (Used by Streamlit)
# -------------------------------------------------------
def run_prediction(symbol, start, end, status_placeholder, epochs):

    df = load_data(symbol, start, end)

    features, feature_cols = feature_creation(df)
    scaled_data, scaler = data_scaling(features)

    X, y = sequence_creation(scaled_data)
    X_train, y_train, X_test, y_test = split_data(X, y)

    num_features = X.shape[2]

    model, history = train_model(
        X_train, y_train,
        X_test, y_test,
        num_features,
        status_placeholder,
        epochs
    )

    predicted_scaled = model.predict(X_test)

    # -------- Future 7-day Forecast --------
    last_60_days = get_last_sequence(scaled_data)

    future_7_days = predict_future(
        model=model,
        scaler=scaler,
        last_sequence=last_60_days,
        num_features=num_features,
        num_days=7
    )

    future_fig = plot_future_predictions(future_7_days)

    real_prices, predicted_prices = inverse_scale(
        predicted_scaled, scaler, feature_cols, y_test
    )

    mae, mse, rmse = evaluate_model(real_prices, predicted_prices)

    return {
        "df": df,
        "real": real_prices,
        "pred": predicted_prices,
        "metrics": {"mae": mae, "mse": mse, "rmse": rmse},
        "loss_plot": plot_loss(history),
        "prediction_plot": plot_predictions(real_prices, predicted_prices),
        "future_prices": future_fig
    }
