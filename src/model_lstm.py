from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    """
    input_shape = (sequence_length, num_features)
    """

    model = Sequential()

    # First LSTM Layer
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Second LSTM Layer
    model.add(LSTM(32))
    model.add(Dropout(0.2))

    # Output Layer → Predict next day's Close price
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")
    return model

import numpy as np

def predict_future(model, scaler, last_sequence, num_features, num_days=7):
    """
    Predict next N days using recursive predictions.

    last_sequence : last 60 days (scaled) → shape (60, num_features)
    """

    future_predictions = []

    curr_seq = last_sequence.copy()

    for _ in range(num_days):
        # Model expects shape: (1, 60, num_features)
        pred_scaled = model.predict(curr_seq.reshape(1, 60, num_features))[0][0]

        # Store scaled prediction
        future_predictions.append(pred_scaled)

        # Prepare input for next step:
        # Create dummy row with pred in Close column (index 0)
        next_row = curr_seq[-1].copy()
        next_row[0] = pred_scaled

        # Append next_row and slide window
        curr_seq = np.vstack([curr_seq[1:], next_row])

        # Convert scaled predictions → real prices
        dummy = np.zeros((len(future_predictions), num_features))
        dummy[:, 0] = future_predictions
        future_real = scaler.inverse_transform(dummy)[:, 0]

    return future_real
