import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------
# Scaling function
# ----------------------------------------------------
def scale_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# ----------------------------------------------------
# Sequence creation for multivariate LSTM
# ----------------------------------------------------
def create_sequence(data, seq_length=60):
    X = []
    y = []

    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])      # All features for last 60 days
        y.append(data[i, 0])                  # Only 'Close' as output

    return np.array(X), np.array(y)

# ----------------------------------------------------
# Train-test split (80/20)
# ----------------------------------------------------
def train_test_split(x, y, split_ratio=0.8):
    split_index = int(len(x) * split_ratio)
    return x[:split_index], y[:split_index], x[split_index:], y[split_index:]

def get_last_sequence(data, seq_length=60):
    return data[-seq_length:]
