import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_lstm(lookback, n_features, learning_rate=3e-4):
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(48, return_sequences=True,
                    kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.LSTM(24, kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),   # predicts the forward change in the cointegration spread
    ], name="LSTM_forecaster")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="huber")
    return model
