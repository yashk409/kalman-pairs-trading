import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_nbeats_block(inputs, units=64, theta_size=10):
    x = layers.Dense(units, activation="relu")(inputs)
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dense(units, activation="relu")(x)
    
    theta = layers.Dense(theta_size, activation="linear")(x)
    
    backcast = layers.Dense(inputs.shape[-1], activation="linear")(theta)
    forecast = layers.Dense(1, activation="linear")(theta)   # forward change in the cointegration spread
    
    return backcast, forecast

def build_nbeats(lookback, n_features, learning_rate=3e-4, num_blocks=3):
    inp = keras.Input(shape=(lookback, n_features))
    x = layers.Flatten()(inp)
    
    forecasts = []
    
    for _ in range(num_blocks):
        backcast, forecast = create_nbeats_block(x)
        x = layers.Subtract()([x, backcast])
        forecasts.append(forecast)
        
    out = layers.Add()(forecasts)
    
    model = keras.Model(inp, out, name="NBEATS_forecaster")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="huber")
    return model
