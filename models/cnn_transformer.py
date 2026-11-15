import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def positional_encoding(length, depth):
    pos = np.arange(length)[:, None]
    i   = np.arange(depth)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / depth)
    pe = np.zeros((length, depth), dtype=np.float32)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.constant(pe[None, ...], dtype=tf.float32)

def build_cnn_transformer(lookback, n_features, learning_rate=3e-4, d_model=32, num_heads=4, ff_dim=64, num_blocks=2, dropout=0.2):
    inp = keras.Input(shape=(lookback, n_features))
    
    # 1D CNN for local feature extraction
    x = layers.Conv1D(filters=d_model, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    
    x = x + positional_encoding(lookback, d_model)
    
    for _ in range(num_blocks):
        a = layers.LayerNormalization(epsilon=1e-6)(x)
        a = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout)(a, a)
        x = x + layers.Dropout(dropout)(a)
        
        f = layers.LayerNormalization(epsilon=1e-6)(x)
        f = layers.Dense(ff_dim, activation="relu")(f)
        f = layers.Dropout(dropout)(f)
        f = layers.Dense(d_model)(f)
        x = x + f
        
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1)(x)   # predicts the forward change in the cointegration spread

    model = keras.Model(inp, out, name="CNN_Transformer_forecaster")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss="huber")
    return model
