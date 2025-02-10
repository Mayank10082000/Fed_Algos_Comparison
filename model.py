import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input

def create_model(n_timesteps, n_features, n_outputs):
    model = Sequential([
        # Input layer with explicit shape
        Input(shape=(n_timesteps, n_features)),
        
        # First LSTM layer
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        # Second LSTM layer
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        
        # Third LSTM layer
        LSTM(128),
        BatchNormalization(),
        Dropout(0.2),
        
        # Dense layers for classification
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_outputs, activation='softmax')
    ])
    
    # Compile with appropriate loss and metrics
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model