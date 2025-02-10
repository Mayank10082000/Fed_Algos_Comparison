# Import required libraries
import tensorflow as tf
from model import create_model

def train_centralized_model(X_train, y_train, X_test, y_test, n_epochs=50, batch_size=64):
    """
    Train a centralized model on the complete dataset.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Get model dimensions
    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    n_outputs = y_train.shape[1]
    
    # Create and compile model
    model = create_model(n_timesteps, n_features, n_outputs)

    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model with progress tracking
    history = model.fit(
        X_train, y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Return model and history
    return model, {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }