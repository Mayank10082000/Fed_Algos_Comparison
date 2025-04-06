import os
import tensorflow as tf

def save_model(model, model_name):
    """
    Save a trained model to disk.
    
    Args:
        model: The trained model to save
        model_name: Name identifier for the saved model
    """
    try:
        # Create models directory if it doesn't exist
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
            
        # Save the model
        model_path = os.path.join('saved_models', f'{model_name}.h5')
        model.save(model_path)
        print(f"Model saved successfully at: {model_path}")
        
    except Exception as e:
        print(f"Error saving model {model_name}: {str(e)}")

def load_model(model_name):
    """
    Load a saved model from disk.
    
    Args:
        model_name: Name identifier of the saved model
        
    Returns:
        The loaded model
    """
    try:
        model_path = os.path.join('saved_models', f'{model_name}.h5')
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
            
        loaded_model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return loaded_model
        
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None