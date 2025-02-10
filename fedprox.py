# Import required libraries
import numpy as np
import tensorflow as tf
from model import create_model

class ProximalLoss(tf.keras.losses.Loss):
    def __init__(self, mu, global_weights, base_loss=None):
        super().__init__()
        self.mu = mu  
        self.global_weights = global_weights  
        self.base_loss = base_loss or tf.keras.losses.CategoricalCrossentropy()
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        # Calculate base loss
        original_loss = self.base_loss(y_true, y_pred)
        
        # Ensure model exists
        if not hasattr(self, 'model'):
            return original_loss
        
        # Get current model's trainable weights
        local_weights = self.model.trainable_weights
        
        # Calculate proximal term
        proximal_term = tf.constant(0.0, dtype=tf.float32)
        for local_w, global_w in zip(local_weights, self.global_weights):
            # Ensure weights are tensors and have compatible shapes
            local_w_tensor = tf.convert_to_tensor(local_w)
            global_w_tensor = tf.convert_to_tensor(global_w)
            
            # Only compare weights if shapes match
            if local_w_tensor.shape == global_w_tensor.shape:
                diff = local_w_tensor - global_w_tensor
                proximal_term += self.mu * tf.reduce_sum(tf.square(diff))
        
        return original_loss + proximal_term

class FedProxLearning:
    def __init__(self, n_clients, n_timesteps, n_features, n_outputs, mu=0.01):
        self.n_clients = n_clients
        self.mu = mu  # Proximal term coefficient
        
        # Initialize global model
        self.global_model = create_model(n_timesteps, n_features, n_outputs)
        # Initialize client models
        self.client_models = [create_model(n_timesteps, n_features, n_outputs) 
                            for _ in range(n_clients)]
    
    def update_client_weights(self, client_idx):
        """Update client model with global model weights"""
        self.client_models[client_idx].set_weights(
            self.global_model.get_weights()
        )
    
    def train_client(self, client_idx, X_train, y_train, local_epochs=5):
        """Train individual client model with proximal term"""
        # Get global weights for proximal term
        global_weights = self.global_model.get_weights()
        
        # Create proximal loss
        proximal_loss = ProximalLoss(
            mu=self.mu,
            global_weights=global_weights
        )
        
        # Explicitly set the model for the loss
        proximal_loss.model = self.client_models[client_idx]
        
        # Compile model with proximal loss
        self.client_models[client_idx].compile(
            optimizer='adam',
            loss=proximal_loss,
            metrics=['accuracy']
        )
        
        return self.client_models[client_idx].fit(
            X_train, y_train,
            epochs=local_epochs,
            batch_size=64,
            verbose=0
        )
    
    def aggregate_weights(self):
        """Aggregate weights from all clients"""
        weights = [model.get_weights() for model in self.client_models]
        
        new_weights = []
        for weight_list_tuple in zip(*weights):
            new_weights.append(
                np.array([np.array(w).mean(axis=0) 
                         for w in zip(*weight_list_tuple)])
            )
        
        self.global_model.set_weights(new_weights)

def train_fedprox_model(X_train, y_train, X_test, y_test, n_clients=5, n_rounds=10, 
                       local_epochs=5, mu=0.01):
    """
    Train a model using FedProx algorithm.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        n_clients: Number of federated clients
        n_rounds: Number of communication rounds
        local_epochs: Number of local training epochs per client
        mu: Proximal term coefficient
        
    Returns:
        fl: FedProxLearning instance
        history: Training history
    """
    # Initialize federated learning with FedProx
    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    n_outputs = y_train.shape[1]
    fl = FedProxLearning(n_clients, n_timesteps, n_features, n_outputs, mu=mu)
    
    # Training history
    history = {
        'accuracy': [],
        'loss': [],
        'val_accuracy': [],
        'val_loss': []
    }
    
    # Get baseline metrics
    base_loss, base_acc = fl.global_model.evaluate(X_test, y_test, verbose=0)
    history['accuracy'].append(base_acc)
    history['val_accuracy'].append(base_acc)
    history['loss'].append(base_loss)
    history['val_loss'].append(base_loss)
    
    # Training loop
    for round_num in range(n_rounds):
        # Split data among clients
        client_data = np.array_split(X_train, n_clients)
        client_labels = np.array_split(y_train, n_clients)
        
        # Train each client
        for client_idx in range(n_clients):
            fl.update_client_weights(client_idx)
            fl.train_client(
                client_idx,
                client_data[client_idx],
                client_labels[client_idx],
                local_epochs=local_epochs
            )
        
        # Aggregate weights
        fl.aggregate_weights()
        
        # Evaluate global model
        train_loss, train_acc = fl.global_model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = fl.global_model.evaluate(X_test, y_test, verbose=0)
        
        # Update history
        history['accuracy'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss)
    
    return fl, history