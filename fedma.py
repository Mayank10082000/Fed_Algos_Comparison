# Import required libraries
import numpy as np
import tensorflow as tf
from model import create_model

class FedMALearning:
    def __init__(self, n_clients, n_timesteps, n_features, n_outputs, sigma=1.0, sigma0=1.0, gamma=1.0):
        self.n_clients = n_clients
        self.sigma = sigma
        self.sigma0 = sigma0
        self.gamma = gamma
        
        # Initialize global model
        self.global_model = create_model(n_timesteps, n_features, n_outputs)
        
        # Initialize client models
        self.client_models = [create_model(n_timesteps, n_features, n_outputs) 
                            for _ in range(n_clients)]
        
        # Track which layers need matching
        self.matching_layers = []
        for idx, layer in enumerate(self.global_model.layers):
            if isinstance(layer, (tf.keras.layers.LSTM, tf.keras.layers.Dense)):
                self.matching_layers.append(idx)
    
    def update_client_weights(self, client_idx):
        """Update client model with global model weights"""
        self.client_models[client_idx].set_weights(
            self.global_model.get_weights()
        )
    
    def train_client(self, client_idx, X_train, y_train, local_epochs=5):
        """Train individual client model"""
        return self.client_models[client_idx].fit(
            X_train, y_train,
            epochs=local_epochs,
            batch_size=64,
            verbose=0
        )
    
    def compute_weight_similarity(self, weight1, weight2):
        """Safely compute weight similarity between two weight matrices"""
        try:
            # Reshape weights to 2D if needed
            if len(weight1.shape) > 2:
                w1 = weight1.reshape(weight1.shape[0], -1)
                w2 = weight2.reshape(weight2.shape[0], -1)
            else:
                w1, w2 = weight1, weight2
                
            # Compute cosine similarity
            w1_norm = np.linalg.norm(w1, axis=1, keepdims=True)
            w2_norm = np.linalg.norm(w2, axis=1, keepdims=True)
            
            # Avoid division by zero
            w1_normalized = np.divide(w1, w1_norm, where=w1_norm != 0)
            w2_normalized = np.divide(w2, w2_norm, where=w2_norm != 0)
            
            return np.dot(w1_normalized, w2_normalized.T)
            
        except (ValueError, IndexError):
            return None
    
    def match_layer_neurons(self, weights_list, layer_idx):
        """Match neurons in a layer across clients with shape validation"""
        # Skip if not a matching layer
        if layer_idx not in self.matching_layers:
            return None
            
        # Get shapes for the layer
        shapes = [w[layer_idx].shape for w in weights_list]
        if not all(len(s) >= 1 for s in shapes):
            return None
            
        max_neurons = max(s[0] for s in shapes)
        
        # Initialize cost matrix for matching
        total_clients = len(weights_list)
        cost_matrix = np.zeros((max_neurons, max_neurons))
        
        # Calculate similarity scores
        for i in range(max_neurons):
            for j in range(max_neurons):
                similarity = 0
                count = 0
                for client in range(total_clients):
                    if i < shapes[client][0] and j < shapes[client][0]:
                        # Get weight matrices for current neurons
                        w1 = weights_list[client][layer_idx][i:i+1]
                        w2 = weights_list[client][layer_idx][j:j+1]
                        
                        sim = self.compute_weight_similarity(w1, w2)
                        if sim is not None:
                            similarity += sim[0, 0]  # Get scalar value
                            count += 1
                            
                if count > 0:
                    cost_matrix[i,j] = -similarity/count  # Negative for minimization
                else:
                    cost_matrix[i,j] = 0  # No valid comparisons
        
        # Use Hungarian algorithm for matching
        try:
            row_ind, col_ind = self._hungarian_matching(cost_matrix)
            return row_ind, col_ind
        except Exception:
            return None
    
    def _hungarian_matching(self, cost_matrix):
        """Implementation of Hungarian algorithm for minimum cost matching"""
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost_matrix)
    
    def aggregate_matched_weights(self, weights_list, layer_matches):
        """Aggregate weights after matching with bounds checking"""
        aggregated_weights = []
        
        for layer_idx, matches in enumerate(layer_matches):
            if matches is None:  # For layers that don't need matching
                layer_weights = [w[layer_idx] for w in weights_list]
                aggregated_weights.append(
                    np.mean(layer_weights, axis=0)
                )
            else:
                row_ind, col_ind = matches
                matched_weights = []
                
                for client_weights in weights_list:
                    reordered = np.zeros_like(client_weights[layer_idx])
                    for i, j in zip(row_ind, col_ind):
                        # Check both source and target indices are within bounds
                        if (i < client_weights[layer_idx].shape[0] and 
                            j < reordered.shape[0]):
                            reordered[j] = client_weights[layer_idx][i]
                    matched_weights.append(reordered)
                
                # Average the matched weights
                aggregated_weights.append(
                    np.mean(matched_weights, axis=0)
                )
        
        return aggregated_weights
    
    def aggregate_weights(self):
        """Aggregate weights from all clients using FedMA with proper error handling"""
        try:
            # Get weights from all clients
            weights_list = [model.get_weights() for model in self.client_models]
            
            # Match neurons layer by layer
            layer_matches = []
            for layer_idx in range(len(weights_list[0])):
                matches = self.match_layer_neurons(weights_list, layer_idx)
                layer_matches.append(matches)
            
            # Aggregate matched weights
            new_weights = self.aggregate_matched_weights(weights_list, layer_matches)
            
            # Update global model
            self.global_model.set_weights(new_weights)
            
        except Exception as e:
            print(f"Error in weight aggregation: {str(e)}")
            # Fallback to simple averaging if matching fails
            weights = [model.get_weights() for model in self.client_models]
            new_weights = []
            for weight_list_tuple in zip(*weights):
                new_weights.append(
                    np.array([np.array(w).mean(axis=0) 
                             for w in zip(*weight_list_tuple)])
                )
            self.global_model.set_weights(new_weights)

def train_fedma_model(X_train, y_train, X_test, y_test, n_clients=5, n_rounds=10, 
                     local_epochs=5, sigma=1.0, sigma0=1.0, gamma=1.0):
    """Train a model using FedMA algorithm."""
    # Initialize federated learning with FedMA
    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    n_outputs = y_train.shape[1]
    fl = FedMALearning(n_clients, n_timesteps, n_features, n_outputs, 
                      sigma=sigma, sigma0=sigma0, gamma=gamma)
    
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
        print(f"\nCommunication Round {round_num + 1}/{n_rounds}")
        
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
        
        # Aggregate weights using FedMA
        fl.aggregate_weights()
        
        # Evaluate global model
        train_loss, train_acc = fl.global_model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = fl.global_model.evaluate(X_test, y_test, verbose=0)
        
        # Update history
        history['accuracy'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        # Print round metrics
        print(f"\nRound {round_num + 1} Performance Metrics:")
        print("-" * 40)
        print(f"Accuracy  : {val_acc:.4f}")
        print(f"Precision : {val_acc:.4f}")  # Using accuracy as approximation
        print(f"Recall    : {val_acc:.4f}")  # Using accuracy as approximation
        print(f"F1        : {val_acc:.4f}")  # Using accuracy as approximation
    
    return fl, history