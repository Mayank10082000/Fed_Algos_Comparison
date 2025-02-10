# Import required libraries
import os
import numpy as np
import logging
import tensorflow as tf

# Import custom modules
from data_preprocessing import load_and_preprocess_data, prepare_sequences
from centralized_learning import train_centralized_model
from fedavg import train_federated_model
from fedprox import train_fedprox_model
from fedma import train_fedma_model  # Add this import
from metrics import calculate_metrics, plot_metrics, plot_confusion_matrix, save_simple_metrics

def main():
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        file_path = '/kaggle/input/wisdm-har-raw-csv/wisdm_HAR_raw.csv'  # Update this path to your data file
        X_scaled, y = load_and_preprocess_data(file_path)
        
        # Get number of classes from the data
        num_classes = len(np.unique(y))
        print(f"\nNumber of classes detected: {num_classes}")
        
        # Prepare sequences
        print("\nPreparing sequences...")
        X_train, X_test, y_train, y_test = prepare_sequences(
            X_scaled, y, sequence_length=128
        )
        
        #========== CENTRALIZED LEARNING ==========
        print("\n" + "="*50)
        print("CENTRALIZED LEARNING")
        print("="*50)
        
        # Train centralized model
        print("\nTraining centralized model...")
        centralized_model, centralized_history = train_centralized_model(
            X_train, y_train, 
            X_test, y_test,
            n_epochs=50,
            batch_size=64
        )
        
        # Get predictions and calculate metrics for centralized model
        print("\nCalculating centralized model metrics...")
        centralized_pred = centralized_model.predict(X_test, verbose=0)
        centralized_metrics = calculate_metrics(y_test, centralized_pred)
        
        print("\nCentralized Learning Final Metrics:")
        print("-" * 40)
        print(f"Accuracy  : {centralized_metrics['accuracy']:.4f}")
        print(f"Precision : {centralized_metrics['precision']:.4f}")
        print(f"Recall    : {centralized_metrics['recall']:.4f}")
        print(f"F1        : {centralized_metrics['f1']:.4f}")
        
        # Generate centralized model visualizations
        plot_metrics(centralized_history, mode='centralized')
        plot_confusion_matrix(y_test, centralized_pred, 
                            mode='centralized', num_classes=num_classes)
        
        #========== FEDERATED LEARNING - FedAvg ==========
        print("\n" + "="*50)
        print("FEDERATED LEARNING - FedAvg")
        print("="*50)
        
        # Train federated model
        fl_model, federated_history = train_federated_model(
            X_train, y_train,
            X_test, y_test,
            n_clients=5,
            n_rounds=10,
            local_epochs=5
        )
        
        # Generate federated model visualizations
        plot_metrics(federated_history, mode='federated')
        
        # Get predictions for federated model
        federated_pred = fl_model.global_model.predict(X_test, verbose=0)
        federated_metrics = calculate_metrics(y_test, federated_pred)
        
        print("\nFedAvg Final Metrics:")
        print("-" * 40)
        print(f"Accuracy  : {federated_metrics['accuracy']:.4f}")
        print(f"Precision : {federated_metrics['precision']:.4f}")
        print(f"Recall    : {federated_metrics['recall']:.4f}")
        print(f"F1        : {federated_metrics['f1']:.4f}")
        
        # Plot confusion matrix for federated model
        plot_confusion_matrix(y_test, federated_pred, 
                            mode='federated', num_classes=num_classes)

        #========== FEDERATED LEARNING - FedProx ==========
        print("\n" + "="*50)
        print("FEDERATED LEARNING - FedProx")
        print("="*50)
        
        # Train FedProx model
        fedprox_model, fedprox_history = train_fedprox_model(
            X_train, y_train,
            X_test, y_test,
            n_clients=5,
            n_rounds=10,
            local_epochs=5,
            mu=0.01
        )
        
        # Generate FedProx visualizations
        plot_metrics(fedprox_history, mode='fedprox')
        
        # Get predictions for FedProx model
        fedprox_pred = fedprox_model.global_model.predict(X_test, verbose=0)
        fedprox_metrics = calculate_metrics(y_test, fedprox_pred)
        
        print("\nFedProx Final Metrics:")
        print("-" * 40)
        print(f"Accuracy  : {fedprox_metrics['accuracy']:.4f}")
        print(f"Precision : {fedprox_metrics['precision']:.4f}")
        print(f"Recall    : {fedprox_metrics['recall']:.4f}")
        print(f"F1        : {fedprox_metrics['f1']:.4f}")
        
        # Plot confusion matrix for FedProx
        plot_confusion_matrix(y_test, fedprox_pred, 
                            mode='fedprox', num_classes=num_classes)

        #========== FEDERATED LEARNING - FedMA ==========
        print("\n" + "="*50)
        print("FEDERATED LEARNING - FedMA")
        print("="*50)
        
        # Train FedMA model
        fedma_model, fedma_history = train_fedma_model(
            X_train, y_train,
            X_test, y_test,
            n_clients=5,
            n_rounds=10,
            local_epochs=5,
            sigma=1.0,
            sigma0=1.0,
            gamma=1.0
        )
        
        # Generate FedMA visualizations
        plot_metrics(fedma_history, mode='fedma')
        
        # Get predictions for FedMA model
        fedma_pred = fedma_model.global_model.predict(X_test, verbose=0)
        fedma_metrics = calculate_metrics(y_test, fedma_pred)
        
        print("\nFedMA Final Metrics:")
        print("-" * 40)
        print(f"Accuracy  : {fedma_metrics['accuracy']:.4f}")
        print(f"Precision : {fedma_metrics['precision']:.4f}")
        print(f"Recall    : {fedma_metrics['recall']:.4f}")
        print(f"F1        : {fedma_metrics['f1']:.4f}")
        
        # Plot confusion matrix for FedMA
        plot_confusion_matrix(y_test, fedma_pred, 
                            mode='fedma', num_classes=num_classes)
        
        #========== SAVE FINAL METRICS ==========
        print("\nSaving final metrics...")
        metrics_file = 'final_model_metrics.txt'
        
        # Clear the file if it exists
        open(metrics_file, 'w').close()
        
        # Write header
        with open(metrics_file, 'w') as f:
            f.write("=== Model Performance Comparison ===\n\n")
        
        # Save metrics for all models
        save_simple_metrics(y_test, centralized_pred, "Centralized Model", metrics_file)
        save_simple_metrics(y_test, federated_pred, "FedAvg Model", metrics_file)
        save_simple_metrics(y_test, fedprox_pred, "FedProx Model", metrics_file)
        save_simple_metrics(y_test, fedma_pred, "FedMA Model", metrics_file)
        
        print(f"\nGlobal metrics have been saved to {metrics_file}")
        print("\nTraining completed successfully!")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()