# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import math

def calculate_metrics(y_true, y_pred):
    # If one-hot encoded, convert to class indices
    if len(y_true.shape) > 1:
        y_true = y_true.argmax(axis=1) + 1  # Add 1 for 1-based indexing
        y_pred = y_pred.argmax(axis=1) + 1  # Add 1 for 1-based indexing
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_metrics(history, mode='centralized'):
    x_label = 'Epochs' if mode == 'centralized' else 'Rounds'
    
    # Get number of points
    total_points = len(history['accuracy'])
    
    # Create x-axis values
    if mode == 'centralized':
        x_values = range(1, total_points + 1)  # Start from 1 for epochs
    else:
        x_values = range(0, total_points)      # Include baseline for federated
    
    # Set interval for x-axis ticks
    interval = max(1, math.ceil(total_points / 10))
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, history['accuracy'], label="Train Accuracy")
    plt.plot(x_values, history['val_accuracy'], label="Validation Accuracy")
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    
    if mode == 'centralized':
        plt.xlim(1, total_points)
        plt.xticks(range(1, total_points + 1, interval))
    else:
        plt.xlim(0, total_points - 1)
        plt.xticks(range(0, total_points, interval))
        
    plt.title(f"{mode.capitalize()} Learning: Accuracy vs {x_label}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{mode}_accuracy_plot.png", bbox_inches='tight')
    plt.close()
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, history['loss'], label="Train Loss")
    plt.plot(x_values, history['val_loss'], label="Validation Loss")
    plt.xlabel(x_label)
    plt.ylabel("Loss")
    
    if mode == 'centralized':
        plt.xlim(1, total_points)
        plt.xticks(range(1, total_points + 1, interval))
    else:
        plt.xlim(0, total_points - 1)
        plt.xticks(range(0, total_points, interval))
        
    plt.title(f"{mode.capitalize()} Learning: Loss vs {x_label}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{mode}_loss_plot.png", bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, mode='centralized', num_classes=None):
    # If one-hot encoded, convert to 1-based class indices
    if len(y_true.shape) > 1:  # One-hot encoded
        y_true_labels = y_true.argmax(axis=1) + 1  # Add 1 for 1-based indexing
        y_pred_labels = y_pred.argmax(axis=1) + 1  # Add 1 for 1-based indexing
    else:  # Raw labels
        y_true_labels = y_true
        y_pred_labels = y_pred
    
    # Use all activity IDs if num_classes is provided, otherwise infer from data
    if num_classes is not None:
        activity_ids = np.arange(1, num_classes + 1)  # Modified to start from 1
    else:
        activity_ids = np.unique(y_true_labels)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=activity_ids)
    
    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=activity_ids, yticklabels=activity_ids)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{mode.capitalize()} Learning: Confusion Matrix")
    plt.savefig(f"{mode}_confusion_matrix.png", bbox_inches='tight')
    plt.close()

def save_simple_metrics(y_true, y_pred, model_name, file_path):
    # If one-hot encoded, convert to 1-based class indices
    if len(y_true.shape) > 1:
        y_true = y_true.argmax(axis=1) + 1
        y_pred = y_pred.argmax(axis=1) + 1
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Save metrics to file
    with open(file_path, 'a') as f:
        f.write(f"\n{model_name} Metrics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n\n")