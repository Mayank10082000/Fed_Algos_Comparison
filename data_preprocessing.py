# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)
    
    # Balance classes through undersampling
    min_class_count = data['activity_id'].value_counts().min()
    balanced_data = pd.DataFrame()
    
    for activity in data['activity_id'].unique():
        activity_data = data[data['activity_id'] == activity]
        balanced_sample = activity_data.sample(n=min_class_count, random_state=42)
        balanced_data = pd.concat([balanced_data, balanced_sample])
    
    # Shuffle the balanced dataset
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split features and target
    X = balanced_data.drop(['activity_id', 'activity_name'], axis=1)
    y = balanced_data['activity_id']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    
    # Extract sequences
    for i in range(len(sequences) - n_steps + 1):
        # Get the sequence window
        seq_x = sequences[i:i + n_steps, :-1]  # All columns except last (label)
        seq_y = sequences[i + n_steps - 1, -1]  # Last column of last timestep
        
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

def prepare_sequences(X_scaled, y, sequence_length=128):
    # Combine scaled features and labels
    combined_data = np.column_stack([X_scaled, y])
    
    # Create sequences
    X_seq, y_seq = split_sequences(combined_data, sequence_length)
    
    # Convert labels to one-hot encoding
    # Subtract 1 temporarily for zero-based one-hot encoding
    y_seq_zero_based = y_seq - 1
    n_classes = len(np.unique(y))
    y_seq = to_categorical(y_seq_zero_based, num_classes=n_classes)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test