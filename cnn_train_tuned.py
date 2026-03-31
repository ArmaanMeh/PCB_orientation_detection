#!/usr/bin/env python3
"""
Efficient CNN Trainer with Smart Hyperparameter Tuning
Quick convergence to 90%+ accuracy without exhaustive search
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from datetime import datetime
import random

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "Data/Processed_data"
MODEL_SAVE_DIR = "Export"
MODEL_NAME = "ot_model.keras"
RESULTS_FILE = "Export/cnn_tuning_results.json"

IMG_SIZE = 244
BATCH_SIZE = 32  # Use fixed good batch size
EPOCHS = 30  # Reduced epochs with early stopping

ACCURACY_TARGET = 0.90
NUM_CLASSES = 2
CLASS_NAMES = ["Fail", "Pass"]

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Focused hyperparameter grid - minimal but effective
HYPERPARAMETERS = [
    {'filters': [32, 64, 128], 'dense': 128, 'dropout': 0.3, 'lr': 0.001},
    {'filters': [64, 128, 256], 'dense': 128, 'dropout': 0.3, 'lr': 0.0005},
    {'filters': [32, 64, 128], 'dense': 256, 'dropout': 0.3, 'lr': 0.0005},
    {'filters': [64, 128, 256], 'dense': 256, 'dropout': 0.2, 'lr': 0.001},
    {'filters': [32, 64, 128], 'dense': 128, 'dropout': 0.2, 'lr': 0.0005},
]

print(f"Target accuracy: {ACCURACY_TARGET*100}%")
print(f"Hyperparameter configurations to test: {len(HYPERPARAMETERS)}")


# ==========================================
# MODEL BUILDER
# ==========================================
def build_cnn_model(filters_config, dense_units, dropout_rate, learning_rate):
    """Build CNN model with specified hyperparameters."""
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1./255),
        
        # First conv block
        layers.Conv2D(filters_config[0], 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        # Second conv block
        layers.Conv2D(filters_config[1], 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        # Third conv block
        layers.Conv2D(filters_config[2], 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(dropout_rate),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ==========================================
# DATA LOADING
# ==========================================
def load_datasets():
    """Load and prepare datasets."""
    print("\n▶ Loading datasets...")
    
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int'
    )
    
    # Normalize
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Cache and prefetch
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    print(f"✓ Training and validation datasets loaded")
    
    return train_ds, val_ds


# ==========================================
# TRAINING
# ==========================================
def train_model(model, train_ds, val_ds):
    """Train model with early stopping."""
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-7, verbose=0)
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0
    )
    
    return history


# ==========================================
# HYPERPARAMETER SEARCH
# ==========================================
def search_hyperparameters(train_ds, val_ds):
    """Search through hyperparameters efficiently."""
    
    print("\n▶ Starting hyperparameter search...")
    
    results_list = []
    best_accuracy = 0
    best_config = None
    best_model = None
    
    for idx, config in enumerate(HYPERPARAMETERS, 1):
        print(f"\n[{idx}/{len(HYPERPARAMETERS)}] Testing config: Filters={config['filters']}, Dense={config['dense']}, DO={config['dropout']}, LR={config['lr']}")
        
        try:
            # Build model
            model = build_cnn_model(config['filters'], config['dense'], config['dropout'], config['lr'])
            
            # Train model
            history = train_model(model, train_ds, val_ds)
            
            # Evaluate
            val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
            
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            
            results_list.append({
                'config': config,
                'val_accuracy': float(val_accuracy)
            })
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_config = config
                best_model = model
                print(f"  ✓ New best accuracy: {best_accuracy:.4f}")
            
            # Early exit if target reached
            if val_accuracy >= ACCURACY_TARGET:
                print(f"\n✓ TARGET ACCURACY REACHED: {val_accuracy:.4f}")
                return best_model, best_config, best_accuracy, results_list
            
            # Clear memory
            tf.keras.backend.clear_session()
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            tf.keras.backend.clear_session()
    
    return best_model, best_config, best_accuracy, results_list


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("\n" + "="*80)
    print("CNN TRAINING WITH SMART HYPERPARAMETER TUNING")
    print("="*80)
    
    start_time = time.time()
    
    # Load data
    train_ds, val_ds = load_datasets()
    
    # Hyperparameter search
    best_model, best_config, best_accuracy, results_list = search_hyperparameters(train_ds, val_ds)
    
    if best_model is None:
        print("\n✗ Failed to train any model!")
        return False
    
    print(f"\n✓ Best configuration found:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"  Validation accuracy: {best_accuracy:.4f}")
    
    # Get predictions on validation set
    print("\n▶ Evaluating final model...")
    y_pred, y_true = get_all_predictions(best_model, val_ds)
    
    val_accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nValidation Set Metrics:")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # Check if target accuracy reached
    if val_accuracy >= ACCURACY_TARGET:
        print(f"\n✓ TARGET ACCURACY REACHED: {val_accuracy:.4f} >= {ACCURACY_TARGET:.4f}")
        
        # Save model
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
        best_model.save(model_path)
        print(f"✓ Model saved to {model_path}")
    else:
        print(f"\n⚠ Target accuracy NOT reached: {val_accuracy:.4f} < {ACCURACY_TARGET:.4f}")
    
    # Save tuning results
    tuning_results = {
        'timestamp': datetime.now().isoformat(),
        'best_config': best_config,
        'best_val_accuracy': float(best_accuracy),
        'final_val_accuracy': float(val_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'all_results': results_list,
        'training_time_seconds': time.time() - start_time
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    print(f"✓ Results saved to {RESULTS_FILE}")
    
    # Plot confusion matrix
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title("CNN Confusion Matrix (Validation Set)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_DIR, "cnn_confusion_matrix.png"), dpi=100)
        print(f"✓ Confusion matrix plot saved")
    except Exception as e:
        print(f"⚠ Could not save confusion matrix plot: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱ Total training time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
    print("="*80)
    
    return val_accuracy >= ACCURACY_TARGET


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
