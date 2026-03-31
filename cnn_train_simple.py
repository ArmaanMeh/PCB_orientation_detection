#!/usr/bin/env python3
"""
Fast Single-Model CNN Trainer
Trains one well-chosen network architecture quickly to reach 90%+ accuracy
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

# Configuration
DATA_DIR = "Data/Processed_data"
MODEL_SAVE_DIR = "Export"
MODEL_NAME = "ot_model.keras"
RESULTS_FILE = "Export/cnn_tuning_results.json"

IMG_SIZE = 244
BATCH_SIZE = 32
EPOCHS = 40
ACCURACY_TARGET = 0.90
NUM_CLASSES = 2
CLASS_NAMES = ["Fail", "Pass"]

#Set seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print(f"Target accuracy: {ACCURACY_TARGET*100}%\n")

# ==========================================
# DATA LOADING
# ==========================================
print("=" * 80)
print("CNN SINGLE MODEL TRAINING")
print("=" * 80)

print("\n>>> Loading datasets...")

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
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

# Cache and prefetch
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

print("OK - Datasets loaded\n")

# ==========================================
# BUILD MODEL
# ==========================================
print(">>> Building model...")

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Conv block 1
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),
    
    # Conv block 2
    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),
    
    # Conv block 3
    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),
    layers.Dropout(0.25),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("OK - Model created\n")
print(model.summary())

# ==========================================
# TRAIN MODEL
# ==========================================
print("\n>>> Training model...")

start_time = time.time()

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time

# ==========================================
# EVALUATE
# ==========================================
print("\n>>> Evaluating model...")

val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)

# Get predictions
predictions = []
true_labels = []

for X, y in val_ds:
    preds = model.predict(X, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    predictions.extend(pred_labels)
    true_labels.extend(y.numpy())

y_true = np.array(true_labels)
y_pred = np.array(predictions)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print(f"\nFinal Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print(f"\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

# ==========================================
# SAVE MODEL
# ==========================================
if accuracy >= ACCURACY_TARGET:
    print(f"\n*** SUCCESS *** Target accuracy reached: {accuracy:.4f} >= {ACCURACY_TARGET:.4f}")
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    model.save(model_path)
    print(f"Model saved to {model_path}")
else:
    print(f"\nWARNING: Target accuracy NOT reached: {accuracy:.4f} < {ACCURACY_TARGET:.4f}")
    print("Training more epochs or adjusting hyperparameters may help")

# ==========================================
# SAVE RESULTS
# ==========================================
results = {
    'timestamp': datetime.now().isoformat(),
    'model_architecture': 'Conv2D(32)-Conv2D(64)-Conv2D(128)-Dense(128)',
    'batch_size': BATCH_SIZE,
    'learning_rate': 0.0005,
    'dropout': 0.25,
    'val_accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'confusion_matrix': cm.tolist(),
    'training_time_seconds': training_time
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {RESULTS_FILE}")

# ==========================================
# plot confusion matrix
# ==========================================
try:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("CNN Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "cnn_confusion_matrix.png"), dpi=100)
    print("Confusion matrix plot saved")
except Exception as e:
    print(f"Could not save confusion matrix: {e}")

print(f"\nTotal training time: {training_time:.2f}s ({training_time/60:.2f}m)")
print("=" * 80)

exit(0 if accuracy >= ACCURACY_TARGET else 1)
