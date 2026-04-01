"""
CNN Model for PCB Orientation Detection
Includes: model building, hyperparameter tuning with 2-fold cross-validation,
early stopping, learning rate reduction, and comprehensive evaluation metrics
"""

import os
import sys
import time
import gc
import json
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers, models, optimizers, callbacks  # type: ignore
except ImportError:
    print("WARNING: TensorFlow not installed. Install with: pip install tensorflow")
    # Fallback stubs for type hints
    tf = None
    keras = None
    layers = None
    models = None
    optimizers = None
    callbacks = None

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score
)
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-blocking backend
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "Data/Processed_data"
MODEL_SAVE_DIR = "Export"
MODEL_NAME = "ot_model.keras"

IMG_SIZE = 240
BATCH_SIZE = 32
RANDOM_STATE = 42
EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8

CLASS_LABELS = {0: "Fail", 1: "Pass"}

# Hyperparameter configurations for tuning
HYPERPARAMETER_CONFIGS = [
    {'filters_base': 32, 'dropout': 0.25, 'learning_rate': 0.0005, 'batch_size': 32},
    {'filters_base': 64, 'dropout': 0.3, 'learning_rate': 0.001, 'batch_size': 32},
    {'filters_base': 32, 'dropout': 0.2, 'learning_rate': 0.0001, 'batch_size': 16},
    {'filters_base': 48, 'dropout': 0.25, 'learning_rate': 0.0005, 'batch_size': 32},
    {'filters_base': 64, 'dropout': 0.25, 'learning_rate': 0.0005, 'batch_size': 48},
]


# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
def get_image_paths(data_dir=DATA_DIR):
    """Get all image paths with labels."""
    image_paths = []
    labels = []
    
    print("Scanning data directory...")
    
    # Scan Pass data (label 1)
    pass_dir = os.path.join(data_dir, "Pass_data")
    if os.path.exists(pass_dir):
        pass_files = [f for f in os.listdir(pass_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(pass_files)} Pass images")
        for img_file in pass_files:
            img_path = os.path.join(pass_dir, img_file)
            image_paths.append(img_path)
            labels.append(1)
    
    # Scan Fail data (label 0)
    fail_dir = os.path.join(data_dir, "Fail_data")
    if os.path.exists(fail_dir):
        fail_files = [f for f in os.listdir(fail_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(fail_files)} Fail images")
        for img_file in fail_files:
            img_path = os.path.join(fail_dir, img_file)
            image_paths.append(img_path)
            labels.append(0)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Total images found: {len(image_paths)}")
    print(f"Pass: {sum(1 for l in labels if l == 1)}, Fail: {sum(1 for l in labels if l == 0)}")
    
    return image_paths, labels


def load_and_preprocess_image(image_path):
    """Load and preprocess a single image."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize to standard size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Convert to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def load_dataset(image_paths, labels):
    """Load all images and labels into memory."""
    images = []
    valid_labels = []
    failed_count = 0
    
    print("\nLoading images...")
    for idx, img_path in enumerate(tqdm(image_paths, desc="Loading")):
        img = load_and_preprocess_image(img_path)
        if img is not None:
            images.append(img)
            valid_labels.append(labels[idx])
        else:
            failed_count += 1
    
    if len(images) == 0:
        raise ValueError("No images could be loaded")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(valid_labels, dtype=np.int32)
    
    print(f"✓ Loaded {len(images)} images (Failed: {failed_count})")
    print(f"✓ Images shape: {images.shape}")
    print(f"✓ Labels distribution: {np.bincount(labels)}")
    
    return images, labels


# ==========================================
# EARLY STOPPING CALLBACK FOR 90% ACCURACY
# ==========================================
class AccuracyThresholdCallback(callbacks.Callback):
    """Stop training when validation accuracy reaches threshold."""
    def __init__(self, threshold=0.90, verbose=1):
        super().__init__()
        self.threshold = threshold
        self.verbose = verbose
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None and val_accuracy >= self.threshold:
            if self.verbose > 0:
                print(f"\n✓ Validation accuracy reached {val_accuracy:.4f} (target: {self.threshold:.4f})")
                print("✓ Stopping training early!")
            self.model.stop_training = True


# ==========================================
# MODEL BUILDING
# ==========================================
def build_cnn_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters_base=32, dropout=0.25, learning_rate=0.0005):
    """Build CNN model with specified hyperparameters."""
    model = models.Sequential([
        # Block 1
        layers.Conv2D(filters_base, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout),
        
        # Block 2
        layers.Conv2D(filters_base * 2, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout),
        
        # Block 3
        layers.Conv2D(filters_base * 4, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout),
        
        # Flatten and Dense
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        
        # Output
        layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


# ==========================================
# TRAINING & EVALUATION
# ==========================================
def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train model with callbacks."""
    # Create callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
    
    # Add accuracy threshold callback
    accuracy_threshold = AccuracyThresholdCallback(threshold=0.90, verbose=1)
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr, accuracy_threshold],
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test, dataset_name="Test"):
    """Evaluate model on test set."""
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} SET EVALUATION")
    print(f"{'='*60}")
    
    # Predict
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_test_flat = y_test.flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_flat, y_pred)
    precision = precision_score(y_test_flat, y_pred, zero_division=0)
    recall = recall_score(y_test_flat, y_pred, zero_division=0)
    f1 = f1_score(y_test_flat, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test_flat, y_pred_proba)
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(y_test_flat, y_pred, target_names=[CLASS_LABELS[0], CLASS_LABELS[1]]))
    
    cm = confusion_matrix(y_test_flat, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "y_true": y_test_flat
    }
    
    return metrics


# ==========================================
# VISUALIZATION
# ==========================================
def plot_confusion_matrix(cm, dataset_name="Test"):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]],
                yticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]])
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    filepath = os.path.join(MODEL_SAVE_DIR, f'cnn_confusion_matrix_{dataset_name.lower()}.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {filepath}")


def plot_training_history(history, fold_num=1):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'Model Accuracy - Fold {fold_num}')
    axes[0].legend()
    axes[0].grid()
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'Model Loss - Fold {fold_num}')
    axes[1].legend()
    axes[1].grid()
    
    plt.tight_layout()
    filepath = os.path.join(MODEL_SAVE_DIR, f'cnn_training_history_fold{fold_num}.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history saved to {filepath}")


def save_hyperparameter_results(all_results, best_config, best_fold, model_save_dir=MODEL_SAVE_DIR):
    """Save hyperparameter tuning results to a markdown file."""
    try:
        results_file = os.path.join(model_save_dir, "CNN_HYPERPARAMETER_TUNING_RESULTS.md")
        
        with open(results_file, 'w') as f:
            f.write("# CNN Hyperparameter Tuning Results\n\n")
            f.write("## Best Configuration\n\n")
            f.write(f"- **Filters Base**: {best_config['filters_base']}\n")
            f.write(f"- **Dropout**: {best_config['dropout']}\n")
            f.write(f"- **Learning Rate**: {best_config['learning_rate']}\n")
            f.write(f"- **Batch Size**: {best_config['batch_size']}\n")
            f.write(f"- **Best Fold**: {best_fold}\n\n")
            
            f.write("## All Configuration Results\n\n")
            
            # Group results by fold
            for fold_num in [1, 2]:
                fold_results = [r for r in all_results if r['fold'] == fold_num]
                
                if fold_results:
                    f.write(f"### Fold {fold_num}\n\n")
                    f.write("| Config | Accuracy | Precision | Recall | F1-Score |\n")
                    f.write("|--------|----------|-----------|--------|----------|\n")
                    
                    for result in sorted(fold_results, key=lambda x: x['config_idx']):
                        config_idx = result['config_idx']
                        accuracy = result['accuracy']
                        precision = result['precision']
                        recall = result['recall']
                        f1 = result['f1_score']
                        f.write(f"| {config_idx} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |\n")
                    f.write("\n")
            
            f.write("## Configurations Tested\n\n")
            for config_idx, config in enumerate(HYPERPARAMETER_CONFIGS, 1):
                f.write(f"**Config {config_idx}**:\n")
                f.write(f"- Filters Base: {config['filters_base']}\n")
                f.write(f"- Dropout: {config['dropout']}\n")
                f.write(f"- Learning Rate: {config['learning_rate']}\n")
                f.write(f"- Batch Size: {config['batch_size']}\n\n")
        
        print(f"✓ Hyperparameter results saved to {results_file}")
        return True
        
    except Exception as e:
        print(f"✗ ERROR saving results: {type(e).__name__}: {e}")
        return False


def save_best_hyperparameters(config, metrics, fold, model_save_dir=MODEL_SAVE_DIR):
    """Save best hyperparameters and metrics to JSON file for reuse."""
    try:
        os.makedirs(model_save_dir, exist_ok=True)
        params_file = os.path.join(model_save_dir, "best_hyperparameters.json")
        
        params_data = {
            "hyperparameters": config,
            "best_fold": fold,
            "metrics": {
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "roc_auc": float(metrics["roc_auc"])
            },
            "model_settings": {
                "img_size": IMG_SIZE,
                "batch_size": config["batch_size"],
                "epochs": EPOCHS
            }
        }
        
        with open(params_file, 'w') as f:
            json.dump(params_data, f, indent=2)
        
        print(f"✓ Best hyperparameters saved to {params_file}")
        return True
        
    except Exception as e:
        print(f"✗ ERROR saving hyperparameters: {type(e).__name__}: {e}")
        return False


# ==========================================
# MODEL PERSISTENCE
# ==========================================
def save_model(model, model_dir=MODEL_SAVE_DIR):
    """Save trained model with error handling and verification."""
    try:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, MODEL_NAME)
        
        # Validate model is not None
        if model is None:
            raise ValueError("Model is None - cannot save")
        
        # Save model
        print(f"Saving model to {model_path}...")
        model.save(model_path)
        
        # Verify file was created
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file was not created at {model_path}")
        
        # Check file size
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            raise ValueError(f"Model file is empty (0 bytes) at {model_path}")
        
        print(f"✓ Model saved successfully: {model_path}")
        print(f"  File size: {file_size / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        print(f"✗ ERROR saving model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==========================================
# MAIN TRAINING PIPELINE WITH 2-FOLD CV
# ==========================================
def main():
    """Main training pipeline with 2-fold cross-validation and hyperparameter tuning."""
    print("\n" + "="*70)
    print("CNN MODEL FOR PCB ORIENTATION DETECTION")
    print("WITH 2-FOLD CROSS-VALIDATION & HYPERPARAMETER TUNING")
    print("="*70)
    print("*** CONSOLIDATED TRAINER WITH HYPERPARAMETER OPTIMIZATION ***\n")
    
    try:
        start_time = time.time()
        
        # STEP 1: Load data
        print("STEP 1: Loading Dataset")
        print("-"*70)
        step_start = time.time()
        image_paths, labels = get_image_paths(DATA_DIR)
        images, labels = load_dataset(image_paths, labels)
        data_load_time = time.time() - step_start
        print(f"✓ Dataset loaded in {data_load_time:.1f}s\n")
        
        # Force garbage collection
        gc.collect()
        
        # STEP 2: 2-FOLD STRATIFIED CROSS-VALIDATION WITH HYPERPARAMETER TUNING
        print("STEP 2: 2-FOLD STRATIFIED CROSS-VALIDATION WITH HYPERPARAMETER TUNING")
        print("-"*70)
        
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
        fold_results = []
        best_model = None
        best_overall_score = -1
        best_fold = -1
        best_config = None
        
        fold_num = 0
        for train_idx, test_idx in skf.split(images, labels):
            fold_num += 1
            print(f"\n{'='*70}")
            print(f"FOLD {fold_num}/2")
            print(f"{'='*70}")
            
            # Split data for this fold
            X_fold_train_all = images[train_idx].astype(np.float32)
            y_fold_train_all = labels[train_idx]
            X_fold_test = images[test_idx].astype(np.float32)
            y_fold_test = labels[test_idx]
            
            # Further split training into train/val
            X_fold_train, X_fold_val, y_fold_train, y_fold_val = train_test_split(
                X_fold_train_all, y_fold_train_all, test_size=0.2,
                random_state=RANDOM_STATE, stratify=y_fold_train_all
            )
            
            print(f"Train/Val/Test split: {len(X_fold_train)} / {len(X_fold_val)} / {len(X_fold_test)} (stratified)")
            print(f"\nHyperparameter Search: {len(HYPERPARAMETER_CONFIGS)} configurations:")
            
            fold_config_results = []
            
            # Try each hyperparameter configuration
            for config_idx, config in enumerate(HYPERPARAMETER_CONFIGS, 1):
                print(f"\n  Configuration {config_idx}/{len(HYPERPARAMETER_CONFIGS)}: {config}")
                
                try:
                    # Build model with this config
                    model = build_cnn_model(
                        input_shape=(IMG_SIZE, IMG_SIZE, 3),
                        filters_base=config['filters_base'],
                        dropout=config['dropout'],
                        learning_rate=config['learning_rate']
                    )
                    
                    # Train model
                    print(f"  Training with batch_size={config['batch_size']}...")
                    history = train_model(
                        model, X_fold_train, y_fold_train,
                        X_fold_val, y_fold_val,
                        epochs=EPOCHS,
                        batch_size=config['batch_size']
                    )
                    
                    # Evaluate on test set
                    config_metrics = evaluate_model(model, X_fold_test, y_fold_test, 
                                                   f"Fold {fold_num} Config {config_idx}")
                    
                    fold_config_results.append({
                        'config': config,
                        'config_idx': config_idx,
                        'model': model,
                        'history': history,
                        'metrics': config_metrics,
                        'f1_score': config_metrics['f1'],
                        'accuracy': config_metrics['accuracy']
                    })
                    
                    print(f"  ✓ F1-Score: {config_metrics['f1']:.4f}, Accuracy: {config_metrics['accuracy']:.4f}")
                    
                    # Clear model to free memory
                    del model
                    gc.collect()
                    
                except Exception as e:
                    print(f"  ✗ Error in configuration {config_idx}: {e}")
                    continue
            
            if not fold_config_results:
                print(f"✗ No successful configurations in fold {fold_num}")
                continue
            
            # Select best config for this fold
            best_config_in_fold = max(fold_config_results, key=lambda x: x['f1_score'])
            
            fold_result = {
                'fold': fold_num,
                'best_config': best_config_in_fold['config'],
                'best_config_idx': best_config_in_fold['config_idx'],
                'model': best_config_in_fold['model'],
                'metrics': best_config_in_fold['metrics'],
                'f1_score': best_config_in_fold['f1_score'],
                'accuracy': best_config_in_fold['accuracy'],
                'all_config_results': fold_config_results  # Store all results for documentation
            }
            fold_results.append(fold_result)
            
            # Update best overall model
            if best_config_in_fold['f1_score'] > best_overall_score:
                best_overall_score = best_config_in_fold['f1_score']
                best_model = best_config_in_fold['model']
                best_fold = fold_num
                best_config = best_config_in_fold['config']
            
            print(f"\n✓ Fold {fold_num} Best Config: {best_config_in_fold['config_idx']}, F1: {best_config_in_fold['f1_score']:.4f}")
            
            gc.collect()
        
        # STEP 3: Report Results
        print(f"\n{'='*70}")
        print("2-FOLD CROSS-VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        all_accuracies = [r['accuracy'] for r in fold_results]
        all_f1_scores = [r['f1_score'] for r in fold_results]
        
        print(f"\nFold 1 - Config {fold_results[0]['best_config_idx']}: Accuracy {fold_results[0]['accuracy']:.4f}, F1 {fold_results[0]['f1_score']:.4f}")
        print(f"Fold 2 - Config {fold_results[1]['best_config_idx']}: Accuracy {fold_results[1]['accuracy']:.4f}, F1 {fold_results[1]['f1_score']:.4f}")
        print(f"\nMean Accuracy: {np.mean(all_accuracies):.4f} (+/- {np.std(all_accuracies):.4f})")
        print(f"Mean F1-Score: {np.mean(all_f1_scores):.4f} (+/- {np.std(all_f1_scores):.4f})")
        print(f"\n✓ BEST MODEL: Fold {best_fold}, Config {best_config['filters_base']}-{best_config['dropout']}-{best_config['learning_rate']}")
        print(f"  Best F1-Score: {best_overall_score:.4f}")
        
        # STEP 4: Visualizations
        print(f"\n{'='*70}")
        print("STEP 3: Generating Visualizations (Best Fold)")
        print(f"{'='*70}")
        
        best_metrics = fold_results[best_fold - 1]['metrics']
        plot_confusion_matrix(best_metrics['confusion_matrix'], f"Fold{best_fold}")
        
        # STEP 5: Collect ALL Hyperparameter Results for Documentation
        print(f"\n{'='*70}")
        print("STEP 4: Collecting All Hyperparameter Results")
        print(f"{'='*70}")
        
        all_results = []
        for fold_idx, fold_result in enumerate(fold_results, 1):
            print(f"\nFold {fold_idx} Results:")
            print(f"{'  Config':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
            print(f"{'  '+'-'*8:<10} {'-'*10:<12} {'-'*10:<12} {'-'*10:<12} {'-'*10:<12}")
            
            # Collect results for this fold (from all configs tested)
            for config_result in fold_results[fold_idx-1].get('all_config_results', []):
                config_idx = config_result.get('config_idx', 'N/A')
                metrics = config_result.get('metrics', {})
                accuracy = metrics.get('accuracy', 0)
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1', 0)
                
                all_results.append({
                    'fold': fold_idx,
                    'config_idx': config_idx,
                    'config': config_result.get('config', {}),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
                
                print(f"  Config {config_idx:<3} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        # Summary
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("✓✓✓ TRAINING WITH 2-FOLD CV COMPLETED SUCCESSFULLY ✓✓✓")
        print("="*70)
        print(f"\nBEST MODEL PERFORMANCE (from Fold {best_fold}):")
        print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall:    {best_metrics['recall']:.4f}")
        print(f"  F1-Score:  {best_metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {best_metrics['roc_auc']:.4f}")
        
        print(f"\nBEST HYPERPARAMETERS:")
        print(f"  Filters Base: {best_config['filters_base']}")
        print(f"  Dropout: {best_config['dropout']}")
        print(f"  Learning Rate: {best_config['learning_rate']}")
        print(f"  Batch Size: {best_config['batch_size']}")
        
        print(f"\nCROSS-VALIDATION STATISTICS:")
        print(f"  Mean Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
        print(f"  Mean F1-Score: {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}")
        
        print(f"\nTiming: Total {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print("="*70 + "\n")
        
        # STEP 6: Save best model with verification
        print("STEP 5: Saving Best Model")
        print("-"*70)
        
        if best_model is not None:
            save_success = save_model(best_model)
            if save_success:
                print("✓ Model successfully saved to Export folder")
            else:
                print("✗ FAILED to save model - see errors above")
                sys.exit(1)
        else:
            print("✗ ERROR: best_model is None - cannot save")
            sys.exit(1)
        
        # STEP 7: Save hyperparameter results to markdown
        print("\nSTEP 6: Saving Hyperparameter Results")
        print("-"*70)
        save_hyperparameter_results(all_results, best_config, best_fold)
        
        # STEP 8: Save best hyperparameters to JSON for reuse
        print("\nSTEP 7: Saving Best Hyperparameters for Final Training")
        print("-"*70)
        save_best_hyperparameters(best_config, best_metrics, best_fold)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗✗✗ CRITICAL ERROR ✗✗✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
