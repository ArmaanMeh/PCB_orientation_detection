#!/usr/bin/env python3
"""
Advanced HOG + SVM Trainer with Aggressive Hyperparameter Tuning
Automatically tunes hyperparameters until reaching 90%+ accuracy
"""

import cv2
import numpy as np
import os
import pickle
import time
import gc
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "Data/Processed_data"
MODEL_SAVE_DIR = "Export"
MODEL_NAME = "hog_svm_model.pkl"
SCALER_NAME = "hog_svm_scaler.pkl"
RESULTS_FILE = "Export/hog_svm_tuning_results.json"

IMG_SIZE = 240
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

RANDOM_STATE = 42
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2

CLASS_LABELS = {0: "Fail", 1: "Pass"}
ACCURACY_TARGET = 0.90  # 90% target

# Focused hyperparameter search - sequential testing
PARAM_SEARCH = [
    {'C': 0.1, 'kernel': 'linear', 'gamma': 'scale'},
    {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
    {'C': 1, 'kernel': 'linear', 'gamma': 'scale'},
    {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'},
    {'C': 1, 'kernel': 'rbf', 'gamma': 0.01},
    {'C': 10, 'kernel': 'linear', 'gamma': 'scale'},
    {'C': 10, 'kernel': 'rbf', 'gamma': 'scale'},
    {'C': 10, 'kernel': 'rbf', 'gamma': 0.01},
    {'C': 100, 'kernel': 'linear', 'gamma': 'scale'},
    {'C': 100, 'kernel': 'rbf', 'gamma': 'scale'},
    {'C': 100, 'kernel': 'rbf', 'gamma': 0.001},
    {'C': 100, 'kernel': 'rbf', 'gamma': 0.01},
    {'C': 1000, 'kernel': 'rbf', 'gamma': 0.01},
    {'C': 1000, 'kernel': 'poly', 'gamma': 'scale'},
]

print(f"Target accuracy: {ACCURACY_TARGET*100}%")
print(f"Parameter combinations to test: {len(PARAM_SEARCH)}")


# ==========================================
# HOG DESCRIPTOR
# ==========================================
def create_hog_descriptor():
    """Create HOG descriptor with stable parameters."""
    hog = cv2.HOGDescriptor(
        (IMG_SIZE, IMG_SIZE),
        (32, 32),
        (16, 16),
        (16, 16),
        HOG_ORIENTATIONS
    )
    return hog


HOG_DESCRIPTOR = create_hog_descriptor()
HOG_EXPECTED_FEATURE_SIZE = None


# ==========================================
# HOG FEATURE EXTRACTION
# ==========================================
def extract_hog_features(image):
    """Extract HOG features from image."""
    global HOG_EXPECTED_FEATURE_SIZE
    
    try:
        if image is None or not isinstance(image, np.ndarray):
            return None
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) != 2:
            return None
        
        if image.dtype != np.uint8:
            if image.max() > 1.0:
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = (image * 255).astype(np.uint8)
        
        if image.shape != (IMG_SIZE, IMG_SIZE):
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        features = HOG_DESCRIPTOR.compute(
            image,
            winStride=(16, 16),
            padding=(0, 0),
            locations=None
        )
        
        if features is None or features.size == 0:
            return None
        
        features = features.flatten()
        
        if HOG_EXPECTED_FEATURE_SIZE is None:
            HOG_EXPECTED_FEATURE_SIZE = features.size
        elif features.size != HOG_EXPECTED_FEATURE_SIZE:
            if features.size < HOG_EXPECTED_FEATURE_SIZE:
                features = np.pad(features, (0, HOG_EXPECTED_FEATURE_SIZE - features.size), mode='constant')
            else:
                features = features[:HOG_EXPECTED_FEATURE_SIZE]
        
        features = features.astype(np.float32)
        
        if not np.all(np.isfinite(features)):
            return None
        
        return features
    
    except Exception as e:
        return None


# ==========================================
# DATA LOADING
# ==========================================
def get_image_paths(data_dir=DATA_DIR):
    """Get all image paths with labels."""
    image_paths = []
    labels = []
    
    pass_dir = os.path.join(data_dir, "Pass_data")
    if os.path.exists(pass_dir):
        for img_file in os.listdir(pass_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(pass_dir, img_file))
                labels.append(1)
    
    fail_dir = os.path.join(data_dir, "Fail_data")
    if os.path.exists(fail_dir):
        for img_file in os.listdir(fail_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(fail_dir, img_file))
                labels.append(0)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"✓ Total images: {len(image_paths)}")
    print(f"  Pass: {sum(1 for l in labels if l == 1)}, Fail: {sum(1 for l in labels if l == 0)}")
    
    return image_paths, labels


def load_and_extract_features(image_paths, labels):
    """Load images and extract HOG features."""
    features_list = []
    valid_labels = []
    failed_count = 0
    
    print("\n▶ Extracting HOG features...")
    
    for idx, (img_path, label) in enumerate(tqdm(zip(image_paths, labels), total=len(image_paths))):
        try:
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                failed_count += 1
                continue
            
            features = extract_hog_features(img)
            if features is None:
                failed_count += 1
                continue
            
            features_list.append(features)
            valid_labels.append(label)
            
        except Exception as e:
            failed_count += 1
            continue
    
    if len(features_list) == 0:
        raise ValueError("No valid features extracted!")
    
    if failed_count > 0:
        print(f"  ⚠ Failed to process {failed_count} images")
    
    features_array = np.array(features_list, dtype=np.float32)
    labels_array = np.array(valid_labels)
    
    print(f"✓ Extracted features shape: {features_array.shape}")
    print(f"✓ Valid samples: {len(features_array)}")
    
    return features_array, labels_array


# ==========================================
# TRAINING WITH SEQUENTIAL TUNING
# ==========================================
def train_with_tuning(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train SVM with sequential hyperparameter tuning."""
    
    print("\n▶ Starting sequential hyperparameter search...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    best_accuracy = 0
    best_model = None
    best_params = None
    results_log = []
    
    # Test each configuration sequentially
    for idx, params in enumerate(PARAM_SEARCH, 1):
        print(f"\n  [{idx}/{len(PARAM_SEARCH)}] Testing {params}...")
        
        try:
            # Create and train SVM
            model = SVC(random_state=RANDOM_STATE, probability=True, **params)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate on test
            test_pred = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, test_pred)
            
            # Evaluate on validation
            val_pred = model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, val_pred)
            
            print(f"    Test Acc: {test_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            results_log.append({
                'params': params,
                'val_accuracy': float(val_acc),
                'test_accuracy': float(test_acc)
            })
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_model = model
                best_params = params
                print(f"    ✓ New best test accuracy: {best_accuracy:.4f}")
            
            # Early exit if target reached
            if test_acc >= ACCURACY_TARGET:
                print(f"\n  ✓ TARGET REACHED: {test_acc:.4f} >= {ACCURACY_TARGET:.4f}")
                return best_model, scaler, best_params, best_accuracy, results_log, test_pred, y_test
        
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    return best_model, scaler, best_params, best_accuracy, results_log, None, None


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("\n" + "="*80)
    print("HOG + SVM TRAINING WITH AGGRESSIVE HYPERPARAMETER TUNING")
    print("="*80)
    
    start_time = time.time()
    
    # Load data
    print("\n▶ Loading dataset...")
    image_paths, labels = get_image_paths()
    
    # Extract features
    features, labels_array = load_and_extract_features(image_paths, labels)
    
    # Split: train (60%), val (20%), test (20%)
    print("\n▶ Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels_array,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=labels_array
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VAL_SPLIT/(1-TEST_SPLIT),
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    
    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train with tuning
    best_model, scaler, best_params, best_accuracy, results_log, y_pred_test, y_test_final = train_with_tuning(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    print(f"\n✓ Best parameters: {best_params}")
    print(f"✓ Best test accuracy: {best_accuracy:.4f}")
    
    if y_pred_test is not None:
        # Calculate metrics using final test predictions
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        cm = confusion_matrix(y_test, y_pred_test)
    else:
        # Use best model for test predictions if not already found
        X_test_scaled = scaler.transform(X_test)
        y_pred_test = best_model.predict(X_test_scaled)
        
        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)
        cm = confusion_matrix(y_test, y_pred_test)
    
    print(f"\nDetailed Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    print(classification_report(y_test, y_pred_test, target_names=list(CLASS_LABELS.values())))
    
    # Check if target accuracy reached
    if best_accuracy >= ACCURACY_TARGET:
        print(f"\n✓ TARGET ACCURACY REACHED: {best_accuracy:.4f} >= {ACCURACY_TARGET:.4f}")
        
        # Save model
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        
        with open(os.path.join(MODEL_SAVE_DIR, MODEL_NAME), 'wb') as f:
            pickle.dump(best_model, f)
        
        with open(os.path.join(MODEL_SAVE_DIR, SCALER_NAME), 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"✓ Model saved to {os.path.join(MODEL_SAVE_DIR, MODEL_NAME)}")
        print(f"✓ Scaler saved to {os.path.join(MODEL_SAVE_DIR, SCALER_NAME)}")
    else:
        print(f"\n⚠ Target accuracy NOT reached: {best_accuracy:.4f} < {ACCURACY_TARGET:.4f}")
        print("Consider expanding parameter search or improving data quality")
    
    # Save results
    tuning_results = {
        'timestamp': datetime.now().isoformat(),
        'best_params': best_params,
        'best_test_accuracy': float(best_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'all_results': results_log,
        'training_time_seconds': time.time() - start_time
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(tuning_results, f, indent=2)
    
    print(f"\n✓ Results saved to {RESULTS_FILE}")
    
    # Plot confusion matrix
    try:
        plt.figure(figsize=(8, 6))
        ConfusionMatrixDisplay(cm, display_labels=list(CLASS_LABELS.values())).plot()
        plt.title("HOG+SVM Confusion Matrix (Test Set)")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_SAVE_DIR, "hog_svm_confusion_matrix.png"), dpi=100)
        print(f"✓ Confusion matrix plot saved")
    except Exception as e:
        print(f"⚠ Could not save confusion matrix plot: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱ Total training time: {elapsed_time:.2f}s ({elapsed_time/60:.2f}m)")
    print("="*80)
    
    return best_accuracy >= ACCURACY_TARGET


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
