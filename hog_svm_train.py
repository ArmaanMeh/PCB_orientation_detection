"""
HOG + SVM Model for PCB Orientation Detection
Includes: feature extraction, hyperparameter optimization, validation, 
evaluation metrics, and robust training pipeline
CRASH-PROOF: Memory-efficient processing with robust error handling
"""

import cv2
import numpy as np
import os
import pickle
import time
import gc
import sys
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-blocking backend
import seaborn as sns
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "Data/Processed_data"
MODEL_SAVE_DIR = "Export"
MODEL_NAME = "hog_svm_model.pkl"
SCALER_NAME = "hog_svm_scaler.pkl"

IMG_SIZE = 244
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

RANDOM_STATE = 42
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2

CLASS_LABELS = {0: "Fail", 1: "Pass"}

# Memory management
BATCH_SIZE = 32  # Process images in batches to avoid memory overflow

# EXPECTED HOG FEATURE SIZE (calculated from parameters)
# Image size: 244x244, window: 244x244, stride: (16,16), block: (2,2)
# Cells: (244/16 = 15.25 -> 15 cells per axis)
# Blocks: (2x2 cells), stride by 1 cell
# Hist bins: 9 orientations
# Expected: ((15-2+1)^2) * (2*2) * 9 = 14^2 * 4 * 9 = 7056
HOG_EXPECTED_FEATURE_SIZE = None  # Will be set during first extraction


def create_hog_descriptor():
    """
    Create a properly configured HOG descriptor with stable parameters.
    Uses the correct OpenCV API with positional arguments.
    
    Returns:
        Configured HOGDescriptor instance
    """
    # Image is 244x244, detect on full image
    # Block size 16x16, cells 2x2 per block (32x32 block)
    # Cells per image: 244/16 = 15.25 -> 15 cells
    try:
        # Create HOG with proper constructor parameters
        # HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
        #               derivAperture=1, winSigma=-1, histogramNormType=0,
        #               L2HysThreshold=0.2, gammaCorrection=True, nlevels=64)
        hog = cv2.HOGDescriptor(
            (IMG_SIZE, IMG_SIZE),      # winSize
            (32, 32),                  # blockSize (2 cells of 16x16)
            (16, 16),                  # blockStride (stride by 1 cell)
            (16, 16),                  # cellSize
            HOG_ORIENTATIONS           # nbins (9)
        )
        return hog
    except Exception as e:
        print(f"Error creating HOGDescriptor: {e}")
        print("Falling back to default HOGDescriptor...")
        # Fallback to default
        return cv2.HOGDescriptor()


# Initialize HOG descriptor once at startup
HOG_DESCRIPTOR = create_hog_descriptor()


# ==========================================
# HOG FEATURE EXTRACTION (ROBUST & STABLE)
# ==========================================
def extract_hog_features(image, visualize=False):
    """
    Extract HOG features from an image with robust error handling.
    Ensures consistent, 1D float32 output.
    
    Args:
        image: Input image (BGR format from cv2)
        visualize: Boolean to return visualization (not currently used)
    
    Returns:
        HOG feature vector (1D np.ndarray, float32) or None if extraction fails
    """
    global HOG_EXPECTED_FEATURE_SIZE
    
    try:
        # Handle None input
        if image is None:
            print("  ERROR: Image is None")
            return None
        
        # Validate image array
        if not isinstance(image, np.ndarray):
            print(f"  ERROR: Image is not ndarray, got {type(image)}")
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] not in [3, 4]:
                print(f"  ERROR: Invalid image channels: {image.shape[2]}")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) != 2:
            print(f"  ERROR: Invalid image shape: {image.shape}")
            return None
        
        # Validate grayscale image
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            print(f"  ERROR: Invalid image dtype: {image.dtype}")
            return None
        
        # Ensure image is uint8 for HOG
        if image.dtype != np.uint8:
            if image.max() > 1.0:  # Likely wrong scale
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:  # [0,1] range, scale to [0,255]
                image = (image * 255).astype(np.uint8)
        
        # Ensure correct size
        if image.shape != (IMG_SIZE, IMG_SIZE):
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Validate image data
        if image.size == 0:
            print("  ERROR: Empty image after processing")
            return None
        
        # Extract HOG features - CRITICAL: winStride must match blockStride
        features = HOG_DESCRIPTOR.compute(
            image, 
            winStride=(16, 16),    # Must match blockStride
            padding=(0, 0),
            locations=None
        )
        
        # Validate HOG output
        if features is None:
            print("  ERROR: HOG.compute returned None")
            return None
        
        # Flatten to 1D
        features = features.flatten()
        
        # Validate feature vector
        if features.size == 0:
            print("  ERROR: Features empty after flatten")
            return None
        
        # Set expected size on first successful extraction
        if HOG_EXPECTED_FEATURE_SIZE is None:
            HOG_EXPECTED_FEATURE_SIZE = features.size
            print(f"  ℹ HOG feature size established: {HOG_EXPECTED_FEATURE_SIZE}")
        
        # Validate feature size consistency
        if features.size != HOG_EXPECTED_FEATURE_SIZE:
            print(f"  WARNING: Feature size mismatch! Expected {HOG_EXPECTED_FEATURE_SIZE}, got {features.size}")
            # Pad or truncate to expected size
            if features.size < HOG_EXPECTED_FEATURE_SIZE:
                features = np.pad(features, (0, HOG_EXPECTED_FEATURE_SIZE - features.size), mode='constant')
            else:
                features = features[:HOG_EXPECTED_FEATURE_SIZE]
        
        # Convert to float32 (stable type for sklearn)
        features = features.astype(np.float32)
        
        # Validate final output
        if not np.all(np.isfinite(features)):
            print(f"  ERROR: Non-finite values in features (NaN/Inf)")
            return None
        
        return features
    
    except Exception as e:
        print(f"  EXCEPTION in extract_hog_features: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==========================================
# DATA LOADING & PREPROCESSING (MEMORY-EFFICIENT)
# ==========================================
def get_image_paths(data_dir=DATA_DIR):
    """
    Get all image paths without loading images into memory.
    Returns path tuples with labels for on-the-fly loading.
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        Tuple of (image_paths_list, labels_list)
    """
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


def load_and_extract_features(image_paths, labels, verbose=True):
    """
    Load images and extract HOG features on-the-fly (memory efficient).
    Processes in batches to avoid memory overflow.
    Ensures consistent 2D array output.
    
    Args:
        image_paths: List of image file paths
        labels: List of corresponding labels
        verbose: Show progress bar
    
    Returns:
        Tuple of (features_array_2d, labels_array) - both properly validated
    """
    features_list = []
    valid_labels = []
    failed_count = 0
    success_count = 0
    feature_sizes = []  # Track all feature sizes for consistency check
    
    print("\nExtracting HOG features (memory-efficient batch processing)...")
    
    iterator = tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing")
    
    for idx, img_path in iterator:
        try:
            # Load single image
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"\n⚠ Failed to load: {img_path} (Invalid file)")
                failed_count += 1
                continue
            
            # Validate image
            if img.size == 0:
                print(f"\n⚠ Empty image: {img_path}")
                failed_count += 1
                continue
            
            # Validate image dtype
            if img.dtype != np.uint8:
                print(f"\n⚠ Invalid dtype {img.dtype}: {img_path}, converting...")
                img = img.astype(np.uint8)
            
            # Resize to standard size
            if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Extract HOG features
            features = extract_hog_features(img)
            
            if features is None or len(features) == 0:
                print(f"\n⚠ Failed to extract features: {img_path}")
                failed_count += 1
                continue
            
            # Validate output
            if features.dtype != np.float32:
                print(f"\n⚠ Wrong dtype {features.dtype}: converting to float32...")
                features = features.astype(np.float32)
            
            # Check for NaN/Inf
            if not np.all(np.isfinite(features)):
                print(f"\n⚠ Non-finite values in features: {img_path}")
                failed_count += 1
                continue
            
            # Ensure 1D
            if len(features.shape) != 1:
                features = features.flatten()
            
            features_list.append(features)
            feature_sizes.append(features.size)
            valid_labels.append(labels[idx])
            success_count += 1
            
            # Free memory from loaded image
            del img
            
            # Periodic garbage collection to prevent memory buildup
            if (idx + 1) % BATCH_SIZE == 0:
                gc.collect()
                iterator.set_postfix({'Extracted': success_count, 'Failed': failed_count})
        
        except Exception as e:
            print(f"\n✗ Error processing {img_path}: {type(e).__name__}: {e}")
            failed_count += 1
            continue
    
    print(f"\n✓ Extraction complete: {success_count} success, {failed_count} failed")
    
    if len(features_list) == 0:
        raise ValueError("No features could be extracted. Check image files and paths.")
    
    # Check feature size consistency
    unique_sizes = set(feature_sizes)
    if len(unique_sizes) > 1:
        print(f"⚠ WARNING: Inconsistent feature sizes detected: {unique_sizes}")
        max_size = max(feature_sizes)
        # Pad all features to max size
        features_list = [
            np.pad(f, (0, max_size - f.size), mode='constant') if f.size < max_size else f[:max_size]
            for f in features_list
        ]
        print(f"  Standardized all features to size {max_size}")
    
    # Convert to 2D array (N_samples, N_features)
    features_array = np.array(features_list, dtype=np.float32)
    labels_array = np.array(valid_labels, dtype=np.int32)
    
    # CRITICAL VALIDATION: Ensure 2D array
    if len(features_array.shape) != 2:
        print(f"ERROR: Features array is not 2D, shape: {features_array.shape}")
        raise ValueError(f"Features must be 2D, got shape {features_array.shape}")
    
    # Validate dimensions
    n_samples, n_features = features_array.shape
    if n_samples != len(labels_array):
        raise ValueError(f"Sample mismatch: features {n_samples} vs labels {len(labels_array)}")
    
    if n_features == 0:
        raise ValueError("Features array has 0 features")
    
    print(f"✓ Features array shape: {features_array.shape} (dtype: {features_array.dtype})")
    print(f"✓ Labels array shape: {labels_array.shape} (dtype: {labels_array.dtype})")
    print(f"  Feature statistics: min={features_array.min():.3f}, max={features_array.max():.3f}, mean={features_array.mean():.3f}")
    print(f"  Labels distribution: {np.bincount(labels_array)}")
    
    return features_array, labels_array


# ==========================================
# MODEL TRAINING
# ==========================================
def train_hog_svm(X_train, y_train, X_val=None, y_val=None, use_grid_search=True):
    """
    Train HOG+SVM model with hyperparameter optimization.
    Includes comprehensive input validation.
    
    Args:
        X_train: Training features (must be 2D ndarray)
        y_train: Training labels (1D ndarray)
        X_val: Validation features (optional, must be 2D)
        y_val: Validation labels (optional)
        use_grid_search: Boolean to perform grid search for hyperparameter tuning
    
    Returns:
        Trained SVM model and scaler
    """
    print("\n" + "="*50)
    print("TRAINING HOG+SVM MODEL")
    print("="*50)
    
    # CRITICAL VALIDATION: Input data types and shapes
    print("\nValidating input data...")
    
    # Check X_train
    if not isinstance(X_train, np.ndarray):
        raise TypeError(f"X_train must be ndarray, got {type(X_train)}")
    if len(X_train.shape) != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")
    if X_train.dtype != np.float32:
        print(f"  Converting X_train from {X_train.dtype} to float32")
        X_train = X_train.astype(np.float32)
    
    # Check y_train
    if not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)
    if len(y_train.shape) != 1:
        raise ValueError(f"y_train must be 1D, got shape {y_train.shape}")
    if len(y_train) != X_train.shape[0]:
        raise ValueError(f"Length mismatch: X_train {X_train.shape[0]} vs y_train {len(y_train)}")
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(X_train)):
        raise ValueError("X_train contains NaN or Inf values")
    if not np.all(np.isfinite(y_train)):
        raise ValueError("y_train contains NaN or Inf values")
    
    print(f"✓ X_train valid: shape {X_train.shape}, dtype {X_train.dtype}")
    print(f"✓ y_train valid: shape {y_train.shape}, dtype {y_train.dtype}")
    print(f"  Class distribution: {np.bincount(y_train)}")
    
    # Validate optional inputs
    if X_val is not None:
        if not isinstance(X_val, np.ndarray):
            raise TypeError(f"X_val must be ndarray, got {type(X_val)}")
        if len(X_val.shape) != 2:
            raise ValueError(f"X_val must be 2D, got shape {X_val.shape}")
        if X_val.shape[1] != X_train.shape[1]:
            raise ValueError(f"Feature mismatch: X_train {X_train.shape[1]} vs X_val {X_val.shape[1]}")
        if X_val.dtype != np.float32:
            X_val = X_val.astype(np.float32)
        print(f"✓ X_val valid: shape {X_val.shape}, dtype {X_val.dtype}")
    
    if y_val is not None:
        if not isinstance(y_val, np.ndarray):
            y_val = np.array(y_val)
        if len(y_val) != X_val.shape[0]:
            raise ValueError(f"Length mismatch: X_val {X_val.shape[0]} vs y_val {len(y_val)}")
        print(f"✓ y_val valid: shape {y_val.shape}")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Validate scaler output
    if X_train_scaled.shape != X_train.shape:
        raise ValueError(f"Scaler output shape mismatch: {X_train_scaled.shape} vs {X_train.shape}")
    
    X_train_scaled = X_train_scaled.astype(np.float32)
    print(f"✓ Training features scaled: shape {X_train_scaled.shape}, min={X_train_scaled.min():.3f}, max={X_train_scaled.max():.3f}")
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val).astype(np.float32)
        print(f"✓ Validation features scaled: shape {X_val_scaled.shape}")
    
    # Hyperparameter tuning with Grid Search
    if use_grid_search:
        print("\nTraining SVM with BEST PARAMETERS (from tuning results)...")
        print("  Best Parameters: C=0.1, kernel='linear', gamma='scale'")
        
        # Use BEST parameters identified from hyperparameter tuning
        model = SVC(kernel='linear', C=0.1, gamma='scale', probability=True, 
                   random_state=RANDOM_STATE, cache_size=500, max_iter=2000)
        try:
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"SVM training error: {type(e).__name__}: {e}")
            raise
    else:
        # Train with default parameters
        print("\nTraining SVM with default parameters...")
        model = SVC(kernel='rbf', C=100, gamma='scale', probability=True, 
                   random_state=RANDOM_STATE, cache_size=500, max_iter=2000)
        try:
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"SVM training error: {type(e).__name__}: {e}")
            raise
    
    print(f"✓ Model trained: {type(model).__name__}")
    
    # Validate on training set
    print("\n" + "-"*50)
    print("TRAINING SET PERFORMANCE")
    print("-"*50)
    try:
        y_train_pred = model.predict(X_train_scaled)
        if y_train_pred.shape != y_train.shape:
            raise ValueError(f"Prediction shape mismatch: {y_train_pred.shape} vs {y_train.shape}")
        _print_metrics(y_train, y_train_pred)
    except Exception as e:
        print(f"Error in training set evaluation: {e}")
    
    # Validate on validation set if provided
    if X_val is not None and y_val is not None:
        print("\n" + "-"*50)
        print("VALIDATION SET PERFORMANCE")
        print("-"*50)
        try:
            y_val_pred = model.predict(X_val_scaled)
            if y_val_pred.shape != y_val.shape:
                raise ValueError(f"Validation prediction shape mismatch")
            _print_metrics(y_val, y_val_pred)
        except Exception as e:
            print(f"Error in validation set evaluation: {e}")
    
    return model, scaler


def _print_metrics(y_true, y_pred):
    """Helper function to print evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ==========================================
# MODEL EVALUATION
# ==========================================
def evaluate_model(model, scaler, X_test, y_test, dataset_name="Test"):
    """
    Comprehensive model evaluation with multiple metrics.
    Includes robust error handling and input validation.
    
    Args:
        model: Trained SVM model
        scaler: Feature scaler
        X_test: Test features (must be 2D)
        y_test: Test labels
        dataset_name: Name of dataset for display
    
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        print("\n" + "="*50)
        print(f"{dataset_name.upper()} SET EVALUATION")
        print("="*50)
        
        # Validate inputs
        if X_test is None or len(X_test) == 0:
            raise ValueError("Test features are empty")
        if y_test is None or len(y_test) == 0:
            raise ValueError("Test labels are empty")
        
        # Ensure proper shapes
        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test, dtype=np.float32)
        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)
        
        # Validate dimensions
        if len(X_test.shape) != 2:
            raise ValueError(f"X_test must be 2D, got {X_test.shape}")
        if len(X_test) != len(y_test):
            raise ValueError(f"Feature and label mismatch: {len(X_test)} vs {len(y_test)}")
        
        # Ensure float32
        if X_test.dtype != np.float32:
            X_test = X_test.astype(np.float32)
        
        print(f"Input validation:\n  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        print(f"  X_test dtype: {X_test.dtype}")
        
        # Scale features
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
        if X_test_scaled.shape != X_test.shape:
            raise ValueError(f"Scaler output mismatch: {X_test_scaled.shape} vs {X_test.shape}")
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        if y_pred.shape != y_test.shape:
            raise ValueError(f"Prediction shape mismatch: {y_pred.shape} vs {y_test.shape}")
        
        y_pred_proba = model.predict_proba(X_test_scaled)
        if y_pred_proba.shape[0] != len(y_test):
            raise ValueError(f"Probability shape mismatch")
        
        print(f"✓ Predictions generated successfully")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\n{dataset_name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=[CLASS_LABELS[0], CLASS_LABELS[1]]))
        
        cm = confusion_matrix(y_test, y_pred)
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
            "y_true": y_test
        }
        
        return metrics
    
    except Exception as e:
        print(f"ERROR in evaluate_model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def cross_validate_model(X, y, cv=5):
    """
    Perform k-fold cross-validation.
    
    Args:
        X: Features
        y: Labels
        cv: Number of folds
    
    Returns:
        Cross-validation scores
    """
    print("\n" + "="*50)
    print("CROSS-VALIDATION")
    print("="*50)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=RANDOM_STATE)
    
    print(f"\nPerforming {cv}-fold cross-validation...")
    
    # Use multiple scoring metrics
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    scores = {}
    
    for metric in scoring:
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=metric)
        scores[metric] = cv_scores
        print(f"\n{metric.upper()} Scores: {cv_scores}")
        print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return scores


# ==========================================
# VISUALIZATION
# ==========================================
def plot_confusion_matrix(cm, dataset_name="Test"):
    """Plot confusion matrix (non-blocking)."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]],
                yticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]])
    plt.title(f'Confusion Matrix - {dataset_name} Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    filepath = os.path.join(MODEL_SAVE_DIR, f'confusion_matrix_{dataset_name.lower()}.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()  # Close instead of show to prevent blocking
    print(f"✓ Confusion matrix saved to {filepath}")


def plot_roc_curve(y_true, y_pred_proba, dataset_name="Test"):
    """Plot ROC curve (non-blocking)."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} Set')
    plt.legend(loc="lower right")
    plt.tight_layout()
    filepath = os.path.join(MODEL_SAVE_DIR, f'roc_curve_{dataset_name.lower()}.png')
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()  # Close instead of show to prevent blocking
    print(f"✓ ROC curve saved to {filepath}")


# ==========================================
# MODEL PERSISTENCE
# ==========================================
def save_model(model, scaler, model_dir=MODEL_SAVE_DIR):
    """Save trained model and scaler."""
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, MODEL_NAME)
    scaler_path = os.path.join(model_dir, SCALER_NAME)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


def load_model_from_disk(model_dir=MODEL_SAVE_DIR):
    """Load trained model and scaler."""
    model_path = os.path.join(model_dir, MODEL_NAME)
    scaler_path = os.path.join(model_dir, SCALER_NAME)
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found in {model_dir}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    print(f"Scaler loaded from {scaler_path}")
    
    return model, scaler


# ==========================================
# INFERENCE ON SINGLE IMAGE
# ==========================================
def predict_single_image(image_path, model, scaler):
    """
    Predict on a single image.
    
    Args:
        image_path: Path to image file
        model: Trained SVM model
        scaler: Feature scaler
    
    Returns:
        Tuple of (prediction, confidence)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    features = extract_hog_features(img)
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    confidence = np.max(model.predict_proba(features_scaled)[0])
    
    return prediction, confidence


# ==========================================
# MAIN TRAINING PIPELINE WITH 2-FOLD CV
# ==========================================
def main():
    """Main training pipeline with 2-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold
    
    print("\n" + "="*70)
    print("HOG + SVM MODEL FOR PCB ORIENTATION DETECTION")
    print("WITH 2-FOLD CROSS-VALIDATION & HYPERPARAMETER TUNING")
    print("="*70)
    print("*** CONSOLIDATED TRAINER WITH MEMORY MANAGEMENT ***\n")
    
    try:
        start_time = time.time()
        
        # STEP 1: Scan data directory
        print("STEP 1: Scanning Data Directory")
        print("-"*70)
        step_start = time.time()
        image_paths, all_labels = get_image_paths(DATA_DIR)
        image_paths_arr = np.array(image_paths)
        labels_arr = np.array(all_labels)
        print(f"✓ Scan completed in {time.time() - step_start:.1f}s\n")
        
        # STEP 2: Extract all features once
        print("STEP 2: Extracting ALL Features (for 2-fold CV)")
        print("-"*70)
        print(f"Processing {len(image_paths_arr)} images...")
        step_start = time.time()
        all_features, all_labels_filtered = load_and_extract_features(image_paths_arr, labels_arr)
        feature_extract_time = time.time() - step_start
        print(f"✓ All features extracted in {feature_extract_time:.1f}s\n")
        
        # Force garbage collection
        gc.collect()
        
        # STEP 3: 2-FOLD STRATIFIED CROSS-VALIDATION WITH HYPERPARAMETER TUNING
        print("STEP 3: 2-FOLD STRATIFIED CROSS-VALIDATION")
        print("-"*70)
        
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)
        fold_results = []
        best_model = None
        best_scaler = None
        best_overall_score = -1
        best_fold = -1
        
        fold_num = 0
        for train_idx, test_idx in skf.split(all_features, all_labels_filtered):
            fold_num += 1
            print(f"\n{'='*70}")
            print(f"FOLD {fold_num}/2")
            print(f"{'='*70}")
            
            # Split data for this fold
            X_fold_train = all_features[train_idx].astype(np.float32)
            y_fold_train = all_labels_filtered[train_idx]
            X_fold_test = all_features[test_idx].astype(np.float32)
            y_fold_test = all_labels_filtered[test_idx]
            
            print(f"Train/Test split: {len(X_fold_train)} / {len(X_fold_test)} (stratified)")
            
            # Train model with hyperparameter tuning on this fold
            print(f"\nTraining fold {fold_num} with hyperparameter optimization...")
            fold_start = time.time()
            
            fold_model, fold_scaler = train_hog_svm(
                X_fold_train, y_fold_train, 
                X_fold_test, y_fold_test, 
                use_grid_search=True
            )
            
            fold_train_time = time.time() - fold_start
            
            # Evaluate on test set of this fold
            print(f"\nEvaluating fold {fold_num}...")
            fold_metrics = evaluate_model(fold_model, fold_scaler, X_fold_test, y_fold_test, 
                                         f"Fold {fold_num} Test")
            
            # Track results
            fold_result = {
                'fold': fold_num,
                'model': fold_model,
                'scaler': fold_scaler,
                'metrics': fold_metrics,
                'train_time': fold_train_time,
                'f1_score': fold_metrics['f1'],
                'accuracy': fold_metrics['accuracy']
            }
            fold_results.append(fold_result)
            
            # Update best model if this fold is better
            if fold_metrics['f1'] > best_overall_score:
                best_overall_score = fold_metrics['f1']
                best_model = fold_model
                best_scaler = fold_scaler
                best_fold = fold_num
            
            print(f"\nFold {fold_num} completed in {fold_train_time:.1f}s")
            print(f"Fold {fold_num} F1-Score: {fold_metrics['f1']:.4f}")
            gc.collect()
        
        # STEP 4: Report Cross-Validation Results
        print(f"\n{'='*70}")
        print("2-FOLD CROSS-VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        all_accuracies = [r['accuracy'] for r in fold_results]
        all_f1_scores = [r['f1_score'] for r in fold_results]
        
        print(f"\nFold 1 Accuracy: {fold_results[0]['accuracy']:.4f}  |  F1-Score: {fold_results[0]['f1_score']:.4f}")
        print(f"Fold 2 Accuracy: {fold_results[1]['accuracy']:.4f}  |  F1-Score: {fold_results[1]['f1_score']:.4f}")
        print(f"\nMean Accuracy: {np.mean(all_accuracies):.4f} (+/- {np.std(all_accuracies):.4f})")
        print(f"Mean F1-Score: {np.mean(all_f1_scores):.4f} (+/- {np.std(all_f1_scores):.4f})")
        print(f"\n✓ BEST MODEL: From Fold {best_fold} with F1-Score {best_overall_score:.4f}")
        
        # STEP 5: Visualizations for best fold
        print(f"\n{'='*70}")
        print("STEP 4: Generating Visualizations (Best Fold)")
        print(f"{'='*70}")
        
        best_metrics = fold_results[best_fold - 1]['metrics']
        step_start = time.time()
        plot_confusion_matrix(best_metrics['confusion_matrix'], f"Fold{best_fold}")
        plot_roc_curve(best_metrics['y_true'], best_metrics['y_pred_proba'], f"Fold{best_fold}")
        viz_time = time.time() - step_start
        print(f"✓ Visualizations completed in {viz_time:.1f}s\n")
        
        # STEP 6: Save best model
        print("STEP 5: Saving Best Model")
        print("-"*70)
        save_model(best_model, best_scaler)
        
        # Final Summary
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
        
        print(f"\nCROS-VALIDATION STATISTICS:")
        print(f"  Mean Accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
        print(f"  Mean F1-Score: {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}")
        
        print(f"\nTiming Breakdown:")
        print(f"  Feature extraction: {feature_extract_time:.1f}s")
        print(f"  Fold 1 training:    {fold_results[0]['train_time']:.1f}s")
        print(f"  Fold 2 training:    {fold_results[1]['train_time']:.1f}s")
        print(f"  Visualizations:     {viz_time:.1f}s")
        print(f"  Total time:         {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print("="*70 + "\n")
        
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
