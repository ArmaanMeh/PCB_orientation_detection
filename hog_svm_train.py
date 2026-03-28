"""
HOG + SVM Model for PCB Orientation Detection
Includes: feature extraction, hyperparameter optimization, validation, 
evaluation metrics, and robust training pipeline
"""

import cv2
import numpy as np
import os
import pickle
import time
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
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


# ==========================================
# HOG FEATURE EXTRACTION
# ==========================================
def extract_hog_features(image, visualize=False):
    """
    Extract HOG features from an image.
    
    Args:
        image: Input image (BGR format from cv2)
        visualize: Boolean to return visualization (not currently used)
    
    Returns:
        HOG feature vector
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Ensure image is the correct size
    if image.shape != (IMG_SIZE, IMG_SIZE):
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Use default HOGDescriptor - works with standard parameters with any image size
    hog = cv2.HOGDescriptor()
    
    features = hog.compute(image, winStride=(8, 8), padding=(0, 0))
    features = features.flatten()
    
    return features


# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
def load_data(data_dir=DATA_DIR):
    """
    Load images and labels from directory structure.
    Expected structure: data_dir/Pass_data/*.jpg and data_dir/Fail_data/*.jpg
    
    Returns:
        Tuple of (images, labels, image_paths)
    """
    images = []
    labels = []
    image_paths = []
    
    print("Loading data from directory structure...")
    
    # Load Pass data (label 1)
    pass_dir = os.path.join(data_dir, "Pass_data")
    if os.path.exists(pass_dir):
        print(f"Loading Pass images from {pass_dir}...")
        for img_file in tqdm(os.listdir(pass_dir)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(pass_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        images.append(img)
                        labels.append(1)
                        image_paths.append(img_path)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    # Load Fail data (label 0)
    fail_dir = os.path.join(data_dir, "Fail_data")
    if os.path.exists(fail_dir):
        print(f"Loading Fail images from {fail_dir}...")
        for img_file in tqdm(os.listdir(fail_dir)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(fail_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        images.append(img)
                        labels.append(0)
                        image_paths.append(img_path)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    if len(images) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Loaded {len(images)} images")
    print(f"Pass images: {sum(1 for l in labels if l == 1)}")
    print(f"Fail images: {sum(1 for l in labels if l == 0)}")
    
    return np.array(images), np.array(labels), image_paths


def extract_features_batch(images, verbose=True):
    """
    Extract HOG features from a batch of images.
    
    Args:
        images: Array of images
        verbose: Show progress bar
    
    Returns:
        Array of HOG features
    """
    features = []
    iterator = tqdm(images, desc="Extracting HOG features") if verbose else images
    
    for img in iterator:
        try:
            hog_features = extract_hog_features(img, visualize=False)
            features.append(hog_features)
        except Exception as e:
            print(f"Error extracting features: {e}")
            continue
    
    return np.array(features)


# ==========================================
# MODEL TRAINING
# ==========================================
def train_hog_svm(X_train, y_train, X_val=None, y_val=None, use_grid_search=True):
    """
    Train HOG+SVM model with hyperparameter optimization.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        use_grid_search: Boolean to perform grid search for hyperparameter tuning
    
    Returns:
        Trained SVM model and scaler
    """
    print("\n" + "="*50)
    print("TRAINING HOG+SVM MODEL")
    print("="*50)
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
    
    # Hyperparameter tuning with Grid Search
    if use_grid_search:
        print("\nPerforming GridSearchCV for hyperparameter optimization...")
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        }
        
        svm = SVC(probability=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        # Train with default parameters
        print("\nTraining SVM with default parameters...")
        model = SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=RANDOM_STATE)
        model.fit(X_train_scaled, y_train)
    
    # Validate on training set
    print("\n" + "-"*50)
    print("TRAINING SET PERFORMANCE")
    print("-"*50)
    y_train_pred = model.predict(X_train_scaled)
    _print_metrics(y_train, y_train_pred)
    
    # Validate on validation set if provided
    if X_val is not None and y_val is not None:
        print("\n" + "-"*50)
        print("VALIDATION SET PERFORMANCE")
        print("-"*50)
        y_val_pred = model.predict(X_val_scaled)
        _print_metrics(y_val, y_val_pred)
    
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
    
    Args:
        model: Trained SVM model
        scaler: Feature scaler
        X_test: Test features
        y_test: Test labels
        dataset_name: Name of dataset for display
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "="*50)
    print(f"{dataset_name.upper()} SET EVALUATION")
    print("="*50)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
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
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]],
                yticklabels=[CLASS_LABELS[0], CLASS_LABELS[1]])
    plt.title(f'Confusion Matrix - {dataset_name} Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, f'confusion_matrix_{dataset_name.lower()}.png'))
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, dataset_name="Test"):
    """Plot ROC curve."""
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
    plt.savefig(os.path.join(MODEL_SAVE_DIR, f'roc_curve_{dataset_name.lower()}.png'))
    plt.show()


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
# MAIN TRAINING PIPELINE
# ==========================================
def main():
    """Main training pipeline."""
    print("\n" + "="*60)
    print("HOG + SVM MODEL FOR PCB ORIENTATION DETECTION")
    print("="*60)
    
    # Step 1: Load data
    print("\nSTEP 1: Loading Data")
    print("-"*60)
    images, labels, _ = load_data(DATA_DIR)
    
    # Step 2: Split data
    print("\nSTEP 2: Splitting Data")
    print("-"*60)
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=TEST_SPLIT, random_state=RANDOM_STATE, stratify=labels
    )
    
    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SPLIT/(1-TEST_SPLIT), 
        random_state=RANDOM_STATE, stratify=y_temp
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Step 3: Extract HOG features
    print("\nSTEP 3: Extracting HOG Features")
    print("-"*60)
    
    X_train_features = extract_features_batch(X_train)
    X_val_features = extract_features_batch(X_val)
    X_test_features = extract_features_batch(X_test)
    
    print(f"Feature vector size: {X_train_features.shape[1]}")
    
    # Step 4: Train model with hyperparameter tuning
    print("\nSTEP 4: Training Model (with Hyperparameter Optimization)")
    print("-"*60)
    
    model, scaler = train_hog_svm(X_train_features, y_train, X_val_features, y_val, use_grid_search=True)
    
    # Step 5: Evaluation on test set
    print("\nSTEP 5: Test Set Evaluation")
    print("-"*60)
    
    test_metrics = evaluate_model(model, scaler, X_test_features, y_test, "Test")
    
    # Step 6: Cross-validation
    print("\nSTEP 6: Cross-Validation Analysis")
    print("-"*60)
    
    cv_scores = cross_validate_model(X_train_features, y_train, cv=5)
    
    # Step 7: Visualizations
    print("\nSTEP 7: Generating Visualizations")
    print("-"*60)
    
    plot_confusion_matrix(test_metrics['confusion_matrix'], "Test")
    plot_roc_curve(test_metrics['y_true'], test_metrics['y_pred_proba'], "Test")
    
    # Step 8: Save model
    print("\nSTEP 8: Saving Model")
    print("-"*60)
    
    save_model(model, scaler)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
