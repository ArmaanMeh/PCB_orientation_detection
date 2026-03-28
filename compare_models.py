"""
Model Comparison and Evaluation Script
Compare HOG+SVM with CNN model on test data
"""

import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ==========================================
# CONFIGURATION
# ==========================================
IMG_SIZE = 244
CLASS_LABELS = ["Fail", "Pass"]

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

CNN_MODEL_PATH = "Export/ot_model.keras"
HOG_SVM_MODEL_PATH = "Export/hog_svm_model.pkl"
HOG_SCALER_PATH = "Export/hog_svm_scaler.pkl"

TEST_DATA_DIR = "Data/Processed_data"


# ==========================================
# HOG FEATURE EXTRACTION
# ==========================================
def extract_hog_features(image):
    """Extract HOG features from image."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hog = cv2.HOGDescriptor(
        (IMG_SIZE, IMG_SIZE),
        HOG_PIXELS_PER_CELL,
        HOG_CELLS_PER_BLOCK,
        HOG_ORIENTATIONS,
        9, 1, -1, 0.2, 0.1, 1.0, 64, 0
    )
    
    features = hog.compute(image)
    return features.flatten()


# ==========================================
# DATA LOADING
# ==========================================
def load_test_data(data_dir=TEST_DATA_DIR):
    """Load test data from directory."""
    images = []
    labels = []
    filenames = []
    
    print(f"Loading test data from {data_dir}...")
    
    # Load Pass data
    pass_dir = os.path.join(data_dir, "Pass_data")
    if os.path.exists(pass_dir):
        for img_file in os.listdir(pass_dir)[:10]:  # Use first 10 for quick testing
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(pass_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        images.append(img)
                        labels.append(1)
                        filenames.append(img_file)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    # Load Fail data
    fail_dir = os.path.join(data_dir, "Fail_data")
    if os.path.exists(fail_dir):
        for img_file in os.listdir(fail_dir)[:10]:  # Use first 10 for quick testing
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(fail_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        images.append(img)
                        labels.append(0)
                        filenames.append(img_file)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    print(f"Loaded {len(images)} test images")
    
    return np.array(images), np.array(labels), filenames


# ==========================================
# CNN PREDICTIONS
# ==========================================
def get_cnn_predictions(images, model_path=CNN_MODEL_PATH):
    """Get predictions from CNN model."""
    print("\nLoading CNN model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✓ CNN model loaded")
    except Exception as e:
        print(f"ERROR loading CNN model: {e}")
        return None, None
    
    print("Running CNN inference...")
    predictions_list = []
    probabilities_list = []
    
    for i, img in enumerate(images):
        img_array = np.expand_dims(img, axis=0)
        logits = model.predict(img_array, verbose=0)
        probs = tf.nn.softmax(logits[0]).numpy()
        
        pred = np.argmax(probs)
        predictions_list.append(pred)
        probabilities_list.append(probs)
    
    return np.array(predictions_list), np.array(probabilities_list)


# ==========================================
# HOG+SVM PREDICTIONS
# ==========================================
def get_hog_svm_predictions(images, model_path=HOG_SVM_MODEL_PATH, scaler_path=HOG_SCALER_PATH):
    """Get predictions from HOG+SVM model."""
    print("\nLoading HOG+SVM model...")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✓ HOG+SVM model loaded")
    except FileNotFoundError as e:
        print(f"ERROR loading HOG+SVM model: {e}")
        return None, None
    
    print("Extracting HOG features...")
    features = []
    for img in images:
        hog_feat = extract_hog_features(img)
        features.append(hog_feat)
    
    features = np.array(features)
    features_scaled = scaler.transform(features)
    
    print("Running HOG+SVM inference...")
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    return predictions, probabilities


# ==========================================
# EVALUATION
# ==========================================
def evaluate_predictions(y_true, y_pred, model_name):
    """Evaluate predictions."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n{model_name} Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:\n{cm}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def compare_models(y_true, y_pred_cnn, y_pred_hog_svm):
    """Compare two models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Evaluate CNN
    cnn_metrics = evaluate_predictions(y_true, y_pred_cnn, "CNN")
    
    # Evaluate HOG+SVM
    hog_svm_metrics = evaluate_predictions(y_true, y_pred_hog_svm, "HOG+SVM")
    
    # Comparison summary
    print("\n" + "-"*60)
    print("COMPARISON SUMMARY")
    print("-"*60)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    comparison_data = {
        'CNN': [cnn_metrics[m] for m in metrics_to_compare],
        'HOG+SVM': [hog_svm_metrics[m] for m in metrics_to_compare]
    }
    
    print(f"\n{'Metric':<15} {'CNN':<15} {'HOG+SVM':<15} {'Difference':<15}")
    print("-"*60)
    
    for metric in metrics_to_compare:
        cnn_val = cnn_metrics[metric]
        hog_val = hog_svm_metrics[metric]
        diff = cnn_val - hog_val
        
        winner = "🔹 CNN" if diff > 0 else "🔹 HOG+SVM" if diff < 0 else "🔹 TIE"
        
        print(f"{metric.capitalize():<15} {cnn_val:<15.4f} {hog_val:<15.4f} {diff:+.4f} {winner}")
    
    return cnn_metrics, hog_svm_metrics


# ==========================================
# VISUALIZATION
# ==========================================
def plot_model_comparison(y_true, y_pred_cnn, y_pred_hog_svm, cnn_metrics, hog_svm_metrics):
    """Plot comparison visualizations."""
    print("\nGenerating comparison visualizations...")
    
    # 1. Confusion Matrices Side by Side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm_cnn = cnn_metrics['confusion_matrix']
    cm_hog = hog_svm_metrics['confusion_matrix']
    
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    axes[0].set_title('CNN Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    sns.heatmap(cm_hog, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    axes[1].set_title('HOG+SVM Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('Export/confusion_matrices_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. Metrics Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    cnn_values = [cnn_metrics['accuracy'], cnn_metrics['precision'], 
                  cnn_metrics['recall'], cnn_metrics['f1']]
    hog_values = [hog_svm_metrics['accuracy'], hog_svm_metrics['precision'],
                  hog_svm_metrics['recall'], hog_svm_metrics['f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, cnn_values, width, label='CNN', color='skyblue')
    ax.bar(x + width/2, hog_values, width, label='HOG+SVM', color='lightgreen')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [ax.patches[:len(metrics)], ax.patches[len(metrics):]]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('Export/metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Visualizations saved to Export/")


def predict_on_single_image(image_path, cnn_model, hog_svm_model, hog_scaler):
    """Make predictions on a single image with both models."""
    print(f"\n{'='*60}")
    print(f"Predicting on: {image_path}")
    print(f"{'='*60}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("ERROR: Could not load image")
        return
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # CNN prediction
    img_array = np.expand_dims(img, axis=0)
    logits = cnn_model.predict(img_array, verbose=0)
    cnn_probs = tf.nn.softmax(logits[0]).numpy()
    cnn_pred = np.argmax(cnn_probs)
    
    # HOG+SVM prediction
    hog_features = extract_hog_features(img)
    hog_features = hog_features.reshape(1, -1)
    hog_features_scaled = hog_scaler.transform(hog_features)
    hog_pred = hog_svm_model.predict(hog_features_scaled)[0]
    hog_probs = hog_svm_model.predict_proba(hog_features_scaled)[0]
    
    print("\nCNN Results:")
    print(f"  Prediction: {CLASS_LABELS[int(cnn_pred)]}")
    print(f"  Confidence: {cnn_probs[int(cnn_pred)]:.2%}")
    print(f"  [Fail: {cnn_probs[0]:.2%}, Pass: {cnn_probs[1]:.2%}]")
    
    print("\nHOG+SVM Results:")
    print(f"  Prediction: {CLASS_LABELS[int(hog_pred)]}")
    print(f"  Confidence: {hog_probs[int(hog_pred)]:.2%}")
    print(f"  [Fail: {hog_probs[0]:.2%}, Pass: {hog_probs[1]:.2%}]")
    
    agreement = "✓ AGREE" if cnn_pred == hog_pred else "✗ DISAGREE"
    print(f"\nModels: {agreement}")


# ==========================================
# MAIN
# ==========================================
def main():
    """Main evaluation pipeline."""
    print("\n" + "="*60)
    print("CNN vs HOG+SVM MODEL COMPARISON")
    print("="*60)
    
    # Load test data
    print("\nSTEP 1: Loading Test Data")
    print("-"*60)
    X_test, y_test, filenames = load_test_data()
    
    if len(X_test) == 0:
        print("ERROR: No test data loaded")
        return
    
    # Get CNN predictions
    print("\nSTEP 2: CNN Predictions")
    print("-"*60)
    y_pred_cnn, y_proba_cnn = get_cnn_predictions(X_test)
    
    if y_pred_cnn is None:
        print("ERROR: Could not get CNN predictions")
        return
    
    # Get HOG+SVM predictions
    print("\nSTEP 3: HOG+SVM Predictions")
    print("-"*60)
    y_pred_hog, y_proba_hog = get_hog_svm_predictions(X_test)
    
    if y_pred_hog is None:
        print("ERROR: Could not get HOG+SVM predictions")
        return
    
    # Compare models
    print("\nSTEP 4: Model Comparison")
    print("-"*60)
    cnn_metrics, hog_svm_metrics = compare_models(y_test, y_pred_cnn, y_pred_hog)
    
    # Visualizations
    print("\nSTEP 5: Generating Visualizations")
    print("-"*60)
    plot_model_comparison(y_test, y_pred_cnn, y_pred_hog, cnn_metrics, hog_svm_metrics)
    
    # Individual predictions
    print("\nSTEP 6: Sample Individual Predictions")
    print("-"*60)
    
    try:
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
        with open(HOG_SVM_MODEL_PATH, 'rb') as f:
            hog_model = pickle.load(f)
        with open(HOG_SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        # Show predictions for first 3 images
        for i in range(min(3, len(X_test))):
            sample_img_path = f"sample_{i}.jpg"
            cv2.imwrite(sample_img_path, X_test[i])
            predict_on_single_image(sample_img_path, cnn_model, hog_model, scaler)
    except Exception as e:
        print(f"Could not perform individual predictions: {e}")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
