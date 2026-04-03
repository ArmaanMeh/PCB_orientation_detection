#!/usr/bin/env python3
"""
Unified Model Management & Comparison Suite
Combines: model_utils, compare_models, and utilities
Minimal, fast, robust, non-crashing
"""

import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import time
from contextlib import contextmanager

# ==========================================
# CONFIGURATION
# ==========================================
IMG_SIZE = 244
CNN_MODEL_PATH = "Export/ot_model.keras"
HOG_SVM_MODEL_PATH = "Export/hog_svm_model.pkl"
HOG_SCALER_PATH = "Export/hog_svm_scaler.pkl"
DATA_DIR = "Data/Processed_data"
CLASS_LABELS = ["Fail", "Pass"]

HOG_CONFIG = {
    'orientations': 9,
    'pixels_per_cell': (16, 16),
    'cells_per_block': (2, 2),
    'img_size': 244
}

@contextmanager
def timer(name):
    """Context manager for timing operations."""
    start = time.time()
    try:
        yield
    finally:
        print(f"  ⏱ {name}: {time.time() - start:.2f}s")

# ==========================================
# HOG FEATURE EXTRACTION
# ==========================================
def extract_hog_features(image):
    """Extract HOG features - robust version."""
    try:
        if image is None:
            return None
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        hog = cv2.HOGDescriptor(
            (HOG_CONFIG['img_size'], HOG_CONFIG['img_size']),
            (32, 32),
            (4, 4),  # Updated stride for 244x244 compatibility
            HOG_CONFIG['pixels_per_cell'],
            HOG_CONFIG['orientations']
        )
        features = hog.compute(image)
        return features.flatten().astype(np.float32) if features is not None else None
    except Exception as e:
        print(f"  ⚠ HOG extraction error: {e}")
        return None

# ==========================================
# MODEL LOADING
# ==========================================
def load_cnn_model():
    """Load CNN model with error handling."""
    try:
        if os.path.exists(CNN_MODEL_PATH):
            model = tf.keras.models.load_model(CNN_MODEL_PATH)
            print(f"✓ CNN model loaded: {CNN_MODEL_PATH}")
            return model
        else:
            print(f"✗ CNN model not found: {CNN_MODEL_PATH}")
            return None
    except Exception as e:
        print(f"✗ Error loading CNN: {e}")
        return None

def load_hog_svm_model():
    """Load HOG+SVM model with error handling."""
    try:
        if os.path.exists(HOG_SVM_MODEL_PATH) and os.path.exists(HOG_SCALER_PATH):
            with open(HOG_SVM_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(HOG_SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✓ HOG+SVM model loaded")
            return model, scaler
        else:
            print(f"✗ HOG+SVM model files not found")
            return None, None
    except Exception as e:
        print(f"✗ Error loading HOG+SVM: {e}")
        return None, None

# ==========================================
# PREDICTION FUNCTIONS
# ==========================================
def predict_cnn(model, image):
    """CNN prediction with error handling."""
    try:
        img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img_array = np.expand_dims(img_resized, axis=0) / 255.0
        logits = model.predict(img_array, verbose=0)
        probs = tf.nn.softmax(logits[0]).numpy()
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        return CLASS_LABELS[pred_class], float(confidence)
    except Exception as e:
        print(f"  ⚠ CNN prediction error: {e}")
        return "Error", 0.0

def predict_hog_svm(model, scaler, image):
    """HOG+SVM prediction with error handling."""
    try:
        img_resized = cv2.resize(image, (HOG_CONFIG['img_size'], HOG_CONFIG['img_size']))
        features = extract_hog_features(img_resized)
        if features is None or len(features) == 0:
            return "Error", 0.0
        features_scaled = scaler.transform([features])
        pred_class = model.predict(features_scaled)[0]
        confidence = abs(model.decision_function(features_scaled)[0])
        return CLASS_LABELS[int(pred_class)], min(float(confidence), 1.0)
    except Exception as e:
        print(f"  ⚠ HOG+SVM prediction error: {e}")
        return "Error", 0.0

# ==========================================
# MODEL COMPARISON
# ==========================================
def compare_models(test_image_count=100):
    """Compare CNN vs HOG+SVM performance."""
    print("\n" + "="*60)
    print("MODEL COMPARISON - CNN vs HOG+SVM")
    print("="*60)
    
    # Load models
    cnn_model = load_cnn_model()
    hog_svm_model, hog_scaler = load_hog_svm_model()
    
    if cnn_model is None or hog_svm_model is None:
        print("✗ Cannot compare: missing models")
        return
    
    # Load test images
    print("\nLoading test images...")
    test_images = []
    test_labels = []
    
    for class_idx, class_name in enumerate(['Fail_data', 'Pass_data']):
        class_dir = Path(DATA_DIR) / class_name
        images = sorted(list(class_dir.glob('*.jpg')))[:test_image_count // 2]
        
        for img_path in images:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    test_images.append(img)
                    test_labels.append(class_idx)
            except:
                pass
    
    print(f"✓ Loaded {len(test_images)} test images")
    
    # Test both models
    cnn_preds = []
    hog_preds = []
    
    print("\nTesting models...")
    with timer("CNN inference"):
        for img in test_images:
            pred, _ = predict_cnn(cnn_model, img)
            cnn_preds.append(0 if pred == "Fail" else 1)
    
    with timer("HOG+SVM inference"):
        for img in test_images:
            pred, _ = predict_hog_svm(hog_svm_model, hog_scaler, img)
            hog_preds.append(0 if pred == "Fail" else 1)
    
    # Calculate metrics
    cnn_acc = accuracy_score(test_labels, cnn_preds)
    hog_acc = accuracy_score(test_labels, hog_preds)
    cnn_f1 = f1_score(test_labels, cnn_preds, average='binary')
    hog_f1 = f1_score(test_labels, hog_preds, average='binary')
    
    # Print results
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    print(f"\n{'Model':<15} {'Accuracy':<12} {'F1 Score':<12}")
    print("-"*60)
    print(f"{'CNN':<15} {cnn_acc:.4f}         {cnn_f1:.4f}")
    print(f"{'HOG+SVM':<15} {hog_acc:.4f}         {hog_f1:.4f}")
    print("-"*60)
    
    winner = "CNN" if cnn_acc > hog_acc else "HOG+SVM"
    diff = abs(cnn_acc - hog_acc)
    print(f"\n✓ Winner: {winner} (+{diff:.4f})")

# ==========================================
# UTILITIES
# ==========================================
def test_single_image(image_path, model_type='both'):
    """Test a single image with specified model."""
    print(f"\n📷 Testing: {image_path}")
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("✗ Could not load image")
            return
        
        if model_type in ['cnn', 'both']:
            model = load_cnn_model()
            if model:
                pred, conf = predict_cnn(model, image)
                print(f"  CNN: {pred} ({conf:.2%})")
        
        if model_type in ['hog', 'both']:
            model, scaler = load_hog_svm_model()
            if model and scaler:
                pred, conf = predict_hog_svm(model, scaler, image)
                print(f"  HOG+SVM: {pred} ({conf:.2%})")
    except Exception as e:
        print(f"✗ Error: {e}")

def print_model_info():
    """Print model information."""
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"\nCNN Model:")
    print(f"  Path: {CNN_MODEL_PATH}")
    print(f"  Input Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Exists: {os.path.exists(CNN_MODEL_PATH)}")
    
    print(f"\nHOG+SVM Model:")
    print(f"  Model: {HOG_SVM_MODEL_PATH}")
    print(f"  Scaler: {HOG_SCALER_PATH}")
    print(f"  HOG Config: {HOG_CONFIG}")
    print(f"  Exists: {os.path.exists(HOG_SVM_MODEL_PATH) and os.path.exists(HOG_SCALER_PATH)}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PCB ORIENTATION DETECTION - UNIFIED MODEL SUITE")
    print("="*60)
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            compare_models(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
        elif sys.argv[1] == "test":
            if len(sys.argv) > 2:
                test_single_image(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'both')
            else:
                print("Usage: python models.py test <image_path> [cnn|hog|both]")
        elif sys.argv[1] == "info":
            print_model_info()
    else:
        print("\nUsage:")
        print("  python models.py compare [count]  - Compare CNN vs HOG+SVM")
        print("  python models.py test <path> [type] - Test single image")
        print("  python models.py info             - Show model info")
        print("\nExample:")
        print("  python models.py compare 100")
        print("  python models.py test Data/Processed_data/Pass_data/image.jpg cnn")
