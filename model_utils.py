"""
Model Utilities and Statistics
Quick model testing, statistics, and performance analysis
"""

import cv2
import numpy as np
import os
import pickle
import json
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 244
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
CLASS_LABELS = {0: "Fail", 1: "Pass"}

MODEL_PATH = "Export/hog_svm_model.pkl"
SCALER_PATH = "Export/hog_svm_scaler.pkl"
DATA_DIR = "Data/Processed_data"


# ==========================================
# UTILITY FUNCTIONS
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


def load_model():
    """Load trained model and scaler."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Model files not found in Export/")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


# ==========================================
# MODEL STATISTICS
# ==========================================
def get_model_statistics():
    """Get comprehensive model statistics."""
    print("\n" + "="*60)
    print("MODEL STATISTICS")
    print("="*60)
    
    try:
        model, scaler = load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Model info
    print("\nModel Information:")
    print(f"  Type: {type(model).__name__}")
    print(f"  Kernel: {model.kernel if hasattr(model, 'kernel') else 'N/A'}")
    print(f"  C: {model.C if hasattr(model, 'C') else 'N/A'}")
    print(f"  Gamma: {model.gamma if hasattr(model, 'gamma') else 'N/A'}")
    print(f"  Support Vectors: {len(model.support_vectors_) if hasattr(model, 'support_vectors_') else 'N/A'}")
    
    # Scaler info
    print("\nFeature Scaler Information:")
    print(f"  Type: {type(scaler).__name__}")
    print(f"  N Features: {len(scaler.mean_) if hasattr(scaler, 'mean_') else 'N/A'}")
    print(f"  Feature Mean (first 5): {scaler.mean_[:5] if hasattr(scaler, 'mean_') else 'N/A'}")
    print(f"  Feature Std (first 5): {scaler.scale_[:5] if hasattr(scaler, 'scale_') else 'N/A'}")
    
    # File sizes
    model_size = os.path.getsize(MODEL_PATH) / (1024*1024)  # MB
    scaler_size = os.path.getsize(SCALER_PATH) / (1024*1024)  # MB
    
    print("\nFile Information:")
    print(f"  Model file size: {model_size:.2f} MB")
    print(f"  Scaler file size: {scaler_size:.2f} MB")
    print(f"  Total size: {model_size + scaler_size:.2f} MB")
    
    # Configuration
    print("\nConfiguration:")
    print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  HOG Orientations: {HOG_ORIENTATIONS}")
    print(f"  HOG Pixels per cell: {HOG_PIXELS_PER_CELL}")
    print(f"  HOG Cells per block: {HOG_CELLS_PER_BLOCK}")


# ==========================================
# DATASET ANALYSIS
# ==========================================
def analyze_dataset():
    """Analyze dataset statistics."""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        return
    
    # Count images
    pass_dir = os.path.join(DATA_DIR, "Pass_data")
    fail_dir = os.path.join(DATA_DIR, "Fail_data")
    
    pass_count = 0
    fail_count = 0
    pass_size = 0
    fail_size = 0
    
    if os.path.exists(pass_dir):
        for f in os.listdir(pass_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                pass_count += 1
                pass_size += os.path.getsize(os.path.join(pass_dir, f)) / (1024*1024)
    
    if os.path.exists(fail_dir):
        for f in os.listdir(fail_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                fail_count += 1
                fail_size += os.path.getsize(os.path.join(fail_dir, f)) / (1024*1024)
    
    total = pass_count + fail_count
    
    print(f"\nPass data:")
    print(f"  Images: {pass_count}")
    print(f"  Size: {pass_size:.2f} MB")
    print(f"  Percentage: {100*pass_count/total:.1f}%" if total > 0 else "  Percentage: N/A")
    
    print(f"\nFail data:")
    print(f"  Images: {fail_count}")
    print(f"  Size: {fail_size:.2f} MB")
    print(f"  Percentage: {100*fail_count/total:.1f}%" if total > 0 else "  Percentage: N/A")
    
    print(f"\nTotal:")
    print(f"  Images: {total}")
    print(f"  Size: {pass_size + fail_size:.2f} MB")
    
    # Balance check
    if total > 0:
        min_count = min(pass_count, fail_count)
        max_count = max(pass_count, fail_count)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nClass Balance:")
        print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2:
            print("  ⚠️  Dataset is significantly imbalanced")
        elif imbalance_ratio > 1.5:
            print("  ⚠️  Dataset has slight imbalance")
        else:
            print("  ✓ Dataset is well-balanced")


# ==========================================
# BATCH TESTING
# ==========================================
def batch_test_dataset():
    """Test model on entire dataset."""
    print("\n" + "="*60)
    print("BATCH TESTING ON DATASET")
    print("="*60)
    
    try:
        model, scaler = load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    y_true = []
    y_pred = []
    
    print("\nTesting on Pass data...")
    pass_dir = os.path.join(DATA_DIR, "Pass_data")
    if os.path.exists(pass_dir):
        for img_file in os.listdir(pass_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(pass_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        features = extract_hog_features(img).reshape(1, -1)
                        features_scaled = scaler.transform(features)
                        pred = model.predict(features_scaled)[0]
                        
                        y_true.append(1)
                        y_pred.append(pred)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    
    print("Testing on Fail data...")
    fail_dir = os.path.join(DATA_DIR, "Fail_data")
    if os.path.exists(fail_dir):
        for img_file in os.listdir(fail_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(fail_dir, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        features = extract_hog_features(img).reshape(1, -1)
                        features_scaled = scaler.transform(features)
                        pred = model.predict(features_scaled)[0]
                        
                        y_true.append(0)
                        y_pred.append(pred)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
    
    if len(y_true) == 0:
        print("No test data found")
        return
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Results
    print(f"\nResults on {len(y_true)} test images:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Per-class accuracy
    pass_correct = np.sum((y_true == 1) & (y_pred == 1))
    pass_total = np.sum(y_true == 1)
    fail_correct = np.sum((y_true == 0) & (y_pred == 0))
    fail_total = np.sum(y_true == 0)
    
    print(f"\nPer-class accuracy:")
    print(f"  Pass: {pass_correct}/{pass_total} ({100*pass_correct/pass_total:.1f}%)" if pass_total > 0 else "  Pass: No samples")
    print(f"  Fail: {fail_correct}/{fail_total} ({100*fail_correct/fail_total:.1f}%)" if fail_total > 0 else "  Fail: No samples")


# ==========================================
# INFERENCE SPEED TEST
# ==========================================
def measure_inference_speed(num_iterations=100):
    """Measure model inference speed."""
    print("\n" + "="*60)
    print("INFERENCE SPEED TEST")
    print("="*60)
    
    try:
        model, scaler = load_model()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Create random test data
    print(f"\nTesting with {num_iterations} random images...")
    
    times = []
    
    for i in range(num_iterations):
        # Create random image
        img = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        
        # Measure feature extraction + prediction time
        start = time.time()
        
        features = extract_hog_features(img)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)
        
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)
    
    times = np.array(times)
    
    print(f"\nTiming Results:")
    print(f"  Mean: {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min: {times.min():.2f} ms")
    print(f"  Max: {times.max():.2f} ms")
    print(f"  Std: {times.std():.2f} ms")
    
    fps = 1000 / times.mean()
    print(f"\n  Estimated FPS: {fps:.1f}")
    
    # Plot timing distribution
    plt.figure(figsize=(10, 5))
    plt.hist(times, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Inference Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Inference Time Distribution')
    plt.axvline(times.mean(), color='r', linestyle='--', label=f'Mean: {times.mean():.2f} ms')
    plt.axvline(np.median(times), color='g', linestyle='--', label=f'Median: {np.median(times):.2f} ms')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Export/inference_timing.png', dpi=150)
    plt.show()
    
    print("\n✓ Timing distribution saved to Export/inference_timing.png")


# ==========================================
# SAVE STATISTICS REPORT
# ==========================================
def save_statistics_report():
    """Generate and save a statistics report."""
    print("\n" + "="*60)
    print("GENERATING STATISTICS REPORT")
    print("="*60)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": MODEL_PATH,
        "data_dir": DATA_DIR,
    }
    
    # File sizes
    try:
        report["model_size_mb"] = os.path.getsize(MODEL_PATH) / (1024*1024)
        report["scaler_size_mb"] = os.path.getsize(SCALER_PATH) / (1024*1024)
    except:
        report["model_size_mb"] = "N/A"
        report["scaler_size_mb"] = "N/A"
    
    # Save report
    report_path = "Export/model_statistics_report.json"
    os.makedirs("Export", exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to {report_path}")
    print(json.dumps(report, indent=2))


# ==========================================
# MAIN MENU
# ==========================================
def main_menu():
    """Interactive menu for model utilities."""
    print("\n" + "="*60)
    print("HOG+SVM MODEL UTILITIES")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("  1. Model Statistics")
        print("  2. Dataset Analysis")
        print("  3. Batch Test on Dataset")
        print("  4. Inference Speed Test")
        print("  5. Save Statistics Report")
        print("  6. Run All Tests")
        print("  0. Exit")
        
        choice = input("\nSelect option (0-6): ").strip()
        
        if choice == '1':
            get_model_statistics()
        elif choice == '2':
            analyze_dataset()
        elif choice == '3':
            batch_test_dataset()
        elif choice == '4':
            num_iter = input("Number of iterations (default 100): ").strip()
            try:
                num_iter = int(num_iter) if num_iter else 100
            except:
                num_iter = 100
            measure_inference_speed(num_iter)
        elif choice == '5':
            save_statistics_report()
        elif choice == '6':
            print("\nRunning all tests...")
            get_model_statistics()
            analyze_dataset()
            batch_test_dataset()
            measure_inference_speed()
            save_statistics_report()
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid option")


if __name__ == "__main__":
    main_menu()
