"""
Live Video Classification using HOG + SVM Model
Real-time PCB orientation detection with performance metrics
OPTIMIZED: Uses cached HOGDescriptor for maximum performance
"""

import cv2
import numpy as np
import pickle
import os
import time

# Configuration
MODEL_PATH = "Export/hog_svm_model.pkl"
SCALER_PATH = "Export/hog_svm_scaler.pkl"
IMG_SIZE = 244
CLASS_LABELS = {0: "Fail", 1: "Pass"}

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
CONFIDENCE_THRESHOLD = 0.5


# ==========================================
# HOG DESCRIPTOR CONFIGURATION
# ==========================================
def create_hog_descriptor():
    """
    Create a properly configured HOG descriptor with stable parameters.
    Must match the training configuration exactly.
    
    Returns:
        Configured HOGDescriptor instance
    """
    try:
        # HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
        #               derivAperture=1, winSigma=-1, histogramNormType=0,
        #               L2HysThreshold=0.2, gammaCorrection=True, nlevels=64)
        hog = cv2.HOGDescriptor(
            (IMG_SIZE, IMG_SIZE),      # winSize (240x240)
            (32, 32),                  # blockSize (2 cells of 16x16)
            (16, 16),                  # blockStride (stride by 1 cell)
            (16, 16),                  # cellSize
            HOG_ORIENTATIONS           # nbins (9)
        )
        return hog
    except Exception as e:
        print(f"Error creating HOGDescriptor: {e}")
        return cv2.HOGDescriptor()


# Global HOG descriptor (created once for efficiency)
HOG_DESCRIPTOR = create_hog_descriptor()


# ==========================================
# HOG FEATURE EXTRACTION (optimized)
# ==========================================
def extract_hog_features(image):
    """
    Extract HOG features from an image - optimized with cached HOGDescriptor.
    
    Args:
        image: Input image (BGR format from cv2)
    
    Returns:
        HOG feature vector
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if image.shape != (IMG_SIZE, IMG_SIZE):
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Use cached HOGDescriptor for performance
    features = HOG_DESCRIPTOR.compute(image, winStride=(8, 8), padding=(0, 0))
    features = features.flatten()
    
    return features


# ==========================================
# MODEL LOADING
# ==========================================
def load_model_and_scaler():
    """Load trained HOG+SVM model and scaler."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Model or scaler not found.\n"
            f"Expected paths:\n  {MODEL_PATH}\n  {SCALER_PATH}\n"
            f"Please train the model first using: python hog_svm_train.py"
        )
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Scaler loaded from {SCALER_PATH}")
    
    return model, scaler


# ==========================================
# INFERENCE
# ==========================================
def predict_frame(frame, model, scaler):
    """
    Predict class and confidence for a frame.
    
    Args:
        frame: Input frame (BGR format)
        model: Trained SVM model
        scaler: Feature scaler
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Resize frame
        img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        
        # Extract HOG features
        features = extract_hog_features(img_resized)
        features = features.reshape(1, -1)
        
        # Normalize features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        confidence = np.max(probabilities)
        
        result = {
            'class_idx': int(prediction),
            'class_name': CLASS_LABELS[int(prediction)],
            'confidence': float(confidence),
            'probabilities': probabilities,
            'success': True
        }
    
    except Exception as e:
        print(f"Error during inference: {e}")
        result = {
            'success': False,
            'error': str(e)
        }
    
    return result


# ==========================================
# VISUALIZATION
# ==========================================
def draw_results_on_frame(frame, prediction_result, fps):
    """
    Draw prediction results on frame.
    
    Args:
        frame: Input frame
        prediction_result: Dictionary with prediction results
        fps: Frames per second
    
    Returns:
        Frame with drawn results
    """
    if not prediction_result['success']:
        return frame
    
    class_idx = prediction_result['class_idx']
    class_name = prediction_result['class_name']
    confidence = prediction_result['confidence']
    
    # Create semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (520, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Choose color based on prediction (green for pass, red for fail)
    color = (0, 255, 0) if class_idx == 1 else (0, 0, 255)
    color_text = (255, 255, 255)
    
    # Draw prediction
    cv2.putText(
        frame,
        f"Class: {class_name}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        color,
        2
    )
    
    # Draw confidence
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2%}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color_text,
        2
    )
    
    # Draw class probabilities
    failed_prob = prediction_result['probabilities'][0]
    passed_prob = prediction_result['probabilities'][1]
    
    cv2.putText(
        frame,
        f"Fail: {failed_prob:.2%} | Pass: {passed_prob:.2%}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color_text,
        2
    )
    
    # Draw FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )
    
    return frame


# ==========================================
# MAIN LIVE CLASSIFICATION
# ==========================================
def main():
    """Main live classification loop."""
    print("\n" + "="*60)
    print("HOG + SVM LIVE CLASSIFICATION")
    print("="*60)
    
    # Load model and scaler
    print("\nLoading model and scaler...")
    try:
        model, scaler = load_model_and_scaler()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return
    
    print("✓ Model and scaler loaded successfully")
    
    # Initialize webcam
    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Webcam initialized")
    print("\nStarted live classification")
    print("Press 'q' to quit | 's' to save frame | 'r' to reset FPS counter")
    print("="*60 + "\n")
    
    # Initialization for FPS calculation
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    frames_saved = 0
    
    # Main loop
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from webcam")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Make prediction
        prediction_result = predict_frame(frame, model, scaler)
        
        # Calculate and update FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()
        else:
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Draw results on frame
        frame_with_results = draw_results_on_frame(frame, prediction_result, fps)
        
        # Display frame
        cv2.imshow("HOG+SVM PCB Orientation Detection", frame_with_results)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting application...")
            break
        elif key == ord('s'):
            # Save current frame
            frames_saved += 1
            filename = f"capture_{frames_saved}_{int(time.time())}.jpg"
            save_path = os.path.join("Export", filename)
            os.makedirs("Export", exist_ok=True)
            cv2.imwrite(save_path, frame_with_results)
            print(f"Frame saved: {filename}")
        elif key == ord('r'):
            # Reset FPS counter
            frame_count = 0
            fps_start_time = time.time()
            fps = 0
            print("FPS counter reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print(f"Session Summary:")
    print(f"  Total frames saved: {frames_saved}")
    print("="*60 + "\n")


# ==========================================
# BATCH PREDICTION ON IMAGES
# ==========================================
def predict_images_in_folder(folder_path, model, scaler):
    """
    Predict on all images in a folder.
    
    Args:
        folder_path: Path to folder containing images
        model: Trained SVM model
        scaler: Feature scaler
    """
    print(f"\nPredicting on images in {folder_path}...")
    
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder not found: {folder_path}")
        return
    
    models_loaded, scaler_loaded = load_model_and_scaler()
    
    results = []
    total_images = 0
    correct_predictions = 0
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(folder_path, filename)
            
            try:
                image = cv2.imread(filepath)
                if image is None:
                    print(f"Could not read: {filename}")
                    continue
                
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                prediction_result = predict_frame(image, model, scaler)
                
                if prediction_result['success']:
                    total_images += 1
                    results.append({
                        'filename': filename,
                        'prediction': prediction_result['class_name'],
                        'confidence': prediction_result['confidence']
                    })
                    
                    print(f"{filename}: {prediction_result['class_name']} ({prediction_result['confidence']:.2%})")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"\nProcessed {total_images} images")
    
    return results


if __name__ == "__main__":
    main()
