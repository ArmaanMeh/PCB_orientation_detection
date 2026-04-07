#!/usr/bin/env python3
"""
Live CNN Classification - Minimal, Fast, Robust
PCB Orientation Detection with real-time video feed
"""

import cv2
import numpy as np
import tensorflow as tf
import time

# Configuration
MODEL_PATH = "Export/ot_model.keras"
IMG_SIZE = 244
CLASS_LABELS = ["Fail", "Pass"]

# Best Hyperparameters (from 2-fold CV tuning - Config 5)
# These MUST match the hyperparameters used in main_cnn.ipynb for model training
HYPERPARAMS = {
    'filters_base': 64,      # Filters in first Conv layer
    'dropout_rate': 0.25,    # Dropout probability
    'learning_rate': 0.0005, # Adam optimizer learning rate
    'batch_size': 48         # Training batch size
}

def main():
    """Main classification loop."""
    print("="*60)
    print("PCB DETECTION - CNN Live Classification")
    print("="*60)
    
    # Load model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✓ Model loaded: {MODEL_PATH}\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🎥 Live classification started (press 'q' to quit)\n")
    frame_count = 0
    fps_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict
            img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(img_resized, axis=0)  # Rescaling layer in model handles normalization
            logits = model.predict(img_array, verbose=0)
            probs = tf.nn.softmax(logits[0]).numpy()  # Convert logits to probabilities
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            
            # FPS calculation
            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed > 0:
                fps = frame_count / elapsed
            else:
                fps = 0
            if elapsed > 1.0:
                frame_count = 0
                fps_time = time.time()
            
            # Draw results
            color = (0, 255, 0) if pred_idx == 1 else (0, 0, 255)
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, f"Result: {CLASS_LABELS[pred_idx]}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow("PCB Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"✗ Error during classification: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Classification ended")

if __name__ == "__main__":
    main()
