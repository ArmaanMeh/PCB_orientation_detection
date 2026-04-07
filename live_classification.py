#!/usr/bin/env python3
"""
Live CNN Classification - Minimal, Fast, Robust
PCB Orientation Detection with real-time video feed
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os

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
        print(f"✓ Model loaded: {MODEL_PATH}")
        
        # Verify model output shape
        if model.output_shape[-1] != len(CLASS_LABELS):
            print(f"✗ ERROR: Model outputs {model.output_shape[-1]} classes but expecting {len(CLASS_LABELS)}")
            return
        print(f"✓ Model output shape verified: {model.output_shape}\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Camera 1 not available, trying camera 0...")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Could not open webcam on any camera index")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✓ Webcam initialized")
    print("🎥 Live classification started")
    print("Controls: 'q' to quit | 's' to save frame\n")
    
    frame_count = 0
    fps_time = time.time()
    frames_saved = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame from webcam")
                break
            
            try:
                # Prepare image for model inference
                # Model has Rescaling(1./255) as first layer, so it expects uint8 input [0,255]
                img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                # Ensure dtype is uint8 (cv2.resize preserves it, but be explicit)
                if img_resized.dtype != np.uint8:
                    img_resized = img_resized.astype(np.uint8)
                img_array = np.expand_dims(img_resized, axis=0)
                
                # Get model prediction
                output = model.predict(img_array, verbose=0)
                
                # Detect if model output is raw logits or already activated
                # Model's last layer is Dense(num_classes) with NO activation, so output is logits
                # Check if we need to apply softmax (last layer is not softmax)
                final_layer = model.layers[-1]
                # Safely check activation name - guard against lambdas/partial functions
                activation = getattr(final_layer, 'activation', None)
                activation_name = getattr(activation, '__name__', None) if activation else None
                has_softmax = activation_name == 'softmax'
                
                if has_softmax:
                    # Output is already softmax probabilities
                    probs = output[0]
                else:
                    # Output is raw logits, convert to probabilities using softmax
                    probs = tf.nn.softmax(output[0]).numpy()
                
                pred_idx = np.argmax(probs)
                confidence = float(probs[pred_idx])
                
                # Validate predictions
                if not (0 <= pred_idx < len(CLASS_LABELS)):
                    print(f"✗ Invalid prediction index: {pred_idx}")
                    continue
                if not (0 <= confidence <= 1.0):
                    print(f"✗ Invalid confidence value: {confidence}")
                    continue
                
            except Exception as e:
                print(f"✗ Error during prediction: {e}")
                continue
            
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
            color = (0, 255, 0) if pred_idx == 1 else (0, 0, 255)  # Green for Pass, Red for Fail
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (500, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, f"Result: {CLASS_LABELS[pred_idx]}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow("PCB Detection - CNN Classification", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting application...")
                break
            elif key == ord('s'):
                # Save frame
                try:
                    frames_saved += 1
                    filename = f"pcb_detection_{frames_saved}_{int(time.time())}.jpg"
                    
                    # Ensure Export directory exists
                    os.makedirs("Export", exist_ok=True)
                    filepath = f"Export/{filename}"
                    
                    # Save and check for success
                    success = cv2.imwrite(filepath, frame)
                    if success:
                        print(f"✓ Frame saved: {filename}")
                    else:
                        print(f"✗ Failed to save frame: {filename} (cv2.imwrite returned False)")
                        frames_saved -= 1  # Decrement counter if save failed
                except Exception as e:
                    print(f"✗ Error saving frame: {e}")
                    frames_saved -= 1
    
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
    except Exception as e:
        print(f"✗ Error during classification: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Classification ended (Saved {frames_saved} frames)")

if __name__ == "__main__":
    main()
