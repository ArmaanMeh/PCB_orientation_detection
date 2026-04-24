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
import traceback
import json
import pathlib

# Configuration
MODEL_PATH = "Export/ot_model.keras"
IMG_SIZE = 244

# Load class names dynamically to ensure they match training data
def load_class_labels():
    """Load class labels from training data or saved configuration."""
    # Try to load from saved class_names.json
    if os.path.exists("Export/class_names.json"):
        try:
            with open("Export/class_names.json", "r") as f:
                data = json.load(f)
                return data["class_names"]
        except Exception as e:
            print(f"Warning: Could not load class names from file: {e}")
    
    # Fallback: Try to determine from data directory
    data_dir = pathlib.Path('Data/Processed_data')
    if data_dir.exists():
        try:
            classes = sorted([d.name for d in data_dir.glob('*') if d.is_dir()])
            if classes:
                print(f"✓ Loaded class names from directory: {classes}")
                return classes
        except Exception as e:
            print(f"Warning: Could not load from directory: {e}")
    
    # Fallback: Hardcoded values
    print("Warning: Using fallback class labels")
    return ["Fail_data", "Pass_data"]

CLASS_LABELS = load_class_labels()

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
                print("Running prediction...")
                img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                # Ensure dtype is uint8 (cv2.resize preserves it, but be explicit)
                if img_resized.dtype != np.uint8:
                    img_resized = img_resized.astype(np.uint8)
                img_array = np.expand_dims(img_resized, axis=0)
                
                # Get model prediction (using raw model call for faster single-frame inference)
                output = model(img_array, training=False).numpy()
                
                # Model's last layer is Dense(num_classes) with NO activation, so output is logits
                # Convert raw logits to probabilities
                probs = tf.nn.softmax(output[0]).numpy()
                print("Raw probs:", probs)
                
                # CRITICAL FIX: Apply class weight adjustment for imbalanced dataset
                # New dataset has 4.21:1 ratio (Fail:Pass), so model is biased towards Fail
                # Apply inverse class weights to balance predictions
                # 1 Define the imbalance ratio.
                ratio = 2991 / 710
                tuning_factor = 1.2  # Adjust this factor to control the strength of correction
                # 2. Calculate inverse class weights
                # We want to multiply the 'Fail' probability by a small number 
                # and the 'Pass' probability by a larger number.
                # Weight for Fail = 1 / (ratio * tuning_factor)
                # Weight for Pass = 1.0
                class_weights_correction = np.array([1.0/(ratio * tuning_factor), 1.0])  # Weights inverse to dataset imbalance
                # 3. Normalize weights so they sum to 1 (prevents exploding probabilities)
                class_weights_correction = class_weights_correction / class_weights_correction.sum()
                # 4. Apply weights to the model's raw probability output
                weighted_probs = probs * class_weights_correction
                # 5. Re-normalize to ensure the final probabilities sum to 1.0
                weighted_probs = weighted_probs / weighted_probs.sum()

                print("Weighted probs:", weighted_probs)
                # 6. Extract prediction and confidence
                pred_idx = np.argmax(weighted_probs)
                confidence = float(weighted_probs[pred_idx])
                
                # Validate predictions
                if not (0 <= pred_idx < len(CLASS_LABELS)):
                    print(f"✗ Invalid prediction index: {pred_idx}")
                    continue
                if not (0 <= confidence <= 1.0):
                    print(f"✗ Invalid confidence value: {confidence}")
                    continue
                
            except Exception as e:
                print(f"✗ Error during prediction: {e}")
                traceback.print_exc()
                break
            
            # FPS calculation
            elapsed = time.time() - fps_time
            frame_count += 1
            if elapsed > 0:
                fps = frame_count / elapsed
            else:
                fps = 0
            if elapsed > 1.0:
                frame_count = 0
                fps_time = time.time()
        
            
            # Draw results
            # Determine color based on whether prediction contains "Pass" or "Fail"
            pred_class_name = CLASS_LABELS[pred_idx].lower()
            if "pass" in pred_class_name:
                color = (0, 255, 0)  # Green for Pass
            else:
                color = (0, 0, 255)  # Red for Fail
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
                    filename = f"pcb_detection_{frames_saved + 1}_{int(time.time())}.jpg"
                    
                    # Ensure Export directory exists
                    os.makedirs("Export", exist_ok=True)
                    filepath = os.path.join("Export", filename)
                    
                    # Save and check for success
                    success = cv2.imwrite(filepath, frame)
                    if success:
                        frames_saved += 1
                        print(f"✓ Frame saved: {filename}")
                    else:
                        print(f"✗ Failed to save frame: {filename} (cv2.imwrite returned False)")
                except Exception as e:
                    print(f"✗ Error saving frame: {e}")
    
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
    except Exception as e:
        print(f"✗ Error during classification: {e}")
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Classification ended (Saved {frames_saved} frames)")

if __name__ == "__main__":
    main()
