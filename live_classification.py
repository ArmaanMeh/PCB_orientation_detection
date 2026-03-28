import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Configuration
MODEL_PATH = "Export/ot_model.keras"
IMG_SIZE = 244
CONFIDENCE_THRESHOLD = 0.5

# Class labels (adjust based on your data)
CLASS_LABELS = ["Fail", "Pass"]

def main():
    # Load the trained model
    print("Loading model...")
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Started live classification. Press 'q' to quit...")
    
    frame_count = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Prepare frame for model (resize and normalize)
        input_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        input_array = np.expand_dims(input_frame, axis=0)
        
        # Make prediction
        logits = model.predict(input_array, verbose=0)
        probabilities = tf.nn.softmax(logits[0]).numpy()
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_LABELS[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            fps_start_time = time.time()
        else:
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Display results on frame
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Classification result color (green for pass, red for fail)
        color = (0, 255, 0) if predicted_class_idx == 1 else (0, 0, 255)
        
        # Display prediction
        cv2.putText(
            frame,
            f"Prediction: {predicted_class}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            2
        )
        
        # Display confidence
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2%}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )
        
        # Show the frame
        cv2.imshow("PCB Orientation Detection", frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Live classification ended.")

if __name__ == "__main__":
    main()
