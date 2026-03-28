# HOG + SVM Model for PCB Orientation Detection

Complete machine learning pipeline for PCB orientation classification using Histogram of Oriented Gradients (HOG) features and Support Vector Machine (SVM) classifier.

## 📋 Project Structure

```
PCB_orientation_detection/
├── hog_svm_train.py          # Train HOG+SVM model (main training script)
├── hog_svm_live.py           # Live video classification with HOG+SVM
├── live_classification.py    # Live video classification with CNN
├── compare_models.py         # Compare CNN vs HOG+SVM models
├── Data/
│   ├── Processed_data/
│   │   ├── Pass_data/       # Images with correct orientation
│   │   └── Fail_data/       # Images with incorrect orientation
│   └── Raw_data/            # Raw video files
├── Export/
│   ├── ot_model.keras       # Pre-trained CNN model
│   ├── hog_svm_model.pkl    # Trained HOG+SVM model
│   └── hog_svm_scaler.pkl   # Feature scaler for HOG+SVM
└── README.md
```

## 🚀 Quick Start

### Prerequisites

Install required dependencies:

```bash
pip install opencv-python numpy scikit-learn tensorflow matplotlib seaborn tqdm
```

### Training HOG+SVM Model

Train the HOG+SVM model on your PCB orientation data:

```bash
python hog_svm_train.py
```

**What this script does:**
- Loads all images from `Data/Processed_data/`
- Extracts HOG (Histogram of Oriented Gradients) features from each image
- Performs hyperparameter optimization using GridSearchCV
- Trains SVM classifier on the extracted features
- Evaluates model on train/validation/test sets
- Performs 5-fold cross-validation
- Generates confusion matrices and ROC curves
- Saves the trained model and scaler for later use

**Output:**
- `Export/hog_svm_model.pkl` - Trained SVM model
- `Export/hog_svm_scaler.pkl` - Feature scaler
- Visualizations in `Export/` directory

### Live Classification with HOG+SVM

Run real-time classification using your webcam:

```bash
python hog_svm_live.py
```

**Controls:**
- **q**: Quit application
- **s**: Save current frame with prediction
- **r**: Reset FPS counter

**What you'll see:**
- Class prediction (Pass/Fail) with color coding (green/red)
- Confidence score
- Individual probability scores
- Real-time FPS counter

### Live Classification with CNN

Run real-time classification using the pre-trained CNN:

```bash
python live_classification.py
```

### Compare Models

Compare performance of CNN and HOG+SVM models:

```bash
python compare_models.py
```

**This generates:**
- Side-by-side confusion matrices
- Metrics comparison bar charts
- Performance summary

## 📊 HOG+SVM Training Script Details

### Features

#### 1. **Data Loading**
- Automatically loads images from directory structure
- Supports JPG, JPEG, PNG formats
- Resizes all images to 244×244 pixels
- Maintains class balance information

#### 2. **Feature Extraction (HOG)**
- Extracts 1764-dimensional feature vectors
- Configuration:
  - Orientations: 9
  - Pixels per cell: 16×16
  - Cells per block: 2×2
- Applied to grayscale images

#### 3. **Data Splitting**
- 60% Training data
- 20% Validation data
- 20% Test data
- Stratified split to maintain class balance

#### 4. **Hyperparameter Optimization**
GridSearchCV searches over:
- **C** (regularization): [0.1, 1, 10, 100]
- **kernel**: ['rbf', 'linear']
- **gamma**: ['scale', 'auto', 0.001, 0.01]

Uses 5-fold cross-validation with F1-score as scoring metric.

#### 5. **Evaluation Metrics**
- **Accuracy**: Overall correctness
- **Precision**: False positive rate control
- **Recall**: False negative rate control
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification breakdown
- **Classification Report**: Per-class metrics

#### 6. **Cross-Validation**
- 5-fold cross-validation with multiple metrics
- Calculates mean and standard deviation for each metric
- Helps assess model generalization

#### 7. **Visualizations**
- Confusion matrix heatmap
- ROC curve with AUC score
- Performance metrics comparison

### Configuration Parameters

Edit in `hog_svm_train.py`:

```python
IMG_SIZE = 244                    # Image size (must match training)
HOG_ORIENTATIONS = 9             # Number of orientation bins
HOG_PIXELS_PER_CELL = (16, 16)   # Pixels per cell
HOG_CELLS_PER_BLOCK = (2, 2)     # Cells per block
TEST_SPLIT = 0.2                 # Test set proportion
VAL_SPLIT = 0.2                  # Validation set proportion
RANDOM_STATE = 42                # Random seed for reproducibility
```

## 📊 Live Classification Script Details

### HOG+SVM Live Classification (`hog_svm_live.py`)

**Features:**
- Loads trained model from disk
- Captures frames from webcam in real-time
- Extracts HOG features from each frame
- Makes predictions using SVM
- Displays results with confidence scores
- Shows class probabilities for both classes
- Calculates and displays FPS
- Saves captured frames with predictions
- Includes error handling and validation

**Performance:**
- Real-time processing at 15-30 FPS (depends on system)
- Efficient feature extraction
- Minimal latency for inference

### CNN Live Classification (`live_classification.py`)

**Features:**
- Loads pre-trained Keras model
- Real-time webcam capture and classification
- Text overlay with predictions
- Confidence display
- FPS counter

## 🧪 Model Comparison Script

### `compare_models.py`

Compares CNN and HOG+SVM models side-by-side:

**Outputs:**
1. Individual metrics for each model
2. Confusion matrices comparison
3. Bar chart comparing accuracy, precision, recall, F1-score
4. Sample predictions from both models
5. Agreement analysis between models

**Use cases:**
- Verify HOG+SVM model performance
- Compare with existing CNN baseline
- Identify which model is more suitable for your use case
- Visualize performance differences

## 🔧 Advanced Usage

### Custom Inference Function

To use the HOG+SVM model in your own code:

```python
from hog_svm_train import extract_hog_features, load_model_from_disk
import cv2

# Load model
model, scaler = load_model_from_disk("Export")

# Load and preprocess image
img = cv2.imread("path/to/image.jpg")
img = cv2.resize(img, (244, 244))

# Extract features
features = extract_hog_features(img)
features = features.reshape(1, -1)

# Predict
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)[0]
confidence = model.predict_proba(features_scaled)[0]

print(f"Class: {['Fail', 'Pass'][int(prediction)]}")
print(f"Confidence: {confidence[int(prediction)]:.2%}")
```

### Batch Prediction

```python
from hog_svm_live import predict_images_in_folder, load_model_and_scaler

model, scaler = load_model_and_scaler()
results = predict_images_in_folder("path/to/images", model, scaler)
```

### Extract HOG Features Only

```python
from hog_svm_train import extract_features_batch
import numpy as np

features = extract_features_batch(images)  # Input: array of images
print(f"Feature shape: {features.shape}")  # Output: (n_samples, 1764)
```

## 📈 Expected Performance

Based on the training pipeline:

| Metric | Typical Range |
|--------|---------------|
| Accuracy | 85-95% |
| Precision | 84-94% |
| Recall | 85-95% |
| F1-Score | 85-94% |
| ROC-AUC | 0.90-0.98 |

*Actual performance depends on data quality, quantity, and diversity*

## ⚙️ Troubleshooting

### Model not found
```
FileNotFoundError: Model or scaler not found
```
**Solution:** Run `hog_svm_train.py` first to train the model

### Webcam not opening
```
Error: Could not open webcam
```
**Solutions:**
- Check if another application is using the webcam
- Ensure camera permissions are granted
- Try device ID 1 instead of 0: `cap = cv2.VideoCapture(1)`

### Out of memory during training
**Solutions:**
- Reduce batch size (if applicable)
- Process fewer images at once
- Increase available RAM

### Low FPS in live classification
**Solutions:**
- Close background applications
- Reduce webcam resolution
- Use HOG+SVM (faster than CNN)

## 🔍 Model Selection Guide

### Use CNN when:
- You have large dataset (>5000 images)
- GPU acceleration is available
- Maximum accuracy is priority
- Real-time processing not critical

### Use HOG+SVM when:
- Dataset is smaller (100-2000 images)
- CPU-only processing needed
- Faster training is important
- Good explainability desired
- Lightweight deployment required

## 📝 Model Characteristics

### HOG+SVM
**Advantages:**
- Fast training (minutes vs hours)
- Works well with small-medium datasets
- CPU efficient
- Interpretable features
- Lower processing latency
- Robust to image variations

**Disadvantages:**
- May require feature tuning
- Less flexible than deep learning
- Struggle with very large datasets

### CNN (Pre-trained)
**Advantages:**
- Superior accuracy on large datasets
- Automatic feature learning
- Better for complex patterns
- Transfer learning possible

**Disadvantages:**
- Slower training
- Requires more data
- Higher computational cost
- Less interpretable

## 📚 References

- HOG Features: https://scikit-image.org/docs/dev/api/skimage.feature.html#hog
- SVM Classifier: https://scikit-learn.org/stable/modules/svm.html
- OpenCV: https://opencv.org/
- Scikit-learn: https://scikit-learn.org/

## 📄 License

See LICENSE file in repository

## ✉️ Support

For issues or questions:
1. Check troubleshooting section
2. Verify data format and directory structure
3. Ensure all dependencies are installed
4. Check model files exist in Export/ directory
