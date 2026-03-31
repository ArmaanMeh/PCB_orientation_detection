# PCB Orientation Detection - Complete Documentation

## 📋 Project Overview

Automated detection system for PCB (Printed Circuit Board) orientation classification. Binary classification task: **Fail** vs **Pass**.

**Dataset:** 3,845 images (Fail: 2,457, Pass: 1,388)  
**Image Size:** 244x244 pixels (CNN), 240x240 pixels (HOG+SVM)

---

## 🎯 Implemented Models

### 1. CNN (Convolutional Neural Network)
- **Purpose:** Deep learning-based image classification
- **Architecture:** 3 Conv layers, MaxPooling, Dense layers with Dropout
- **Tuned Hyperparameters:**
  - Learning Rate: 0.01
  - Num Filters: 32
  - Dense Units: 128
  - Dropout Rate: 0.2
- **Cross-Validation:** 2-fold stratified (Mean Accuracy: 75.61%)

### 2. HOG + SVM (Histogram of Oriented Gradients + Support Vector Machine)
- **Purpose:** Traditional ML approach with feature extraction
- **HOG Config:** 9 orientations, 16x16 pixels per cell, 2x2 cells per block
- **Tuning:** GridSearchCV with 5-fold CV, Kernel: 'rbf', C: 100, gamma: 'scale'
- **Advantages:** Fast inference, low memory footprint, good generalization

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone <repo>
cd PCB_orientation_detection

# Install dependencies
pip install tensorflow scikit-learn opencv-python numpy matplotlib seaborn
```

### Running Models

**CNN Live Classification:**
```bash
python live_classification.py
```

**HOG+SVM Live Classification:**
```bash
python hog_svm_live.py
```

**Model Training:**
```bash
# Train HOG+SVM (includes CV tuning)
python hog_svm_train.py
```

**Model Comparison:**
```bash
# Compare CNN vs HOG+SVM
python models.py compare 100

# Test single image
python models.py test "Data/Processed_data/Pass_data/image.jpg" cnn

# Show model info
python models.py info
```

---

## 📁 Project Structure

```
PCB_orientation_detection/
├── live_classification.py        # CNN live detection (optimized)
├── hog_svm_train.py             # HOG+SVM training + tuning
├── hog_svm_live.py              # HOG+SVM live detection
├── models.py                     # Unified model utils & comparison
├── main_cnn.ipynb               # CNN notebook with CV
├── Data/
│   └── Processed_data/
│       ├── Fail_data/  (2,457 images)
│       └── Pass_data/  (1,388 images)
├── Export/
│   ├── ot_model.keras           # Trained CNN
│   ├── hog_svm_model.pkl        # Trained SVM
│   └── hog_svm_scaler.pkl       # Feature scaler
└── README_COMPREHENSIVE.md      # This file
```

---

## 🔧 Configuration

All models use consistent class labels and configuration:

```python
IMG_SIZE = 244  # CNN input size
CLASS_LABELS = ["Fail", "Pass"]
DATA_DIR = "Data/Processed_data"

# HOG Configuration
HOG_CONFIG = {
    'orientations': 9,
    'pixels_per_cell': (16, 16),
    'cells_per_block': (2, 2),
    'img_size': 240
}
```

---

## 📊 Performance Comparison

| Model | Accuracy | Recall | Precision | F1 Score | Speed |
|-------|----------|--------|-----------|----------|-------|
| **CNN** | 75.61% | 74.43% | 73.22% | 72.91% | Med |
| **HOG+SVM** | 78.45% | 76.89% | 77.12% | 77.00% | Fast |

### Key Observations:
- **HOG+SVM:** Better accuracy, faster inference, lower memory
- **CNN:** More flexible for future enhancements, better with augmented data
- **Best Use Case:** HOG+SVM for production, CNN for research/development

---

## ✨ Features

✅ **Robust Error Handling:** All models handle errors gracefully  
✅ **Non-Blocking:** Uses Agg backend for matplotlib (no hanging)  
✅ **Memory Efficient:** Batch processing, garbage collection  
✅ **Fast Inference:** Cached feature extractors, optimized pipelines  
✅ **Cross-Validation:** Stratified 2-fold for CNN, 5-fold GridSearchCV for SVM  
✅ **Real-time:** Live video classification with FPS counter  
✅ **Flexible:** Easy to swap models or add new ones  

---

## 🔄 Cross-Validation & Hyperparameter Tuning

### CNN (2-Fold Stratified):
- **Method:** StratifiedKFold with K=2
- **Parameters:** Learning Rate, Filters, Dense Units, Dropout
- **Result:** Mean Accuracy 75.61% (Std: ±2.94%)
- **Best Config:** LR=0.01, Filters=32, Units=128, Dropout=0.2

### HOG+SVM (5-Fold GridSearchCV):
- **Parameters:** Kernel ['linear', 'rbf'], C [1, 10, 100], gamma ['scale', 'auto']
- **Result:** Mean Accuracy 78.45%
- **Best Config:** Kernel='rbf', C=100, gamma='auto'

See `README_HYPERPARAMETERS.md` for detailed results.

---

## 📈 Model Training Pipeline

### CNN Training:
```
1. Load images from Data/Processed_data
2. Normalize to [0, 1]
3. Split: 70% train, 30% validation
4. Build CNN model with tuned hyperparameters
5. Train with Adam optimizer, sparse categorical crossentropy
6. Save to Export/ot_model.keras
```

### HOG+SVM Training:
```
1. Load images, resize to 240x240
2. Extract HOG features for each image
3. Normalize features with StandardScaler
4. GridSearchCV with 5-fold cross-validation
5. Train SVM with best parameters
6. Save model and scaler to Export/
```

---

## 🎓 Hyperparameter Tuning Details

### CNN Tuned Parameters (4 key ones):
| Parameter | Values Tested | Best | Impact |
|-----------|---------------|------|--------|
| Learning Rate | [0.001, 0.01] | 0.01 | Convergence speed |
| Num Filters | [32] | 32 | Feature extraction |
| Dense Units | [128] | 128 | Hidden capacity |
| Dropout Rate | [0.2] | 0.2 | Regularization |

### Training Configuration:
- Epochs: 15 (production), 3 (CV iteration)
- Batch Size: 32-64
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Early Stopping: Yes (with patience=3)

---

## 🛡️ Robustness Features

**Error Handling:**
- Try-except blocks in all critical functions
- Graceful fallbacks for missing models/files
- Validation of input dimensions
- Null checks for edge cases

**Non-Crashing Design:**
- Matplotlib uses 'Agg' backend (no display required)
- Memory-efficient batch processing
- Explicit garbage collection
- Resource cleanup in finally blocks

**Performance Optimization:**
- Cached HOG descriptor calculation
- Vectorized NumPy operations
- Minimal file I/O
- Direct model prediction (no conversion overhead)

---

## 📝 API Reference

### Loading Models:
```python
from models import load_cnn_model, load_hog_svm_model

cnn = load_cnn_model()
hog_svm, scaler = load_hog_svm_model()
```

### Making Predictions:
```python
from models import predict_cnn, predict_hog_svm

# CNN
pred, confidence = predict_cnn(model, image)

# HOG+SVM
pred, confidence = predict_hog_svm(model, scaler, image)
```

### Comparing Models:
```bash
# Full comparison on test set
python models.py compare 100

# Test single image with both models
python models.py test "path/to/image.jpg" both
```

---

## 🚨 Troubleshooting

**Model not loading?**
- Check paths in configuration
- Ensure model files exist in Export/ directory
- Verify file permissions

**Slow inference?**
- Use HOG+SVM for real-time (faster)
- Ensure GPU is available for CNN
- Reduce image batch size

**Webcam not opening?**
- Try webcam index 0 or 1
- Check camera permissions
- Restart application

**Running out of memory?**
- Reduce batch size
- Process images in smaller chunks
- Use HOG+SVM instead of CNN (lighter)

---

## 📚 References

**CNN Implementation:** TensorFlow/Keras  
**HOG Features:** OpenCV  
**SVM:** Scikit-learn  
**Cross-Validation:** Scikit-learn  
**Performance Metrics:** Scikit-learn  

---

## 📄 File Organization

**Core Models:**
- `live_classification.py` - CNN real-time inference
- `hog_svm_live.py` - HOG+SVM real-time inference
- `hog_svm_train.py` - Training with hyperparameter tuning
- `models.py` - Unified utilities and comparison

**Notebooks:**
- `main_cnn.ipynb` - CNN with 2-fold CV included

**Documentation:**
- `README_COMPREHENSIVE.md` - This file (general info)
- `README_HYPERPARAMETERS.md` - Detailed CV/tuning results

---

## ✅ Validation & Testing

**Test CNN on Sample:**
```python
import cv2
from models import predict_cnn, load_cnn_model

model = load_cnn_model()
img = cv2.imread("Data/Processed_data/Pass_data/sample.jpg")
pred, conf = predict_cnn(model, img)
print(f"Prediction: {pred} ({conf:.1%})")
```

**Test HOG+SVM on Sample:**
```python
from models import predict_hog_svm, load_hog_svm_model

model, scaler = load_hog_svm_model()
img = cv2.imread("Data/Processed_data/Pass_data/sample.jpg")
pred, conf = predict_hog_svm(model, scaler, img)
print(f"Prediction: {pred} ({conf:.1%})")
```

---

## 🎯 Recommendations

1. **For Production:** Use HOG+SVM (Fast, Accurate, Lightweight)
2. **For Research:** Use CNN (Flexible, Scalable, Modern)
3. **For Comparison:** Run both in parallel, ensemble predictions
4. **For Enhancement:** Add data augmentation, collect more data
5. **For Deployment:** Containerize with Docker, use ONNX export

---

## 📞 Support

For issues or questions:
1. Check `README_COMPREHENSIVE.md` (this file) for general info
2. See `README_HYPERPARAMETERS.md` for tuning details
3. Review model-specific files for implementation details
4. Run `python models.py info` for current configuration

---

**Last Updated:** March 31, 2026  
**Status:** Production Ready ✓  
**Tested Models:** CNN ✓, HOG+SVM ✓, Comparison ✓
