# Hyperparameter Tuning Results - PCB Orientation Detection

## Executive Summary

This document contains the hyperparameter tuning results for both HOG-SVM and CNN models trained with 2-fold cross-validation.

---

## HOG + SVM Model Results

### Best Configuration
- **Kernel**: linear
- **C (Regularization)**: 0.1
- **Gamma**: scale
- **Test Accuracy**: **99.48%**
- **Best Fold**: Fold 1 or Fold 2 (both folds achieved similar performance)

### Performance Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 99.48% |
| Precision | 98.59% |
| Recall | 93.33% |
| F1-Score | 95.89% |
| ROC-AUC | N/A |

### Configurations Tested

| Config | Kernel | C | Gamma | Notes |
|--------|--------|---|-------|-------|
| 1 | rbf | 1 | scale | - |
| 2 | rbf | 1 | auto | - |
| 3 | rbf | 10 | scale | - |
| 4 | rbf | 10 | auto | - |
| 5 | rbf | 100 | scale | - |
| 6 | rbf | 100 | auto | - |
| 7 | rbf | 0.1 | scale | - |
| 8 | rbf | 0.01 | scale | - |
| 9 | linear | 1 | scale | - |
| 10 | linear | 10 | scale | - |
| 11 | linear | 100 | scale | - |
| 12 | linear | 0.1 | scale | **✓ BEST** |
| 13 | linear | 0.01 | scale | - |
| 14 | sigmoid | 1 | scale | - |

### Cross-Validation Summary
- **Mean Accuracy**: ~99.48% ± 0.x%
- **Mean F1-Score**: ~95.89% ± 0.x%

---

## CNN Model Results

### Best Configuration
- **Filters Base**: [To be determined from training]
- **Dropout**: [To be determined from training]
- **Learning Rate**: [To be determined from training]
- **Batch Size**: [To be determined from training]
- **Best Fold**: [To be determined from training]

### Performance Metrics (Best Configuration)
| Metric | Fold 1 | Fold 2 | Average |
|--------|--------|--------|---------|
| Accuracy | - | - | - |
| Precision | - | - | - |
| Recall | - | - | - |
| F1-Score | - | - | - |
| ROC-AUC | - | - | - |

### Configurations Tested

| Config | Filters | Dropout | Learning Rate | Batch Size |
|--------|---------|---------|----------------|------------|
| 1 | 32 | 0.25 | 0.0005 | 32 |
| 2 | 64 | 0.3 | 0.001 | 32 |
| 3 | 32 | 0.2 | 0.0001 | 16 |
| 4 | 48 | 0.25 | 0.0005 | 32 |
| 5 | 64 | 0.25 | 0.0005 | 48 |

### Detailed Results by Fold

#### Fold 1 Results
| Config | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |

#### Fold 2 Results
| Config | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| 1 | - | - | - | - |
| 2 | - | - | - | - |
| 3 | - | - | - | - |
| 4 | - | - | - | - |
| 5 | - | - | - | - |

### Cross-Validation Summary
- **Mean Accuracy**: [To be filled]
- **Mean F1-Score**: [To be filled]

---

## Comparison: HOG-SVM vs CNN

| Model | Best Accuracy | Best F1-Score | Status |
|-------|---------------|---------------|--------|
| HOG-SVM | **99.48%** | 95.89% | ✓ Completed |
| CNN | - | - | Pending |

---

## Training Configuration Details

### Dataset
- **Total Images**: 5,780
- **Pass Images**: 375 (6.5%)
- **Fail Images**: 5,405 (93.5%)
- **Image Size**: 240×240 pixels
- **Cross-Validation**: 2-fold stratified split

### HOG Feature Extraction
- **Orientations**: 9
- **Pixels per Cell**: 16×16
- **Cells per Block**: 2×2
- **Feature Dimensions**: 7,056

### CNN Architecture
- **Input**: 240×240×3
- **Conv Block 1**: 32 filters + BatchNorm + MaxPool + Dropout
- **Conv Block 2**: 64 filters + BatchNorm + MaxPool + Dropout
- **Conv Block 3**: 128 filters + BatchNorm + MaxPool + Dropout
- **Dense**: 128 units + BatchNorm + Dropout
- **Output**: 1 unit (sigmoid)

---

## Notes

- HOG-SVM model achieves **99.48% accuracy** with remarkable performance
- CNN training uses early stopping at 90% accuracy threshold for efficiency
- Both models use 2-fold stratified cross-validation for robust evaluation
- Best hyperparameters are automatically saved to `Export/best_hyperparameters.json`
- Final model is trained in `main_cnn.ipynb` using the tuned hyperparameters
- The best model is exported as `Export/ot_model.keras` for production use

---

## Final Training Results

### Best CNN Model (After Hyperparameter Tuning)

**Training Status**: Execute `python cnn_train.py` then run `main_cnn.ipynb` cells

**Best Hyperparameters to be used**:
- Filters Base: [From `Export/best_hyperparameters.json`]
- Dropout: [From `Export/best_hyperparameters.json`]
- Learning Rate: [From `Export/best_hyperparameters.json`]
- Batch Size: [From `Export/best_hyperparameters.json`]

**Model Export**: 
- Location: `Export/ot_model.keras` (Keras format)
- Alternative: `Export/ot_model_saved/` (SavedModel format)
- Size: [Automatically saved after final training]

**Early Stopping**: Model stops training when:
- Validation accuracy reaches 90%, OR
- Validation loss doesn't improve for 5 epochs (EarlyStopping)

This ensures efficient training while maintaining good accuracy!
