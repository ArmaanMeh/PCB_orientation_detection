# Hyperparameter Tuning Results - 2-Fold Cross-Validation

**Generated:** 2026-03-31 09:30:01

## Tuned Hyperparameters

- **Learning Rate:** [0.001, 0.01]
- **Number of Filters:** [32]
- **Dense Units:** [128]
- **Dropout Rate:** [0.2]

**Total Combinations Tested:** 2
**Cross-Validation Folds:** 2
**Total Results:** 4

## Best Hyperparameters Found

```
learning_rate: 0.001
num_filters: 32
dense_units: 128
dropout_rate: 0.2
Best Mean Accuracy: 0.6391
```

## Detailed Results by Hyperparameter Combination

### Configuration: {'learning_rate': 0.001, 'num_filters': 32, 'dense_units': 128, 'dropout_rate': 0.2}

| Metric | Fold 1 | Fold 2 | Mean |
|--------|--------|--------|------|
| Accuracy | 0.6391 | 0.6389 | 0.6390 |
| Recall | 0.0000 | 0.0000 | 0.0000 |
| Precision | 0.0000 | 0.0000 | 0.0000 |
| F1 Score | 0.0000 | 0.0000 | 0.0000 |

### Configuration: {'learning_rate': 0.01, 'num_filters': 32, 'dense_units': 128, 'dropout_rate': 0.2}

| Metric | Fold 1 | Fold 2 | Mean |
|--------|--------|--------|------|
| Accuracy | 0.6391 | 0.6389 | 0.6390 |
| Recall | 0.0000 | 0.0000 | 0.0000 |
| Precision | 0.0000 | 0.0000 | 0.0000 |
| F1 Score | 0.0000 | 0.0000 | 0.0000 |

## Cross-Validation Fold Summary

### Fold 1

**Best Performance in this Fold:**

```
Configuration: {'learning_rate': 0.001, 'num_filters': 32, 'dense_units': 128, 'dropout_rate': 0.2}
Accuracy:  0.6391
Recall:    0.0000
Precision: 0.0000
F1 Score:  0.0000
```

### Fold 2

**Best Performance in this Fold:**

```
Configuration: {'learning_rate': 0.001, 'num_filters': 32, 'dense_units': 128, 'dropout_rate': 0.2}
Accuracy:  0.6389
Recall:    0.0000
Precision: 0.0000
F1 Score:  0.0000
```

## Recommendations

Based on 2-fold cross-validation results, the recommended hyperparameters are:

```python
model = build_model(
    learning_rate=0.001,
    num_filters=32,
    dense_units=128,
    dropout_rate=0.2,
)
```

## Notes

- All models trained for 3 epochs with batch size 32
- Data was split using stratified 2-fold cross-validation
- Each hyperparameter combination was tested on both folds
- Metrics calculated for binary classification (Fail vs Pass)
- Total images in dataset: 3845 (Fail: 2457, Pass: 1388)
