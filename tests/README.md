# HealthML-Toolkit Test Suite

This directory contains unit and integration tests for all three modules.

## Structure

```
tests/
├── __init__.py
├── test_chatbot.py       # Chatbot pattern matching tests
├── test_imputation.py    # Data quality imputation tests
├── test_vision.py        # Vision pipeline tests
├── fixtures/             # Test data
│   ├── sample_patient_data.csv
│   └── sample_retina_image.jpg
└── conftest.py          # pytest configuration
```

## Running Tests

### All Tests
```bash
pytest tests/
```

### Specific Module
```bash
pytest tests/test_chatbot.py
pytest tests/test_imputation.py
pytest tests/test_vision.py
```

### With Coverage
```bash
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

### Verbose Output
```bash
pytest tests/ -v
```

## Test Categories

### Unit Tests
- Individual function testing
- Mock external dependencies
- Fast execution (<1s per test)

### Integration Tests
- End-to-end workflows
- Real dataset samples
- Model training (on small data)

### Coverage Goals
- **Core modules**: 85%+
- **Critical paths**: 95%+ (medical logic, Grad-CAM)
- **Total**: 80%+

## Writing New Tests

### Example: Imputation Test

```python
# tests/test_imputation.py
import pytest
import numpy as np
from sklearn.impute import KNNImputer

def test_knn_imputation():
    # Arrange
    data = np.array([[1, 2], [np.nan, 4], [5, 6]])
    expected_shape = (3, 2)
    
    # Act
    imputer = KNNImputer(n_neighbors=1)
    result = imputer.fit_transform(data)
    
    # Assert
    assert result.shape == expected_shape
    assert not np.isnan(result).any()
    assert result[1, 0] == pytest.approx(3.0, rel=1e-1)  # Should be ~3
```

### Example: Vision Test

```python
# tests/test_vision.py
import tensorflow as tf
from vision.cli_train_model import build_baseline_model

def test_baseline_model_creation():
    # Arrange
    num_classes = 5
    img_size = 224
    
    # Act
    model = build_baseline_model(num_classes, img_size)
    
    # Assert
    assert model.input_shape == (None, img_size, img_size, 3)
    assert model.output_shape == (None, num_classes)
    assert len(model.layers) > 3  # At least conv, pool, dense
```

## CI/CD Integration

Tests run automatically on:
- Push to `main` branch
- Pull request creation
- Pre-commit hook (optional)

See `.github/workflows/ci.yml` for configuration.

## Test Data

- **Sample images**: `fixtures/sample_retina_image.jpg` (224x224 RGB)
- **Sample CSV**: `fixtures/sample_patient_data.csv` (100 rows with missing values)
- **Mock responses**: `fixtures/chatbot_test_patterns.json`

## Troubleshooting

### TensorFlow Import Errors
```bash
# Use CPU-only TensorFlow for testing
pip install tensorflow-cpu
```

### Slow Tests
```bash
# Run only fast tests (skip integration)
pytest tests/ -m "not slow"
```

### Debugging Failed Tests
```bash
# Print output
pytest tests/ -s

# Drop into debugger on failure
pytest tests/ --pdb
```

---

**Keep tests green! 🟢✅**
