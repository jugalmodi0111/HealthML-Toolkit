# Contributing to HealthML-Toolkit

Thank you for your interest in contributing to **HealthML-Toolkit**! This document provides guidelines and best practices for contributing.

---

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)

---

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Prioritize patient safety and data privacy in healthcare applications
- Follow HIPAA/GDPR principles when handling medical data

---

## 🚀 Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/your-username/HealthML-Toolkit.git
cd HealthML-Toolkit

# Add upstream remote
git remote add upstream https://github.com/original-owner/HealthML-Toolkit.git
```

### Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

---

## 🔄 Development Workflow

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/your-feature-name`: New features
- `bugfix/issue-number-description`: Bug fixes
- `docs/topic`: Documentation updates

```bash
# Create a feature branch
git checkout -b feature/new-imputation-method

# Make changes, commit often
git add .
git commit -m "Add MICE imputation with custom parameters"

# Keep up to date with upstream
git fetch upstream
git rebase upstream/develop

# Push to your fork
git push origin feature/new-imputation-method
```

---

## 📝 Coding Standards

### Python Style

- **Formatter**: [Black](https://black.readthedocs.io/) (line length 100)
- **Linter**: Flake8
- **Type Hints**: Use type annotations for function signatures

```bash
# Format code
black .

# Lint
flake8 .

# Type check
mypy .
```

### Code Structure

```python
def process_medical_image(
    image_path: str,
    model: keras.Model,
    img_size: int = 224,
) -> dict:
    """
    Process a medical image through the model and return predictions.
    
    Args:
        image_path: Path to the input image
        model: Trained Keras model
        img_size: Target image size for preprocessing
        
    Returns:
        Dictionary with prediction, confidence, and metadata
        
    Example:
        >>> result = process_medical_image("retina.jpg", model)
        >>> print(result["prediction"])
        'No DR'
    """
    # Implementation
    pass
```

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Modules**: lowercase, short, descriptive

---

## 🧪 Testing

### Writing Tests

```python
# tests/test_imputation.py
import pytest
import numpy as np
from data_quality.imputation import knn_impute

def test_knn_imputation_basic():
    """Test KNN imputation with simple missing values"""
    data = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
    result = knn_impute(data, k=1)
    assert not np.isnan(result).any(), "Result should have no NaN values"
    assert result.shape == data.shape, "Shape should be preserved"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific module
pytest tests/test_chatbot.py

# With coverage
pytest --cov=. --cov-report=html

# View coverage
open htmlcov/index.html
```

### Coverage Goals

- **Minimum**: 70% overall coverage
- **Target**: 85%+ for core modules (imputation, model training)
- **Critical**: 95%+ for chatbot medical logic

---

## 📚 Documentation

### Docstrings

Use Google-style docstrings:

```python
def train_retinopathy_model(
    data_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
) -> keras.Model:
    """
    Train a diabetic retinopathy classification model.
    
    This function implements a two-phase training approach:
    1. Train classification head with frozen EfficientNetB0 backbone
    2. Fine-tune top layers with reduced learning rate
    
    Args:
        data_dir: Directory containing training images organized by class
        epochs: Total training epochs (split between phases)
        batch_size: Number of samples per batch
        
    Returns:
        Trained Keras model ready for inference
        
    Raises:
        ValueError: If data_dir doesn't exist or contains no images
        RuntimeError: If training fails due to GPU/memory issues
        
    Example:
        >>> model = train_retinopathy_model(
        ...     Path("data/retinopathy"),
        ...     epochs=15,
        ...     batch_size=32
        ... )
        >>> model.save("outputs/final_model.keras")
    """
    pass
```

### Updating Documentation

- Update `README.md` for user-facing changes
- Add module docs in `docs/` for technical details
- Update inline comments for complex logic

---

## 🔀 Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines (Black, Flake8)
- [ ] All tests pass (`pytest`)
- [ ] New code has tests (coverage ≥70%)
- [ ] Documentation updated (docstrings, README, docs/)
- [ ] Notebook cells are cleared (no output in commits)
- [ ] Commit messages are descriptive

### PR Checklist

```markdown
## Description
Brief summary of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix/feature causing existing functionality to break)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if UI/visualization changes)
[Attach before/after images]

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Review Process

1. Automated checks (GitHub Actions): linting, tests
2. Peer review by maintainer
3. Address feedback
4. Approval and merge

---

## 🎯 Areas for Contribution

### High Priority

- **Data Augmentation**: New augmentation strategies for medical images
- **Imputation Methods**: MICE variants, deep learning-based imputation
- **Model Architectures**: Vision Transformers, EfficientNetV2
- **Chatbot**: Medical entity recognition, multi-turn conversations
- **Testing**: Expand test coverage, add integration tests

### Medium Priority

- **Performance**: Optimize data loading, mixed precision training
- **Documentation**: Video tutorials, architecture diagrams
- **Deployment**: Docker, FastAPI inference server
- **Datasets**: Support for additional medical imaging datasets

### Good First Issues

- Improve error messages
- Add type hints
- Write unit tests
- Fix typos/grammar in docs
- Add examples to notebooks

---

## 📧 Questions?

- **GitHub Discussions**: For general questions
- **GitHub Issues**: For bugs and feature requests
- **Email**: maintainer@example.com

---

**Thank you for contributing to HealthML-Toolkit! 🙏**
