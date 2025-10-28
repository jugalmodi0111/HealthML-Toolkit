# HealthML-Toolkit - AI Development Instructions

## Project Overview
**HealthML-Toolkit** is a comprehensive, modular healthcare machine learning system with three core components:

1. **Healthcare Chatbot** (`chatbot/`) - NLTK-based conversational AI for medical queries and appointments
2. **Data Quality Pipeline** (`data_quality/`) - Advanced imputation techniques for missing healthcare data
3. **Medical Vision System** (`vision/`) - CNN-based diabetic retinopathy classification with Grad-CAM explainability

## Architecture & Module Descriptions

### 1. Chatbot Module (`chatbot/`)
**File**: `healthcare_chatbot.ipynb`

- Pattern-matching conversational AI using NLTK's `Chat` class
- Regex-based intent recognition for symptoms, appointments, health advice
- Conversation persistence (binary `.bin` files with timestamps)
- PDF export for appointment records and chat history
- Emergency detection and escalation logic

**Key Patterns**:
```python
pairs = [
    (r"pattern", ["Response with %1 placeholder"]),
    ...
]
```

### 2. Data Quality Module (`data_quality/`)
**File**: `data_imputation_methods.ipynb`

Comprehensive comparison of missing data handling techniques:
- **Simple**: Mean/Mode imputation
- **Advanced**: KNN (k-nearest neighbors), MICE (multivariate imputation)
- **Domain-specific**: Hot-deck (group-based), Cold-deck (reference-based)
- **Time-series**: Forward/backward fill, interpolation

**Evaluation**: Uses Breast Cancer Wisconsin dataset to compare ML model accuracy across imputation methods

### 3. Vision Pipeline (`vision/`)
**Files**: 
- `diabetic_retinopathy_pipeline.ipynb` - Full interactive workflow
- `cli_train_model.py` - Command-line trainer


**Workflow**:
1. Auto-download Indian Diabetic Retinopathy Dataset from Kaggle
2. Detect dataset layout (flat class folders vs train/val/test split)
3. **Baseline CNN**: Quick 3-layer ConvNet for validation
4. **Transfer Learning**: EfficientNetB0 with:
   - On-device data augmentation (flip, rotation, zoom, contrast)
   - Class weighting to handle imbalance
   - Two-phase training: freeze backbone → fine-tune top layers
5. **Grad-CAM Explainability**: Generate heatmap overlays on sample images
6. Export: models (`.keras`), metrics (`.json`), class names, Grad-CAM images

**Key Technologies**: TensorFlow/Keras, tf.data pipeline, sklearn metrics, OpenCV/PIL

---

## Development Workflows

### Running Each Module

#### Chatbot
```bash
jupyter notebook chatbot/healthcare_chatbot.ipynb
# Or CLI: python main.py chatbot --interactive
```

#### Data Imputation
```bash
jupyter notebook data_quality/data_imputation_methods.ipynb
# Or CLI: python main.py impute --input data.csv --method knn --output imputed.csv
```

#### Vision Pipeline
```bash
# Full interactive pipeline
jupyter notebook vision/diabetic_retinopathy_pipeline.ipynb

# CLI training only
python vision/cli_train_model.py \
  --data-dir data/retinopathy \
  --epochs 15 \
  --img-size 256 \
  --batch-size 32 \
  --auto-download

# Grad-CAM on single image
python main.py gradcam \
  --model vision/models/model_improved.keras \
  --image sample.jpg \
  --output gradcam_output.jpg
```

### Unified CLI Orchestrator (`main.py`)

Central entry point for all workflows:
- `python main.py chatbot` - Launch chatbot
- `python main.py impute` - Run imputation
- `python main.py vision-train` - Train retinopathy model
- `python main.py gradcam` - Generate single Grad-CAM

---

## File Organization & Naming Conventions

### Structure
```
HealthML-Toolkit/
├── chatbot/                  # Conversational AI
│   ├── healthcare_chatbot.ipynb
│   └── saved_conversations/
├── data_quality/             # Imputation & preprocessing
│   ├── data_imputation_methods.ipynb
│   └── examples/
├── vision/                   # Medical imaging
│   ├── diabetic_retinopathy_pipeline.ipynb
│   ├── cli_train_model.py
│   ├── models/               # Saved .keras files, metrics
│   └── gradcam/              # Heatmap outputs
├── docs/                     # Detailed guides
├── configs/                  # YAML/JSON configs
├── tests/                    # pytest suite
├── main.py                   # CLI orchestrator
├── requirements.txt
└── setup.py
```

### Naming Standards
- **Notebooks**: `{module}_{purpose}.ipynb` (e.g., `data_imputation_methods.ipynb`)
- **Scripts**: `cli_{action}.py` or `{function}.py`
- **Outputs**: 
  - Models: `model.keras`, `model_improved.keras`, `best_improved.keras`
  - Metrics: `metrics.json`, `metrics_improved.json`
  - Conversations: `appointments_YYYYMMDD_HHMMSS.bin`
  - Grad-CAM: `{image_stem}_pred-{class}.jpg`

---

## Key Data Flows

### Vision Pipeline Flow
```
Kaggle Dataset 
  → Layout Detection (flat vs split)
  → tf.data.Dataset (prefetch, cache)
  → Baseline CNN (quick validation)
  → EfficientNetB0 Transfer Learning
    - Phase 1: Freeze backbone, train head (higher LR)
    - Phase 2: Unfreeze top N layers, fine-tune (lower LR)
  → Evaluation (sklearn metrics)
  → Grad-CAM (last Conv2D layer)
  → Save: model.keras, metrics.json, gradcam/*.jpg
```

### Imputation Flow
```
Raw Data (CSV with NaN)
  → Choose method (mean/knn/mice/hot-deck/cold-deck/time-series)
  → Apply sklearn Imputer or custom logic
  → Train ML model (RandomForest) on imputed data
  → Compare accuracy vs baseline (no missing values)
  → Visualize (bar chart)
```

### Chatbot Flow
```
User Input
  → Regex Pattern Matching (pairs list)
  → Intent Detection (symptom/appointment/emergency)
  → Response Generation (with %1 placeholders)
  → Conversation Storage (.bin)
  → Optional PDF Export
```

---

## Configuration Management

### YAML Config (`configs/model_params.yaml`)
- `vision.img_size`, `vision.epochs`, `vision.batch_size`
- `data_quality.default_method`, `data_quality.knn_neighbors`
- `chatbot.save_format`, `chatbot.pdf_export`
- `logging.level`, `paths.data_root`

### Kaggle Credentials (`configs/kaggle_config_template.json`)
```json
{
  "username": "your_username",
  "key": "your_api_key"
}
```
Copy to `~/.kaggle/kaggle.json` and `chmod 600`

---

## Best Practices

### Chatbot
1. Always include emergency escalation for serious symptoms (chest pain, severe bleeding)
2. Use regex capture groups `(...)` and `%1`, `%2` for personalized responses
3. Maintain `reflections` dict for pronoun transformations ("I" → "you")
4. Test new patterns with common variations and typos

### Data Quality
1. Visualize missing data patterns before imputation
2. Use domain knowledge to choose method (e.g., time-series for IoT)
3. Compare multiple methods and report accuracy/report
4. Handle edge cases (all NaN columns, no valid values in group)

### Vision
1. **Dataset Layout**: Auto-detect or manual override in config
2. **Class Imbalance**: Always compute class weights for medical imaging
3. **Augmentation**: On-device (Keras layers) for GPU acceleration
4. **Interpretability**: Generate Grad-CAM for model validation and clinical trust
5. **Checkpointing**: Save best model during training (ModelCheckpoint callback)
6. **Two-Phase Training**: Essential for transfer learning stability

### Code Quality
- **Type Hints**: Use for all function signatures
- **Docstrings**: Google-style with Args/Returns/Example
- **Logging**: Use `logger.info()` instead of `print()` in scripts
- **Notebooks**: Clear outputs before committing (`jupyter nbconvert --clear-output`)

---

## Testing Strategy

### Unit Tests (`tests/`)
```python
# tests/test_imputation.py
def test_knn_imputation():
    data = np.array([[1, 2], [np.nan, 4], [5, 6]])
    result = knn_impute(data, k=1)
    assert not np.isnan(result).any()
```

### Integration Tests
- End-to-end vision pipeline (small dataset)
- Chatbot conversation flows
- Imputation → ML training → accuracy check

### Coverage Goals
- **Core modules**: 85%+
- **Critical paths**: 95%+ (chatbot medical logic, Grad-CAM)

---

## Common Tasks

### Adding New Imputation Method
1. Add function in `data_quality/data_imputation_methods.ipynb`
2. Update comparison cell to include new method
3. Add to `main.py impute` command choices
4. Document in README and notebook markdown
5. Write unit test in `tests/test_imputation.py`

### Adding New Vision Backbone
1. Import model from `tf.keras.applications`
2. Update `build_improved_model()` function
3. Add to `configs/model_params.yaml` options
4. Test with small dataset sample
5. Update README performance table

### Extending Chatbot
1. Add new pattern-response pair to `pairs` list
2. Use regex for flexibility: `r"(.*) (keyword1|keyword2)"`
3. Test with variations
4. Update appointment detection logic if needed
5. Document in `.github/copilot-instructions.md`

---

## Troubleshooting

### Vision Pipeline
- **OOM Error**: Reduce `batch_size`, `img_size`, or `limit_per_class`
- **Kaggle 403**: Ensure you've accepted the dataset on Kaggle website
- **TensorFlow Import**: Check Python version (3.10-3.11 recommended)
- **Grad-CAM No Conv2D**: Model must have at least one Conv2D layer

### Data Quality
- **All NaN Column**: Handle explicitly or drop column
- **Hot-deck No Valid Values**: Fall back to mean/mode
- **MICE Not Converging**: Reduce `max_iter` or check data distribution

### Chatbot
- **Pattern Not Matching**: Escape special regex characters (`\.`, `\?`)
- **Reflection Issues**: Update `reflections` dictionary
- **Binary File Corrupt**: Check file permissions and encoding

---

## Integration Points

### External APIs
- **Kaggle API**: Dataset download (`kaggle datasets download`)
- **Future**: Telemedicine APIs, EHR systems, appointment schedulers

### Data Formats
- **Input**: CSV (imputation), Images (JPG/PNG), JSON (configs)
- **Output**: Keras models, JSON metrics, Binary logs, PDF reports, JPG Grad-CAM

### Deployment
- **CLI**: `main.py` for production scripts
- **Notebooks**: Interactive exploration and research
- **Future**: FastAPI server, Docker containers, CI/CD (GitHub Actions)

---

## Performance Optimization

### Vision
- Mixed precision training: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
- Dataset caching: `train_ds.cache().prefetch(AUTOTUNE)`
- Limit samples for testing: `--limit-per-class 200`

### Data Quality
- Vectorized operations (NumPy/Pandas) over loops
- Sample large datasets before full imputation
- Parallel processing for multiple files (future)

---

## Documentation

### Required for Each Module
- **Notebook Markdown**: Purpose, workflow, outputs
- **README Section**: Installation, usage example
- **CONTRIBUTING.md**: How to extend
- **Docstrings**: Function-level documentation

### Updating Docs
- `README.md`: User-facing changes
- `docs/ARCHITECTURE.md`: System design
- `docs/USAGE.md`: Detailed examples
- `.github/copilot-instructions.md`: AI assistant guidance

---

## Release Checklist

Before version bump:
- [ ] All tests pass (`pytest`)
- [ ] Notebooks cleared (`jupyter nbconvert --clear-output`)
- [ ] README updated with new features
- [ ] CHANGELOG.md entry added
- [ ] Version bumped in `setup.py`
- [ ] Git tag created (`git tag v1.0.0`)

---

## Contact & Support

- **GitHub Issues**: Bug reports, feature requests
- **Discussions**: General questions, showcases
- **Email**: maintainer@example.com

---

**Happy coding! 🚀🏥**
