# HealthML-Toolkit Architecture

Technical deep-dive into the system design, data flows, and component interactions.

---

## Table of Contents
- [System Overview](#system-overview)
- [Module Architecture](#module-architecture)
- [Data Flow Diagrams](#data-flow-diagrams)
- [Component Interactions](#component-interactions)
- [Extension Points](#extension-points)
- [Design Patterns](#design-patterns)
- [Performance Considerations](#performance-considerations)

---

## System Overview

HealthML-Toolkit is a **modular, loosely-coupled** healthcare ML system with three independent pipelines that can operate standalone or in combination.

### Design Principles

1. **Modularity**: Each module (Chatbot, Data Quality, Vision) is self-contained
2. **Configurability**: YAML/JSON configs separate logic from parameters
3. **Extensibility**: Plugin architecture for new imputation methods, vision backbones
4. **Reproducibility**: Seeded random states, versioned dependencies
5. **Explainability**: Grad-CAM for vision, pattern transparency for chatbot

### Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                     │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐ │
│  │   Chatbot   │  │ Data Quality │  │ Vision Pipeline│ │
│  │   (NLTK)    │  │  (sklearn)   │  │  (TensorFlow)  │ │
│  └─────────────┘  └─────────────┘  └────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                   Orchestration Layer                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │           main.py (CLI Orchestrator)             │   │
│  │  Subcommands: chatbot | impute | vision | gradcam│  │
│  └─────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                  Configuration Layer                    │
│  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │ model_params.yaml│  │ kaggle_config_template.json│  │
│  └──────────────────┘  └──────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                     Data Layer                          │
│  data/ → retinopathy/, patient_records/, conversations/ │
└─────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### 1. Chatbot Module

**Purpose**: Conversational AI for health queries, symptom analysis, appointment scheduling

**Architecture**:
```
User Input
    ↓
┌─────────────────────────────────────┐
│      NLTK Chat Engine               │
│  ┌─────────────────────────────┐   │
│  │  Regex Pattern Matching     │   │
│  │  - pairs: [(pattern, resp)] │   │
│  │  - reflections: pronoun map │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    Intent Classification            │
│  - Symptom query                    │
│  - Appointment request              │
│  - Emergency detection              │
│  - General health advice            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    Response Generation              │
│  - Template filling (%1, %2)        │
│  - Pronoun transformation           │
│  - Context-aware responses          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    Persistence Layer                │
│  - Binary .bin files                │
│  - Timestamp-based naming           │
│  - PDF export (ReportLab)           │
└─────────────────────────────────────┘
```

**Key Components**:
- `pairs`: List of (regex_pattern, [responses]) tuples
- `reflections`: Dictionary mapping pronouns (I → you, my → your)
- `Chat`: NLTK class for conversation management
- `read_appointments()`: Binary file deserializer
- `export_to_pdf()`: PDF report generator

**Extension Points**:
- Add new patterns to `pairs` list
- Integrate external APIs (appointment systems, EHR)
- Replace NLTK with Rasa, Dialogflow, or LLMs

---

### 2. Data Quality Module

**Purpose**: Comparative analysis of missing data imputation techniques

**Architecture**:
```
Raw Data (CSV with NaN)
    ↓
┌─────────────────────────────────────┐
│   Data Validation & Profiling       │
│  - Missing value detection          │
│  - Pattern analysis (MCAR/MAR/MNAR) │
│  - Feature type inference           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│      Imputation Strategy Selection  │
│  ┌─────────────┬─────────────┐     │
│  │ Simple      │ Advanced    │     │
│  │ - Mean      │ - KNN       │     │
│  │ - Mode      │ - MICE      │     │
│  │ - Median    │ - Hot-deck  │     │
│  │             │ - Cold-deck │     │
│  │             │ - Time-series│    │
│  └─────────────┴─────────────┘     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    sklearn Imputation Engine        │
│  - SimpleImputer                    │
│  - KNNImputer                       │
│  - IterativeImputer (MICE)          │
│  - Custom hot-deck/cold-deck        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    Quality Evaluation               │
│  - Train ML model (RandomForest)    │
│  - Compare accuracy across methods  │
│  - Visualize performance            │
└─────────────────────────────────────┘
    ↓
Imputed Data (CSV)
```

**Key Components**:
- `SimpleImputer`: sklearn mean/mode imputation
- `KNNImputer`: k-nearest neighbors averaging
- `IterativeImputer`: MICE (chained equations)
- Custom functions: `hot_deck_impute()`, `cold_deck_impute()`, `time_series_impute()`
- Evaluation: Breast Cancer Wisconsin dataset for benchmarking

**Extension Points**:
- Add deep learning imputation (autoencoders, GANs)
- Integrate domain-specific rules (medical constraints)
- Support categorical missing value handling

---

### 3. Vision Module

**Purpose**: Diabetic retinopathy classification with Grad-CAM explainability

**Architecture**:
```
Kaggle Dataset
    ↓
┌─────────────────────────────────────┐
│   Dataset Preprocessing             │
│  - Layout detection (flat/split)    │
│  - Auto-download via Kaggle API     │
│  - Class balance analysis           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     tf.data Pipeline                │
│  - image_dataset_from_directory()   │
│  - Augmentation (flip, rotate, zoom)│
│  - Normalization (rescale 1/255)    │
│  - Caching & Prefetching            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    Model Architecture Selection     │
│  ┌─────────────┬─────────────────┐ │
│  │ Baseline    │ Transfer Learning│ │
│  │ - 3-layer   │ - EfficientNetB0 │ │
│  │   CNN       │ - Frozen backbone│ │
│  │ - ReLU      │ - Custom head    │ │
│  │ - MaxPool   │ - Fine-tuning    │ │
│  └─────────────┴─────────────────┘ │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│      Training Strategy              │
│  Phase 1: Freeze backbone           │
│    - Train dense layers only        │
│    - Higher learning rate (1e-3)    │
│    - 10 epochs                      │
│                                     │
│  Phase 2: Fine-tune top layers      │
│    - Unfreeze last N layers         │
│    - Lower learning rate (1e-5)     │
│    - 5-10 epochs                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│     Evaluation & Metrics            │
│  - sklearn classification_report    │
│  - Confusion matrix                 │
│  - Precision, Recall, F1            │
│  - Save metrics.json                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│    Grad-CAM Explainability          │
│  - Extract last Conv2D layer output │
│  - Compute gradients w.r.t. class   │
│  - Weighted activation heatmap      │
│  - Overlay on original image        │
└─────────────────────────────────────┘
    ↓
Outputs: model.keras, metrics.json, gradcam/*.jpg
```

**Key Components**:

1. **Data Pipeline**:
   ```python
   train_ds = tf.keras.preprocessing.image_dataset_from_directory(
       data_dir,
       image_size=(img_size, img_size),
       batch_size=batch_size,
       subset='training',
       validation_split=0.2,
       seed=42
   )
   train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
   ```

2. **Baseline CNN**:
   ```python
   model = tf.keras.Sequential([
       tf.keras.layers.Rescaling(1./255),
       tf.keras.layers.Conv2D(32, 3, activation='relu'),
       tf.keras.layers.MaxPooling2D(),
       tf.keras.layers.Conv2D(64, 3, activation='relu'),
       tf.keras.layers.MaxPooling2D(),
       tf.keras.layers.Conv2D(128, 3, activation='relu'),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(num_classes, activation='softmax')
   ])
   ```

3. **Transfer Learning (EfficientNetB0)**:
   ```python
   base_model = tf.keras.applications.EfficientNetB0(
       include_top=False,
       weights='imagenet',
       input_shape=(img_size, img_size, 3)
   )
   base_model.trainable = False  # Phase 1
   
   model = tf.keras.Sequential([
       augmentation_layers,
       base_model,
       tf.keras.layers.GlobalAveragePooling2D(),
       tf.keras.layers.Dropout(0.3),
       tf.keras.layers.Dense(num_classes, activation='softmax')
   ])
   
   # Phase 2: Fine-tune
   base_model.trainable = True
   for layer in base_model.layers[:-20]:
       layer.trainable = False
   ```

4. **Grad-CAM**:
   ```python
   grad_model = tf.keras.models.Model(
       inputs=[model.inputs],
       outputs=[last_conv_layer.output, model.output]
   )
   
   with tf.GradientTape() as tape:
       conv_outputs, predictions = grad_model(img_array)
       class_channel = predictions[:, class_index]
   
   grads = tape.gradient(class_channel, conv_outputs)
   pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
   heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
   ```

**Extension Points**:
- Add other backbones (ResNet, VGG, Inception, ViT)
- Multi-task learning (classification + segmentation)
- Ensemble methods (combine multiple models)
- Active learning for data labeling

---

## Data Flow Diagrams

### Cross-Module Integration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Patient Data Workflow                      │
└─────────────────────────────────────────────────────────────────┘

Patient Records (CSV)
    ↓
┌──────────────────────────┐
│  Data Quality Module     │
│  - Impute missing values │
│  - Validate constraints  │
└──────────────────────────┘
    ↓
Clean Patient Data
    ↓
┌──────────────────────────┐
│   Vision Module          │
│  - Retina image analysis │
│  - Risk score prediction │
│  - Grad-CAM visualization│
└──────────────────────────┘
    ↓
Risk Scores + Explanations
    ↓
┌──────────────────────────┐
│   Chatbot Module         │
│  - Personalized advice   │
│  - Appointment scheduling│
│  - Report generation     │
└──────────────────────────┘
    ↓
Patient Report (PDF)
```

### Vision Pipeline Detailed Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              Diabetic Retinopathy Classification                │
└─────────────────────────────────────────────────────────────────┘

Kaggle API
    ↓
┌──────────────────────┐
│ Dataset Download     │
│ - authenticate       │
│ - download ZIP       │
│ - extract to data/   │
└──────────────────────┘
    ↓
Raw Images (JPG)
    ↓
┌──────────────────────┐
│ Layout Detection     │
│ - check folder struct│
│ - flat vs split      │
└──────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│      tf.data Pipeline Construction       │
│  ┌────────────────────────────────────┐ │
│  │ image_dataset_from_directory()     │ │
│  │   ↓                                │ │
│  │ Data Augmentation Layers           │ │
│  │ - RandomFlip(horizontal)           │ │
│  │ - RandomRotation(0.2)              │ │
│  │ - RandomZoom(0.15)                 │ │
│  │ - RandomContrast(0.2)              │ │
│  │   ↓                                │ │
│  │ Normalization (Rescaling 1/255)    │ │
│  │   ↓                                │ │
│  │ Caching & Prefetching              │ │
│  │ - .cache()                         │ │
│  │ - .prefetch(AUTOTUNE)              │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
    ↓
Train/Val Datasets
    ↓
┌──────────────────────────────────────────┐
│        Model Training                    │
│  ┌────────────────────────────────────┐ │
│  │ Phase 1: Transfer Learning         │ │
│  │ - EfficientNetB0 (frozen)          │ │
│  │ - Adam(lr=1e-3)                    │ │
│  │ - Class weights                    │ │
│  │ - 10 epochs                        │ │
│  │   ↓                                │ │
│  │ Phase 2: Fine-Tuning               │ │
│  │ - Unfreeze top 20 layers           │ │
│  │ - Adam(lr=1e-5)                    │ │
│  │ - 5-10 epochs                      │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
    ↓
Trained Model (.keras)
    ↓
┌──────────────────────────────────────────┐
│        Evaluation                        │
│  - Predict on test set                   │
│  - sklearn.metrics.classification_report │
│  - Confusion matrix                      │
│  - Save metrics.json                     │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│        Grad-CAM Generation               │
│  - Select sample images                  │
│  - Compute activation heatmaps           │
│  - Overlay on original images            │
│  - Save to gradcam/*.jpg                 │
└──────────────────────────────────────────┘
    ↓
Outputs: model.keras, metrics.json, gradcam/
```

---

## Component Interactions

### CLI Orchestrator (`main.py`)

**Responsibility**: Unified interface for all modules

```python
# Subcommand architecture
main.py
├── chatbot
│   └── Imports: chatbot/healthcare_chatbot.ipynb functions
├── impute
│   └── Imports: sklearn imputers, pandas
├── vision-train
│   └── Calls: vision/cli_train_model.py main()
└── gradcam
    └── Imports: TensorFlow, cv2, PIL
```

**Inter-module Communication**:
- No direct coupling between modules
- Communication via **file I/O** (CSV, JSON, binary)
- Future: REST API for microservices deployment

---

## Extension Points

### 1. Adding New Imputation Methods

**Example: Autoencoder-based Imputation**

```python
# data_quality/autoencoder_impute.py
import tensorflow as tf

class AutoencoderImputer:
    def __init__(self, encoding_dim=32):
        self.encoding_dim = encoding_dim
        self.model = None
    
    def fit(self, X_complete):
        input_dim = X_complete.shape[1]
        
        # Build autoencoder
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(self.encoding_dim, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X_complete, X_complete, epochs=50, verbose=0)
    
    def transform(self, X_missing):
        # Use model to predict missing values
        return self.model.predict(X_missing)
```

**Integration**:
1. Add to `data_quality/data_imputation_methods.ipynb`
2. Update `main.py impute` subcommand
3. Add to README performance table

### 2. Adding New Vision Backbone

**Example: Vision Transformer (ViT)**

```python
# vision/models/vit_model.py
import tensorflow as tf
from transformers import TFViTModel

def build_vit_model(num_classes, img_size=224):
    base_model = TFViTModel.from_pretrained('google/vit-base-patch16-224')
    
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs).last_hidden_state[:, 0, :]  # CLS token
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

**Integration**:
1. Add to `vision/cli_train_model.py`
2. Update `configs/model_params.yaml` with ViT parameters
3. Test on small dataset subset

### 3. Chatbot LLM Integration

**Example: OpenAI GPT Integration**

```python
# chatbot/llm_chatbot.py
import openai

class LLMChatbot:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.conversation_history = []
    
    def chat(self, user_input):
        self.conversation_history.append({"role": "user", "content": user_input})
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful healthcare assistant."},
                *self.conversation_history
            ]
        )
        
        assistant_message = response['choices'][0]['message']['content']
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
```

---

## Design Patterns

### 1. **Strategy Pattern** (Imputation Methods)

```python
class ImputationStrategy:
    def impute(self, data):
        raise NotImplementedError

class MeanImputation(ImputationStrategy):
    def impute(self, data):
        return SimpleImputer(strategy='mean').fit_transform(data)

class KNNImputation(ImputationStrategy):
    def __init__(self, k=5):
        self.k = k
    
    def impute(self, data):
        return KNNImputer(n_neighbors=self.k).fit_transform(data)

# Usage
strategy = KNNImputation(k=5)
imputed_data = strategy.impute(raw_data)
```

### 2. **Factory Pattern** (Model Creation)

```python
class ModelFactory:
    @staticmethod
    def create_model(model_type, num_classes, img_size):
        if model_type == 'baseline':
            return build_baseline_cnn(num_classes, img_size)
        elif model_type == 'efficientnet':
            return build_efficientnet_model(num_classes, img_size)
        elif model_type == 'vit':
            return build_vit_model(num_classes, img_size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Usage
model = ModelFactory.create_model('efficientnet', num_classes=5, img_size=224)
```

### 3. **Observer Pattern** (Training Callbacks)

```python
class TrainingObserver:
    def on_epoch_end(self, epoch, logs):
        raise NotImplementedError

class MetricsLogger(TrainingObserver):
    def on_epoch_end(self, epoch, logs):
        print(f"Epoch {epoch}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}")

class CheckpointSaver(TrainingObserver):
    def on_epoch_end(self, epoch, logs):
        if logs['val_accuracy'] > self.best_acc:
            model.save(f'checkpoint_epoch{epoch}.keras')
```

---

## Performance Considerations

### 1. **Memory Optimization**

**Problem**: Large datasets cause OOM

**Solutions**:
```python
# Limit dataset size during development
train_ds = train_ds.take(100)

# Use generator for large files
def data_generator():
    for file in file_list:
        yield load_and_preprocess(file)

train_ds = tf.data.Dataset.from_generator(data_generator, ...)

# Clear session between experiments
tf.keras.backend.clear_session()
```

### 2. **Training Speed**

**Techniques**:
- **Mixed Precision**: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
- **Caching**: `train_ds.cache().prefetch(tf.data.AUTOTUNE)`
- **Multi-GPU**: `tf.distribute.MirroredStrategy()`
- **Reduce Image Size**: 128x128 instead of 256x256

### 3. **Inference Latency**

**Optimizations**:
```python
# Convert to TFLite for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Quantization
converter.target_spec.supported_types = [tf.float16]
```

---

## Security Considerations

1. **Kaggle Credentials**: Store in `~/.kaggle/kaggle.json` with `chmod 600`
2. **Patient Data**: Never commit PHI/PII to version control
3. **Model Artifacts**: Use `.gitignore` for large model files
4. **API Keys**: Use environment variables, never hardcode

---

## Future Architecture Enhancements

### Microservices Deployment

```
┌─────────────────────────────────────────────────────────┐
│                   API Gateway (FastAPI)                 │
└─────────────────────────────────────────────────────────┘
         │                │                 │
    ┌────▼────┐     ┌────▼────┐     ┌─────▼─────┐
    │Chatbot  │     │ Impute  │     │  Vision   │
    │Service  │     │ Service │     │  Service  │
    │(Docker) │     │(Docker) │     │ (Docker)  │
    └─────────┘     └─────────┘     └───────────┘
         │                │                 │
    ┌────▼────────────────▼─────────────────▼────┐
    │          Shared Data Store (S3/GCS)         │
    └─────────────────────────────────────────────┘
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: HealthML-Toolkit CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/ --cov=.
      - name: Lint
        run: flake8 . --max-line-length=88
```

---

## Conclusion

HealthML-Toolkit's architecture prioritizes:
- **Modularity**: Independent components for flexibility
- **Extensibility**: Plugin-based design for new methods
- **Performance**: Optimized data pipelines and caching
- **Explainability**: Grad-CAM for model transparency

For implementation details, see [USAGE.md](USAGE.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).

---

**Questions? Open a GitHub issue or discussion!**
