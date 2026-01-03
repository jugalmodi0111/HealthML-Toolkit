# HealthML-Toolkit

**An end-to-end Machine Learning toolkit for healthcare applications**: conversational AI, data quality management, and medical image analysis with explainable AI.

---


https://github.com/user-attachments/assets/cebc0d9e-ec73-4d87-b86b-a85142c9c895



## 📋 Overview

**HealthML-Toolkit** integrates three core healthcare ML workflows:

1. **💬 Healthcare Chatbot** – NLTK-powered conversational agent for health queries and appointment scheduling
2. **🔧 Data Quality & Imputation** – Advanced missing data handling techniques (Mean/Mode, KNN, MICE, Hot-deck, Cold-deck, Time-series)
3. **👁️ Diabetic Retinopathy Vision Pipeline** – CNN-based image classification with Grad-CAM explainability for retinal disease grading

### Key Features

- **Unified Pipeline**: Modular design allows independent or combined use of chatbot, data preprocessing, and vision modules
- **Transfer Learning**: EfficientNetB0 with class balancing, augmentation, and fine-tuning for medical imaging
- **Explainability**: Grad-CAM visualizations for model interpretability
- **Production-Ready**: CLI tool, configurable parameters, comprehensive logging, and export capabilities (JSON, PDF, binary)

---

## 🏗️ Architecture

```
HealthML-Toolkit/
├── chatbot/                      # Conversational AI module
│   ├── healthcare_chatbot.ipynb      # Interactive notebook
│   └── saved_conversations/          # Persisted chat histories (.bin)
├── data_quality/                 # Data imputation & quality tools
│   ├── data_imputation_methods.ipynb  # Comparative imputation techniques
│   └── examples/                     # Sample datasets and results
├── vision/                       # Medical image analysis
│   ├── diabetic_retinopathy_pipeline.ipynb  # End-to-end training+Grad-CAM
│   ├── cli_train_model.py              # Command-line trainer
│   └── models/                         # Saved models and metrics
├── docs/                         # Documentation
│   ├── SETUP.md                      # Installation guide
│   ├── USAGE.md                      # Usage examples
│   └── ARCHITECTURE.md               # System design
├── configs/                      # Configuration templates
│   ├── kaggle_config_template.json
│   └── model_params.yaml
├── tests/                        # Unit and integration tests
├── main.py                       # Unified CLI orchestrator
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installer
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (3.11 recommended for TensorFlow compatibility)
- macOS/Linux/Windows
- (Optional) Kaggle account for retinopathy dataset

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/HealthML-Toolkit.git
cd HealthML-Toolkit

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) Apple Silicon GPU acceleration
# pip install tensorflow-macos tensorflow-metal
```

### Usage

#### 1️⃣ Healthcare Chatbot

```bash
# Interactive notebook
jupyter notebook chatbot/healthcare_chatbot.ipynb

# Or use the CLI
python main.py chatbot --interactive
```

#### 2️⃣ Data Imputation

```bash
# Run comparative analysis
jupyter notebook data_quality/data_imputation_methods.ipynb

# Or via CLI with your dataset
python main.py impute --input data.csv --method knn --output imputed_data.csv
```

#### 3️⃣ Diabetic Retinopathy Pipeline

```bash
# Full pipeline (download, train, evaluate, Grad-CAM)
jupyter notebook vision/diabetic_retinopathy_pipeline.ipynb

# Or CLI for training only
python vision/cli_train_model.py \
  --data-dir data/retinopathy \
  --epochs 15 \
  --img-size 256 \
  --batch-size 32 \
  --out-dir outputs_retino
```

**Kaggle Setup** (for retinopathy dataset):

1. Get API token: https://www.kaggle.com/settings → "Create New API Token"
2. Place `kaggle.json` at `~/.kaggle/kaggle.json` (chmod 600 on Unix)
3. Accept dataset: https://www.kaggle.com/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset

---

## 📊 Modules

### 💬 Chatbot Module

- **File**: `chatbot/healthcare_chatbot.ipynb`
- **Tech Stack**: NLTK Chat, regex pattern matching
- **Features**:
  - Symptom assessment (fever, headache, cough, chest pain)
  - Appointment scheduling
  - Health advice (diet, exercise)
  - Emergency detection and escalation
- **Outputs**: Conversation logs (.bin), appointment records, PDF exports

### 🔧 Data Quality Module

- **File**: `data_quality/data_imputation_methods.ipynb`
- **Methods**:
  - **Simple**: Mean/Mode imputation
  - **Advanced**: KNN, MICE (Iterative Imputer)
  - **Domain-specific**: Hot-deck, Cold-deck
  - **Time-series**: Forward-fill, Backward-fill, Interpolation
- **Evaluation**: Accuracy comparison on Breast Cancer Wisconsin dataset
- **Visualizations**: Matplotlib/Seaborn comparative bar charts

### 👁️ Vision Module

- **File**: `vision/diabetic_retinopathy_pipeline.ipynb`
- **Workflow**:
  1. Auto-download Indian Diabetic Retinopathy Dataset (Kaggle)
  2. Dataset layout detection (flat vs split)
  3. **Baseline CNN**: 3-layer ConvNet (quick validation)
  4. **Improved Model**: EfficientNetB0 transfer learning
     - Data augmentation (flip, rotation, zoom, contrast)
     - Class weighting for imbalance
     - Two-phase training (freeze → fine-tune)
  5. **Explainability**: Grad-CAM heatmaps on sample images
- **Outputs**: `model.keras`, `metrics.json`, `gradcam/` folder
- **CLI Tool**: `vision/cli_train_model.py` for headless execution

---

## 🔬 Performance

### Data Imputation (Breast Cancer Dataset)

| Method     | Accuracy | Notes                          |
|------------|----------|--------------------------------|
| Baseline   | 0.9766   | No missing values              |
| Mean       | 0.9708   | Simple, fast                   |
| Hot-deck   | 0.9649   | Group-based sampling           |
| Cold-deck  | 0.9766   | External reference (matched baseline) |

### Diabetic Retinopathy Classification

| Model              | Val Accuracy | Notes                                |
|--------------------|--------------|--------------------------------------|
| Baseline CNN       | ~0.55–0.65   | 3-layer custom ConvNet (5 epochs)    |
| EfficientNetB0     | ~0.75–0.85   | Transfer learning + augmentation     |
| Improved (fine-tune) | ~0.82–0.90 | Class weights + 2-phase training    |

*Results vary based on LIMIT_PER_CLASS and dataset split.*

---

## 📚 Documentation

- **[SETUP.md](docs/SETUP.md)**: Detailed installation (venv, conda, Docker)
- **[USAGE.md](docs/USAGE.md)**: Step-by-step examples for each module
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: System design, data flow, and API reference
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Guidelines for contributors
- **[Copilot Instructions](.github/copilot-instructions.md)**: AI-assisted development guide

---

## 🛠️ Configuration

### Model Parameters (`configs/model_params.yaml`)

```yaml
vision:
  img_size: 256
  batch_size: 32
  epochs: 15
  val_split: 0.2
  limit_per_class: null  # null = use full dataset
  
chatbot:
  save_format: binary  # or 'json'
  pdf_export: true

imputation:
  default_method: knn
  knn_neighbors: 5
```

### Kaggle API (`configs/kaggle_config_template.json`)

```json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Specific module
pytest tests/test_chatbot.py
pytest tests/test_imputation.py
pytest tests/test_vision.py

# With coverage
pytest --cov=. --cov-report=html
```

---

## 📦 Outputs

### Chatbot
- `chatbot/saved_conversations/appointments_YYYYMMDD_HHMMSS.bin`
- `chatbot/saved_conversations/conversation_log.pdf`

### Data Quality
- `data_quality/examples/imputed_data.csv`
- `data_quality/examples/accuracy_comparison.png`

### Vision
- `vision/models/model.keras` (baseline)
- `vision/models/model_improved.keras` (EfficientNet)
- `vision/models/best_improved.keras` (checkpoint)
- `vision/models/metrics.json`, `metrics_improved.json`
- `vision/models/class_names.json`
- `vision/gradcam/*.jpg` (heatmap overlays)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Code style guidelines (Black, Flake8)
- Branch naming conventions
- Pull request process
- Testing requirements

---

## 📄 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Datasets**:
  - [Indian Diabetic Retinopathy Image Dataset (Kaggle)](https://www.kaggle.com/datasets/aaryapatel98/indian-diabetic-retinopathy-image-dataset)
  - [Breast Cancer Wisconsin (sklearn)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Frameworks**: TensorFlow/Keras, scikit-learn, NLTK, Pandas, Matplotlib
- **Research**: Grad-CAM paper ([Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391))

---

## 📧 Contact

- **Maintainer**: Your Name
- **Email**: your.email@example.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/HealthML-Toolkit/issues)

---

**⭐ If you find this toolkit useful, please star the repo and share it with the community!**
