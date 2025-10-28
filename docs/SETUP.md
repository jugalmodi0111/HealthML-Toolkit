# HealthML-Toolkit Setup Guide

Complete installation and configuration guide for HealthML-Toolkit.

---

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
  - [Quick Install (pip)](#quick-install-pip)
  - [Development Install](#development-install)
  - [Docker Install](#docker-install-coming-soon)
- [Platform-Specific Setup](#platform-specific-setup)
  - [macOS (Apple Silicon)](#macos-apple-silicon)
  - [Linux](#linux)
  - [Windows](#windows)
- [Kaggle API Configuration](#kaggle-api-configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum
- **Python**: 3.10 - 3.11 (TensorFlow compatibility)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for datasets and models
- **OS**: macOS 11+, Ubuntu 20.04+, Windows 10+

### Recommended (for Vision Module)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (Linux/Windows) or Apple Silicon M1/M2 (macOS)
- **RAM**: 16GB+
- **Storage**: SSD with 10GB+ free space

---

## Installation Methods

### Quick Install (pip)

**1. Create Virtual Environment**

```bash
# Using venv (recommended)
python3.11 -m venv healthml-env
source healthml-env/bin/activate  # macOS/Linux
# healthml-env\Scripts\activate   # Windows

# Using conda
conda create -n healthml python=3.11
conda activate healthml
```

**2. Install Package**

```bash
# Clone repository
git clone https://github.com/yourusername/HealthML-Toolkit.git
cd HealthML-Toolkit

# Install with all dependencies
pip install -e .

# Or install with extras
pip install -e ".[dev,pdf,apple-silicon]"
```

**3. Download NLTK Data**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

---

### Development Install

For contributors who need testing tools, code quality checks, and PDF export.

```bash
# Clone and navigate
git clone https://github.com/yourusername/HealthML-Toolkit.git
cd HealthML-Toolkit

# Create environment
python3.11 -m venv healthml-dev
source healthml-dev/bin/activate

# Install with development extras
pip install -e ".[dev,pdf]"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

**Development Tools Installed:**
- `pytest` - Unit testing
- `pytest-cov` - Coverage reporting
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking
- `reportlab` - PDF export

---

### Docker Install (Coming Soon)

```bash
# Pull image
docker pull healthml/toolkit:latest

# Run chatbot
docker run -it healthml/toolkit python main.py chatbot

# Run with mounted data
docker run -v $(pwd)/data:/app/data healthml/toolkit \
  python main.py vision-train --data-dir /app/data/retinopathy
```

---

## Platform-Specific Setup

### macOS (Apple Silicon)

**TensorFlow Metal Acceleration**

For M1/M2 Macs, install Apple-optimized TensorFlow:

```bash
# Ensure you're on Python 3.10 or 3.11
python --version

# Install with Apple Silicon extras
pip install -e ".[apple-silicon]"

# This installs:
# - tensorflow-macos
# - tensorflow-metal (GPU acceleration)
```

**Verify GPU Support**

```python
import tensorflow as tf
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
```

**Common Issues**
- **ImportError: Metal plugin**: Reinstall `tensorflow-metal`
- **Slow Training**: Check Activity Monitor → GPU History for usage

---

### Linux

**NVIDIA GPU Setup (CUDA)**

For NVIDIA GPUs, install CUDA-enabled TensorFlow:

```bash
# Check NVIDIA driver
nvidia-smi

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Ubuntu Dependencies**

```bash
# System packages for OpenCV
sudo apt-get update
sudo apt-get install -y python3-opencv libsm6 libxext6 libxrender-dev

# Install HealthML-Toolkit
pip install -e .
```

---

### Windows

**Prerequisites**

1. **Microsoft Visual C++ 14.0+**: Required for some packages
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Install "Desktop development with C++"

2. **Python 3.11**: Install from python.org or Microsoft Store

**Installation**

```powershell
# Create virtual environment
python -m venv healthml-env
healthml-env\Scripts\activate

# Install package
pip install -e .

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

**GPU Support (NVIDIA)**

```powershell
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]
```

---

## Kaggle API Configuration

Required for automatic retinopathy dataset download.

### Step 1: Get API Credentials

1. Login to [Kaggle](https://www.kaggle.com)
2. Navigate to **Account** → **API** → **Create New API Token**
3. Download `kaggle.json`

### Step 2: Install Credentials

**macOS/Linux**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows**
```powershell
mkdir $env:USERPROFILE\.kaggle
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\
```

### Step 3: Accept Dataset Terms

Visit the [Indian Diabetic Retinopathy Dataset](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data) and click **"Download"** (you must accept terms).

### Step 4: Test API

```bash
kaggle datasets list -s diabetic
```

---

## Verification

### Quick Test

```bash
# Test CLI
python main.py --help

# Test chatbot (exit with 'quit')
python main.py chatbot

# Test imputation
echo -e "col1,col2\n1,2\n,4\n5," > test.csv
python main.py impute --input test.csv --method mean --output out.csv
```

### Full System Check

```python
# test_install.py
import sys
import tensorflow as tf
import sklearn
import nltk
import pandas as pd
import numpy as np

print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")

# Test NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    print("✅ NLTK data installed")
except LookupError:
    print("❌ Run: nltk.download('punkt')")

print("\n✅ All components installed successfully!")
```

Run: `python test_install.py`

---

## Troubleshooting

### Common Issues

#### 1. **TensorFlow Import Error**

**Symptom**: `ImportError: DLL load failed` (Windows) or segmentation fault

**Solution**:
```bash
# Reinstall TensorFlow
pip uninstall tensorflow tensorflow-macos tensorflow-metal
pip install tensorflow  # or tensorflow-macos for Apple Silicon
```

#### 2. **NLTK Data Missing**

**Symptom**: `LookupError: Resource punkt not found`

**Solution**:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

#### 3. **Kaggle API 403 Forbidden**

**Symptom**: `403 - Forbidden` when downloading dataset

**Solution**:
- Visit dataset page and accept terms
- Verify `~/.kaggle/kaggle.json` exists with correct permissions (`chmod 600`)
- Ensure credentials are valid (check username/key)

#### 4. **Out of Memory (OOM) During Training**

**Symptom**: Training crashes with memory error

**Solution**:
```bash
# Reduce batch size and image size
python vision/cli_train_model.py \
  --batch-size 16 \
  --img-size 128 \
  --limit-per-class 100
```

#### 5. **OpenCV Import Error**

**Symptom**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution (Linux)**:
```bash
sudo apt-get install -y libsm6 libxext6 libxrender-dev libgomp1
pip install opencv-python-headless
```

#### 6. **Jupyter Notebook Not Found**

**Symptom**: `jupyter: command not found`

**Solution**:
```bash
pip install jupyter notebook
# Or install dev extras
pip install -e ".[dev]"
```

---

### Platform-Specific Issues

#### macOS: Rosetta 2 vs Native ARM

If you accidentally installed x86_64 Python on Apple Silicon:

```bash
# Check architecture
python -c "import platform; print(platform.machine())"
# Should print "arm64", not "x86_64"

# If x86_64, reinstall Python from python.org (macOS installer)
```

#### Windows: Long Path Issues

Enable long paths in Windows 10/11:

```powershell
# Run PowerShell as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### Linux: CUDA Version Mismatch

```bash
# Check CUDA version
nvcc --version

# Install compatible TensorFlow
# CUDA 11.8 → tensorflow>=2.13
# CUDA 12.x → tensorflow>=2.15
pip install tensorflow==2.15.0
```

---

## Next Steps

After successful installation:

1. **Read Documentation**: See [USAGE.md](USAGE.md) for detailed examples
2. **Run Notebooks**: Explore interactive workflows in `chatbot/`, `data_quality/`, `vision/`
3. **Configure**: Edit `configs/model_params.yaml` for your setup
4. **Contribute**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines

---

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/HealthML-Toolkit/issues)
- **Discussions**: [Ask questions](https://github.com/yourusername/HealthML-Toolkit/discussions)
- **Email**: maintainer@example.com

---

**Installation successful? 🎉 Head to [USAGE.md](USAGE.md) to get started!**
