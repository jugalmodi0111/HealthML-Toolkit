# HealthML-Toolkit Usage Guide

Comprehensive usage examples for all three modules: Chatbot, Data Quality, and Vision.

---

## Table of Contents
- [Chatbot Module](#chatbot-module)
- [Data Quality Module](#data-quality-module)
- [Vision Module](#vision-module)
- [Advanced Workflows](#advanced-workflows)
- [Configuration](#configuration)

---

## Chatbot Module

The healthcare chatbot provides conversational AI for symptom analysis, appointment scheduling, and health advice.

### Interactive Mode (Notebook)

```bash
# Launch Jupyter notebook
jupyter notebook chatbot/healthcare_chatbot.ipynb

# Or using the CLI
python main.py chatbot
```

**Example Conversation:**

```
User: Hi, I need help with a headache
Bot: Hello! I can help you with health information, symptoms, and appointments.

User: I have a severe headache and fever
Bot: I understand you're experiencing headache and fever. Please consult a doctor...

User: I want to book an appointment
Bot: I can help you schedule an appointment. What day works best for you?

User: Tomorrow at 3pm
Bot: Great! I've noted your appointment for tomorrow at 3pm...

User: quit
Bot: Goodbye! Take care of your health.
```

### Saved Conversations

Conversations are automatically saved to binary files:

```python
from chatbot.healthcare_chatbot import read_appointments

# Read saved appointment data
appointments = read_appointments('chatbot/saved_conversations/appointments_20240115_143022.bin')

for appt in appointments:
    print(f"Time: {appt['timestamp']}")
    print(f"User: {appt['user_input']}")
    print(f"Response: {appt['bot_response']}\n")
```

### Export to PDF

```python
from chatbot.healthcare_chatbot import export_to_pdf

# Export conversation history
export_to_pdf(
    appointments,
    filename='chatbot/saved_conversations/appointments_summary.pdf'
)
```

### Customizing Patterns

Edit the `pairs` list in `healthcare_chatbot.ipynb`:

```python
pairs = [
    # Add custom pattern
    [
        r"(.*) diabetes (.*)",
        [
            "Diabetes is a chronic condition affecting blood sugar levels. "
            "Please consult an endocrinologist for proper diagnosis and management."
        ]
    ],
    # ... existing patterns
]
```

---

## Data Quality Module

Comprehensive imputation methods for handling missing healthcare data.

### Notebook Workflow

```bash
jupyter notebook data_quality/data_imputation_methods.ipynb
```

The notebook demonstrates:
1. **Simple Imputation** (Mean, Mode)
2. **KNN Imputation** (k-nearest neighbors)
3. **MICE** (Multivariate Imputation by Chained Equations)
4. **Hot-deck** (Group-based)
5. **Cold-deck** (Reference dataset)
6. **Time-series** (Forward/backward fill)

### CLI Imputation

#### Mean Imputation

```bash
python main.py impute \
  --input data/patient_records.csv \
  --method mean \
  --output data/patient_records_imputed.csv
```

#### KNN Imputation

```bash
python main.py impute \
  --input data/patient_records.csv \
  --method knn \
  --output data/patient_records_knn.csv
```

### Programmatic Usage

```python
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load data with missing values
df = pd.read_csv('data/patient_data.csv')

# Method 1: Mean Imputation
mean_imputer = SimpleImputer(strategy='mean')
df_mean = pd.DataFrame(
    mean_imputer.fit_transform(df),
    columns=df.columns
)

# Method 2: KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(
    knn_imputer.fit_transform(df),
    columns=df.columns
)

# Method 3: MICE (Iterative)
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
df_mice = pd.DataFrame(
    mice_imputer.fit_transform(df),
    columns=df.columns
)

# Compare results
print("Original missing values:", df.isnull().sum().sum())
print("Mean imputed missing:", df_mean.isnull().sum().sum())
print("KNN imputed missing:", df_knn.isnull().sum().sum())
print("MICE imputed missing:", df_mice.isnull().sum().sum())
```

### Evaluating Imputation Quality

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load complete dataset
df_complete = pd.read_csv('data/complete_data.csv')
X = df_complete.drop('target', axis=1)
y = df_complete['target']

# Train model on imputed data
X_train, X_test, y_train, y_test = train_test_split(
    df_knn.drop('target', axis=1),
    y,
    test_size=0.2,
    random_state=42
)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(f"Accuracy with KNN imputation: {accuracy_score(y_test, predictions):.4f}")
```

### Visualizing Missing Data

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap of missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title('Missing Data Pattern')
plt.xlabel('Features')
plt.ylabel('Samples')
plt.show()

# Bar chart of missing percentages
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_pct.plot(kind='bar', figsize=(10, 4))
plt.title('Percentage of Missing Values per Feature')
plt.ylabel('Missing %')
plt.show()
```

---

## Vision Module

Diabetic retinopathy classification with explainable AI (Grad-CAM).

### Full Pipeline (Notebook)

```bash
jupyter notebook vision/diabetic_retinopathy_pipeline.ipynb
```

**Workflow:**
1. Auto-download Kaggle dataset
2. Train baseline CNN
3. Train improved EfficientNetB0 model
4. Generate evaluation metrics
5. Create Grad-CAM visualizations

### CLI Training

#### Quick Training

```bash
python vision/cli_train_model.py \
  --data-dir data/retinopathy \
  --epochs 10 \
  --batch-size 32 \
  --img-size 224 \
  --auto-download
```

#### Advanced Training with Custom Parameters

```bash
python vision/cli_train_model.py \
  --data-dir data/retinopathy \
  --epochs 20 \
  --batch-size 16 \
  --img-size 256 \
  --limit-per-class 200 \
  --output-dir vision/models_custom \
  --checkpoint-freq 5
```

**Parameters:**
- `--epochs`: Training epochs (default: 15)
- `--batch-size`: Batch size (reduce if OOM)
- `--img-size`: Image dimensions (224, 256, 384)
- `--limit-per-class`: Limit samples per class (for testing)
- `--checkpoint-freq`: Save checkpoint every N epochs

### Using Trained Models

#### Load Model for Inference

```python
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model and class names
model = tf.keras.models.load_model('vision/models/model_improved.keras')
with open('vision/models/class_names.json', 'r') as f:
    class_names = json.load(f)

# Preprocess image
def preprocess_image(image_path, img_size=224):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict
image_path = 'test_retina_image.jpg'
img_array = preprocess_image(image_path)
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions[0])]
confidence = np.max(predictions[0]) * 100

print(f"Predicted: {predicted_class} ({confidence:.2f}% confidence)")
print(f"All probabilities: {dict(zip(class_names, predictions[0]))}")
```

#### Batch Prediction

```python
from pathlib import Path

# Predict on all images in a directory
image_dir = Path('test_images/')
results = []

for img_path in image_dir.glob('*.jpg'):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    results.append({
        'image': img_path.name,
        'prediction': predicted_class,
        'confidence': confidence
    })

# Save results
import pandas as pd
df_results = pd.DataFrame(results)
df_results.to_csv('batch_predictions.csv', index=False)
print(df_results)
```

### Grad-CAM Explainability

#### CLI Grad-CAM

```bash
python main.py gradcam \
  --model vision/models/model_improved.keras \
  --image test_image.jpg \
  --output gradcam_output.jpg
```

#### Programmatic Grad-CAM

```python
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

def generate_gradcam(model, img_array, last_conv_layer_name='top_conv'):
    # Create Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        class_channel = predictions[:, predicted_class]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Generate heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def overlay_heatmap(image_path, heatmap, alpha=0.4):
    # Load original image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Overlay
    superimposed = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    return superimposed

# Generate and save Grad-CAM
image_path = 'retina_image.jpg'
img_array = preprocess_image(image_path)
heatmap = generate_gradcam(model, img_array)
gradcam_img = overlay_heatmap(image_path, heatmap)

# Save
Image.fromarray(gradcam_img).save('gradcam_result.jpg')
```

### Model Evaluation

```python
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load metrics
with open('vision/models/metrics_improved.json', 'r') as f:
    metrics = json.load(f)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")

# Confusion matrix
cm = np.array(metrics['confusion_matrix'])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("\nClassification Report:")
print(metrics['classification_report'])
```

---

## Advanced Workflows

### 1. End-to-End Healthcare Pipeline

Combine all three modules for comprehensive healthcare data processing:

```python
# Step 1: Impute missing patient data
from sklearn.impute import KNNImputer
import pandas as pd

patient_data = pd.read_csv('patient_records.csv')
imputer = KNNImputer(n_neighbors=5)
patient_data_clean = pd.DataFrame(
    imputer.fit_transform(patient_data),
    columns=patient_data.columns
)

# Step 2: Predict retinopathy risk
model = tf.keras.models.load_model('vision/models/model_improved.keras')
for patient_id, retina_image in patient_images.items():
    img_array = preprocess_image(retina_image)
    risk_score = model.predict(img_array)[0]
    patient_data_clean.loc[patient_id, 'retinopathy_risk'] = risk_score

# Step 3: Generate chatbot recommendations
from chatbot.healthcare_chatbot import get_recommendation

for _, patient in patient_data_clean.iterrows():
    if patient['retinopathy_risk'] > 0.7:
        recommendation = "Schedule urgent ophthalmology appointment"
    else:
        recommendation = "Routine eye exam recommended"
    
    print(f"Patient {patient['id']}: {recommendation}")
```

### 2. Model Fine-Tuning

Continue training a saved model:

```python
import tensorflow as tf

# Load existing model
model = tf.keras.models.load_model('vision/models/model_improved.keras')

# Prepare new data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/new_retinopathy_images/train',
    image_size=(224, 224),
    batch_size=32
)

# Fine-tune
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, epochs=5)
model.save('vision/models/model_finetuned.keras')
```

### 3. Hyperparameter Tuning

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    img_size = trial.suggest_categorical('img_size', [128, 224, 256])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    
    # Build and train model
    model = build_model(img_size=img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(train_ds, epochs=10, validation_data=val_ds, verbose=0)
    return max(history.history['val_accuracy'])

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best parameters: {study.best_params}")
print(f"Best accuracy: {study.best_value:.4f}")
```

---

## Configuration

### YAML Configuration (`configs/model_params.yaml`)

```yaml
vision:
  img_size: 224
  batch_size: 32
  epochs: 15
  learning_rate: 0.001
  augmentation:
    rotation: 0.2
    zoom: 0.15
    flip: horizontal

data_quality:
  default_method: knn
  knn_neighbors: 5
  mice_max_iter: 10

chatbot:
  save_format: binary
  pdf_export: true
  auto_save: true

logging:
  level: INFO
  format: "%(asctime)s - %(levelname)s - %(message)s"

paths:
  data_root: data/
  output_dir: outputs/
  models_dir: vision/models/
```

### Environment Variables

```bash
# .env file
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
HEALTHML_DATA_DIR=/custom/data/path
HEALTHML_LOG_LEVEL=DEBUG
```

Load in Python:

```python
from dotenv import load_dotenv
import os

load_dotenv()
kaggle_user = os.getenv('KAGGLE_USERNAME')
```

---

## Tips & Best Practices

### Performance Optimization

1. **Use Mixed Precision (Vision)**:
   ```python
   from tensorflow.keras.mixed_precision import set_global_policy
   set_global_policy('mixed_float16')
   ```

2. **Dataset Caching**:
   ```python
   train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
   ```

3. **Limit Data for Testing**:
   ```bash
   python vision/cli_train_model.py --limit-per-class 100
   ```

### Reproducibility

```python
import numpy as np
import tensorflow as tf
import random

# Set seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
```

### Monitoring Training

```python
# TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs/vision',
    histogram_freq=1
)

model.fit(train_ds, epochs=15, callbacks=[tensorboard_callback])

# View logs
# tensorboard --logdir logs/vision
```

---

## Next Steps

- **Explore Notebooks**: Interactive workflows with visualizations
- **Contribute**: See [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design

---

**Questions? Open an issue or discussion on GitHub!**
