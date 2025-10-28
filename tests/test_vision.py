"""
HealthML-Toolkit Test Suite - Vision Pipeline

Tests for diabetic retinopathy classification and Grad-CAM.
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path


class TestModelArchitecture:
    """Test model creation and architecture."""
    
    def test_baseline_cnn_structure(self):
        """Test baseline CNN creation."""
        # Arrange
        num_classes = 5
        img_size = 224
        
        # Act - Simplified baseline model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(img_size, img_size, 3)),
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Assert
        assert model.input_shape == (None, img_size, img_size, 3)
        assert model.output_shape == (None, num_classes)
        assert len([l for l in model.layers if 'conv' in l.name]) >= 2
    
    def test_efficientnet_transfer_learning(self):
        """Test EfficientNetB0 transfer learning setup."""
        # Arrange
        num_classes = 5
        img_size = 224
        
        # Act
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,  # No weights for testing
            input_shape=(img_size, img_size, 3)
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Assert
        assert not base_model.trainable
        assert model.output_shape == (None, num_classes)


class TestDataPipeline:
    """Test tf.data pipeline creation."""
    
    def test_image_normalization(self):
        """Test image rescaling to [0, 1]."""
        # Arrange
        img_array = np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8)
        
        # Act
        rescaling_layer = tf.keras.layers.Rescaling(1./255)
        normalized = rescaling_layer(img_array)
        
        # Assert
        assert normalized.numpy().min() >= 0.0
        assert normalized.numpy().max() <= 1.0
    
    def test_augmentation_layers(self):
        """Test data augmentation does not change shape."""
        # Arrange
        img_size = 224
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.15)
        ])
        
        img = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
        
        # Act
        augmented = augmentation(img, training=True)
        
        # Assert
        assert augmented.shape == img.shape


class TestGradCAM:
    """Test Grad-CAM heatmap generation."""
    
    def test_gradcam_heatmap_shape(self):
        """Test Grad-CAM heatmap has correct dimensions."""
        # Arrange
        img_size = 224
        num_classes = 5
        
        # Create simple model with Conv2D
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(img_size, img_size, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu', name='last_conv'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        img_array = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
        
        # Act - Simplified Grad-CAM
        last_conv_layer = model.get_layer('last_conv')
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            predicted_class = tf.argmax(predictions[0])
            class_channel = predictions[:, predicted_class]
        
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Assert
        assert grads is not None, "Gradients are None"
        assert grads.shape[1:3] == conv_outputs.shape[1:3], "Gradient shape mismatch"


class TestTraining:
    """Test training workflow."""
    
    @pytest.mark.slow
    def test_single_epoch_training(self):
        """Test model can train for 1 epoch on dummy data."""
        # Arrange
        num_classes = 5
        img_size = 64  # Smaller for speed
        batch_size = 4
        num_samples = 20
        
        # Create dummy dataset
        X = np.random.rand(num_samples, img_size, img_size, 3).astype(np.float32)
        y = np.random.randint(0, num_classes, num_samples)
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(batch_size)
        
        # Simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(16, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Act
        history = model.fit(dataset, epochs=1, verbose=0)
        
        # Assert
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        assert len(history.history['loss']) == 1


class TestModelEvaluation:
    """Test evaluation metrics."""
    
    def test_classification_metrics(self):
        """Test classification report generation."""
        # Arrange
        from sklearn.metrics import classification_report, accuracy_score
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1])
        
        # Act
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Assert
        assert 0.0 <= accuracy <= 1.0
        assert 'accuracy' in report
        assert 'macro avg' in report
    
    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        # Arrange
        from sklearn.metrics import confusion_matrix
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1])
        
        # Act
        cm = confusion_matrix(y_true, y_pred)
        
        # Assert
        assert cm.shape == (3, 3)  # 3 classes
        assert cm.sum() == len(y_true)


class TestUtilities:
    """Test utility functions."""
    
    def test_class_weight_computation(self):
        """Test class weight calculation for imbalance."""
        # Arrange
        from sklearn.utils.class_weight import compute_class_weight
        
        y = np.array([0, 0, 0, 1, 1, 2])  # Imbalanced: 3:2:1
        
        # Act
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y),
            y=y
        )
        
        # Assert
        assert len(class_weights) == 3
        assert class_weights[0] < class_weights[2]  # Class 0 has more samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
