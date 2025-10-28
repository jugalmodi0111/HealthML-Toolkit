"""
HealthML-Toolkit: Unified CLI Orchestrator

Command-line interface for running healthcare ML workflows:
- Chatbot: Interactive conversational AI for health queries
- Imputation: Data quality and missing value handling
- Vision: Diabetic retinopathy image classification with Grad-CAM

Usage:
    python main.py chatbot --interactive
    python main.py impute --input data.csv --method knn --output imputed.csv
    python main.py vision-train --data-dir data/retinopathy --epochs 15
    python main.py gradcam --model vision/models/model_improved.keras --image sample.jpg
"""

import argparse
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('healthml_toolkit.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('HealthML-Toolkit')


def run_chatbot(args):
    """Launch the healthcare chatbot"""
    logger.info("Starting Healthcare Chatbot...")
    logger.info("Note: For full interactive experience, use: jupyter notebook chatbot/healthcare_chatbot.ipynb")
    
    # Minimal CLI version
    try:
        import nltk
        from nltk.chat.util import Chat, reflections
    except ImportError:
        logger.error("NLTK not installed. Run: pip install nltk")
        return
    
    pairs = [
        [r"(hi|hello|hey)", ["Hello! I am your healthcare assistant. How can I help you today?"]],
        [r"(.*) (fever|temperature)", ["Stay hydrated and consult a doctor if fever persists."]],
        [r"(.*) (headache|migraine)", ["Headaches have many causes. If severe, consult a professional."]],
        [r"quit", ["Take care of your health. Goodbye!"]],
    ]
    
    chatbot = Chat(pairs, reflections)
    print("\n💬 Healthcare Chatbot (type 'quit' to exit)")
    print("=" * 50)
    chatbot.converse()


def run_imputation(args):
    """Run data imputation pipeline"""
    logger.info(f"Running imputation: method={args.method}, input={args.input}, output={args.output}")
    
    try:
        import pandas as pd
        from sklearn.impute import SimpleImputer, KNNImputer
        import numpy as np
    except ImportError:
        logger.error("Required packages not installed. Run: pip install pandas scikit-learn numpy")
        return
    
    # Load data
    df = pd.read_csv(args.input)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Missing values: {df.isna().sum().sum()} total")
    
    # Apply imputation
    if args.method == 'mean':
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    elif args.method == 'knn':
        imputer = KNNImputer(n_neighbors=args.knn_neighbors)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    else:
        logger.error(f"Unsupported method: {args.method}")
        return
    
    # Save
    df_imputed.to_csv(args.output, index=False)
    logger.info(f"Imputed data saved to: {args.output}")
    logger.info(f"Remaining missing values: {df_imputed.isna().sum().sum()}")


def run_vision_train(args):
    """Train diabetic retinopathy model"""
    logger.info("Launching vision training...")
    logger.info("Note: For full pipeline with Grad-CAM, use: jupyter notebook vision/diabetic_retinopathy_pipeline.ipynb")
    
    # Import the CLI trainer
    sys.path.insert(0, str(Path(__file__).parent / 'vision'))
    try:
        from cli_train_model import main as train_main
    except ImportError:
        logger.error("Could not import vision trainer. Check vision/cli_train_model.py")
        return
    
    # Build sys.argv for the trainer
    sys.argv = [
        'cli_train_model.py',
        '--data-dir', args.data_dir,
        '--epochs', str(args.epochs),
        '--img-size', str(args.img_size),
        '--batch-size', str(args.batch_size),
        '--out-dir', args.out_dir,
    ]
    if args.limit_per_class:
        sys.argv.extend(['--limit-per-class', str(args.limit_per_class)])
    if args.auto_download:
        sys.argv.append('--auto-download')
    
    train_main()


def run_gradcam(args):
    """Generate Grad-CAM visualization for a single image"""
    logger.info(f"Generating Grad-CAM for: {args.image}")
    logger.info("Note: For batch Grad-CAM, use the notebook: vision/diabetic_retinopathy_pipeline.ipynb")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np
        import cv2
        from PIL import Image
    except ImportError:
        logger.error("Required packages not installed. Run: pip install tensorflow opencv-python pillow")
        return
    
    # Load model
    model = keras.models.load_model(args.model)
    logger.info(f"Loaded model: {args.model}")
    
    # Find last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        logger.error("No Conv2D layer found in the model")
        return
    
    logger.info(f"Using Conv2D layer: {last_conv_layer.name}")
    
    # Load and preprocess image
    img = Image.open(args.image).convert('RGB')
    img_size = args.img_size
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img_array = np.array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Grad-CAM
    grad_model = keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    
    # Overlay
    heatmap = cv2.resize(heatmap, (img_size, img_size))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, 0.4, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR), 0.6, 0)
    
    # Save
    output_path = args.output or f"gradcam_{Path(args.image).stem}.jpg"
    cv2.imwrite(output_path, overlay)
    logger.info(f"Grad-CAM saved to: {output_path}")
    logger.info(f"Predicted class index: {int(pred_index)}")


def main():
    parser = argparse.ArgumentParser(
        description='HealthML-Toolkit: Unified healthcare ML pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Chatbot command
    chatbot_parser = subparsers.add_parser('chatbot', help='Run healthcare chatbot')
    chatbot_parser.add_argument('--interactive', action='store_true', help='Interactive mode (default)')
    
    # Imputation command
    impute_parser = subparsers.add_parser('impute', help='Run data imputation')
    impute_parser.add_argument('--input', required=True, help='Input CSV file with missing values')
    impute_parser.add_argument('--output', required=True, help='Output CSV file for imputed data')
    impute_parser.add_argument('--method', choices=['mean', 'knn'], default='knn', help='Imputation method')
    impute_parser.add_argument('--knn-neighbors', type=int, default=5, help='K for KNN imputation')
    
    # Vision training command
    vision_parser = subparsers.add_parser('vision-train', help='Train diabetic retinopathy model')
    vision_parser.add_argument('--data-dir', required=True, help='Path to dataset directory')
    vision_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    vision_parser.add_argument('--img-size', type=int, default=224, help='Image size (square)')
    vision_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    vision_parser.add_argument('--limit-per-class', type=int, default=None, help='Limit samples per class (for testing)')
    vision_parser.add_argument('--out-dir', default='vision/models', help='Output directory')
    vision_parser.add_argument('--auto-download', action='store_true', help='Auto-download from Kaggle')
    
    # Grad-CAM command
    gradcam_parser = subparsers.add_parser('gradcam', help='Generate Grad-CAM visualization')
    gradcam_parser.add_argument('--model', required=True, help='Path to trained Keras model')
    gradcam_parser.add_argument('--image', required=True, help='Input image path')
    gradcam_parser.add_argument('--output', help='Output path (default: gradcam_<input>.jpg)')
    gradcam_parser.add_argument('--img-size', type=int, default=224, help='Image size')
    
    args = parser.parse_args()
    
    if args.command == 'chatbot':
        run_chatbot(args)
    elif args.command == 'impute':
        run_imputation(args)
    elif args.command == 'vision-train':
        run_vision_train(args)
    elif args.command == 'gradcam':
        run_gradcam(args)
    else:
        parser.print_help()
        logger.info("\n💡 Tip: For full interactive experience, use Jupyter notebooks in respective folders")


if __name__ == '__main__':
    main()
