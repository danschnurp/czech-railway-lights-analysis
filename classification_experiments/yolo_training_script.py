import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import Trainer, TrainingArguments
from PIL import Image
import cv2

# Import ultralytics YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics not available. Install with: pip install ultralytics")

def confusion_matrix_to_pdf(confusion_matrix, class_names=None, output_path='confusion_matrix.pdf',
                            title='Confusion Matrix', cmap='Blues', normalize=False, figsize=(10, 8)):
    cm = confusion_matrix.copy()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, square=True,
                xticklabels=class_names, yticklabels=class_names, cbar=True, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    if not normalize:
        for i, total in enumerate(np.sum(cm, axis=1)):
            ax.text(-0.5, i, f'Total: {total}', ha='right', va='center')
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(cm.shape[1] / 2, -0.5, f'Accuracy: {accuracy:.2f}',
                ha='center', va='center', fontsize=12)
    plt.tight_layout()
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig)
    plt.close()
    return output_path

class YOLOClassificationDataset(Dataset):
    """Custom dataset for YOLO classification training"""
    def __init__(self, data_dir, img_size=(68, 144)):
        self.dataset = ImageFolder(root=data_dir)
        self.img_size = img_size
        
        # Load class mapping from YAML
        with open("../metacentrum_experiments/CRL_single_images_less_balanced.yaml") as f:
            config = yaml.load(f, yaml.SafeLoader)
            self.interesting_labels = config["names"]

        # Create mapping from folder names to indices
        folder_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.interesting_labels)}
        original_idx_to_folder = {idx: cls_name for cls_name, idx in self.dataset.class_to_idx.items()}

        # Remap samples to use custom order
        self.samples = []
        self.targets = []

        for path, original_idx in self.dataset.samples:
            folder_name = original_idx_to_folder[original_idx]
            if folder_name in folder_to_idx:
                new_idx = folder_to_idx[folder_name]
                self.samples.append((path, new_idx))
                self.targets.append(new_idx)

        self.classes = self.interesting_labels
        self.class_to_idx = folder_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        
        # Resize image to specified dimensions
        image = image.resize(self.img_size)
        
        return image, label

def create_yolo_classification_dataset_structure(data_dir, output_dir="yolo_dataset"):
    """
    Create YOLO-compatible dataset structure for classification
    YOLO classification expects: dataset/train/class_name/image.jpg
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    # Create train and val directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Load original dataset
    dataset = ImageFolder(root=data_dir)
    
    # Split data (80% train, 20% val)
    from sklearn.model_selection import train_test_split
    
    for class_name, class_idx in dataset.class_to_idx.items():
        class_samples = [s for s in dataset.samples if s[1] == class_idx]
        
        if len(class_samples) == 0:
            continue
            
        # Split samples
        train_samples, val_samples = train_test_split(
            class_samples, test_size=0.2, random_state=42
        )
        
        # Create class directories
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Copy or symlink files
        import shutil
        
        for i, (src_path, _) in enumerate(train_samples):
            dst_path = os.path.join(train_class_dir, f"{i}_{os.path.basename(src_path)}")
            shutil.copy2(src_path, dst_path)
        
        for i, (src_path, _) in enumerate(val_samples):
            dst_path = os.path.join(val_class_dir, f"{i}_{os.path.basename(src_path)}")
            shutil.copy2(src_path, dst_path)
    
    return output_dir, train_dir, val_dir

def load_model_yolo_ultralytics(num_classes, model_name='yolov8n', img_size=(160, 160)):
    """
    Load Ultralytics YOLO model for classification
    """
    if not ULTRALYTICS_AVAILABLE:
        raise ImportError("Ultralytics not available. Install with: pip install ultralytics")
    
    height, width = img_size
    print(f"Loading Ultralytics YOLO model {model_name} for classification with {num_classes} classes")
    
    try:
        # For classification, use YOLO classification models
        if 'cls' not in model_name:
            model_name = model_name.replace('.pt', '') + '-cls.pt'
        
        # Load pre-trained classification model
        model = YOLO(model_name)
        
        # The model will be automatically configured for the number of classes
        # during training when we provide the dataset structure
        
        print(f"YOLO classification model {model_name} loaded successfully")
        return model, None, None
        
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        # Fallback to detection model and modify for classification
        try:
            model = YOLO(model_name.replace('-cls', ''))
            print(f"Loaded detection model {model_name}, will modify for classification")
            return model, None, None
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
            return None, None, None

class YOLOTrainer:
    """Custom trainer for YOLO models"""
    
    def __init__(self, model, dataset_path, img_size=(68, 144), epochs=30, batch_size=16):
        self.model = model
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.epochs = epochs
        self.batch_size = batch_size
        
    def train(self):
        """Train the YOLO model"""
        if self.model is None:
            raise ValueError("Model not loaded properly")
        
        # Train the model
        results = self.model.train(
            data=self.dataset_path,
            epochs=self.epochs,
            imgsz=max(self.img_size),  # YOLO uses single size, take the larger dimension
            batch=self.batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            project='yolo_training',
            name='czech_railway_lights',
            exist_ok=True,
            verbose=True
        )
        
        return results
    
    def evaluate(self, val_path=None):
        """Evaluate the trained model"""
        if val_path is None:
            val_path = os.path.join(self.dataset_path, 'val')
        
        # Validate the model
        results = self.model.val(
            data=self.dataset_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        return results
    
    def predict_and_get_metrics(self, val_dataset):
        """Get detailed metrics including confusion matrix"""
        predictions = []
        true_labels = []
        
        for image, label in val_dataset:
            # Convert PIL image to numpy for YOLO
            img_array = np.array(image)
            
            # Predict
            results = self.model.predict(img_array, verbose=False)
            
            if len(results) > 0 and hasattr(results[0], 'probs'):
                pred_class = results[0].probs.top1
                predictions.append(pred_class)
                true_labels.append(label)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'conf_matrix': conf_matrix.tolist(),
            'predictions': predictions,
            'true_labels': true_labels
        }

def load_data_yolo(data_dir, img_size=(68, 144)):
    """Load data for YOLO training"""
    dataset = YOLOClassificationDataset(data_dir=data_dir, img_size=img_size)
    return dataset

def split_data_yolo(dataset, val_size=0.2):
    """Split dataset for YOLO training"""
    train_size = int((1 - val_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def train_model_yolo(model, dataset_path, img_size=(68, 144), epochs=30):
    """Train YOLO model"""
    trainer = YOLOTrainer(model, dataset_path, img_size, epochs)
    
    # Train the model
    train_results = trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    
    return train_results, eval_results, trainer

def save_yolo_model(model, filepath='czech_railway_lights_yolo.pt'):
    """Save YOLO model"""
    try:
        # YOLO models have built-in save functionality
        model.save(filepath)
        print(f"YOLO model saved to {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")
        # Fallback to torch.save
        torch.save(model, filepath)
        print(f"Model saved using torch.save to {filepath}")

def main():
    data_dir = '../reconstructed/czech_railway_light_dataset_roi'
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist!")

    # Create YOLO-compatible dataset structure
    print("Creating YOLO dataset structure...")
    yolo_dataset_path, train_dir, val_dir = create_yolo_classification_dataset_structure(data_dir)
    
    # Get dataset distribution
    dataset_dist = {}
    for i in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, i)):
            dataset_dist[i] = len(list(os.listdir(os.path.join(data_dir, i))))

    print(f"Dataset distribution: {dataset_dist}")

    # Load original dataset for metrics calculation
    original_dataset = load_data_yolo(data_dir, img_size=(68, 144))
    train_dataset, val_dataset = split_data_yolo(original_dataset, val_size=0.2)

    num_classes = len(original_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {original_dataset.classes}")

    # Load YOLO model
    print("Loading YOLO model...")
    model, _, _ = load_model_yolo_ultralytics(num_classes, model_name="yolov8n-cls")
    
    if model is None:
        print("Failed to load YOLO model. Exiting.")
        return

    # Define class mappings
    id2label = {
        0: 'stop',
        1: 'go', 
        2: 'warning',
        3: 'adjust speed and warning',
        4: 'adjust speed and go',
        5: 'lights off'
    }
    label2id = {value: key for key, value in id2label.items()}

    # Train the model
    print("Starting YOLO training...")
    try:
        train_results, eval_results, trainer = train_model_yolo(
            model, yolo_dataset_path, img_size=(68, 144), epochs=30
        )
        
        # Get detailed metrics
        print("Calculating detailed metrics...")
        detailed_metrics = trainer.predict_and_get_metrics(val_dataset)
        
        # Save the model
        save_yolo_model(model, 'czech_railway_lights_yolo.pt')
        
        # Save results
        results = {
            "dataset_classes": dataset_dist,
            "model_name": "yolov8n-cls_ultralytics",
            "num_classes": num_classes,
            "class_names": original_dataset.classes,
            "detailed_metrics": detailed_metrics,
            "training_completed": True
        }
        
        with open("yolo_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=3, ensure_ascii=True)
        
        # Generate confusion matrices
        if 'conf_matrix' in detailed_metrics:
            confusion_matrix_to_pdf(
                confusion_matrix=np.array(detailed_metrics["conf_matrix"]),
                class_names=original_dataset.classes,
                title='YOLO Classification Results',
                output_path='yolo_confusion_matrix_normalized.pdf',
                normalize=True
            )
            confusion_matrix_to_pdf(
                confusion_matrix=np.array(detailed_metrics["conf_matrix"]),
                class_names=original_dataset.classes,
                title='YOLO Classification Results',
                output_path='yolo_confusion_matrix.pdf',
                normalize=False
            )
        
        print("Training completed successfully!")
        print(f"Final metrics: Accuracy: {detailed_metrics.get('accuracy', 'N/A'):.4f}, "
              f"F1: {detailed_metrics.get('f1', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Training failed. Check your dataset and model configuration.")

if __name__ == "__main__":
    main()