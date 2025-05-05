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
from transformers import Trainer, TrainingArguments, AutoModelForImageClassification, AutoImageProcessor

from czech_railway_lights_net import CzechRailwayLightNet


def confusion_matrix_to_pdf(confusion_matrix, class_names=None, output_path='confusion_matrix.pdf',
                            title='Confusion Matrix', cmap='Blues', normalize=False, figsize=(10, 8)):
    cm = confusion_matrix.copy()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # Create class names if not provided
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

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    conf_matrix = confusion_matrix(labels, preds)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        "conf_matrix": conf_matrix.tolist()
    }


# Custom Dataset Wrapper for Hugging Face Trainer
class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Load the original dataset
        self.dataset = ImageFolder(root=data_dir)
        self.transform = transform

        # Load class mapping from YAML
        with open("../metacentrum_experiments/CRL_single_images_less_balanced.yaml") as f:
            config = yaml.load(f, yaml.SafeLoader)
            self.interesting_labels = config["names"]

        # Create mapping from folder names to indices based on interesting_labels
        folder_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.interesting_labels)}

        # Get the reverse mapping from original dataset
        original_idx_to_folder = {idx: cls_name for cls_name, idx in self.dataset.class_to_idx.items()}

        # Remap samples to use our custom order
        self.samples = []
        self.targets = []

        for path, original_idx in self.dataset:
            folder_name = original_idx_to_folder[original_idx]
            if folder_name in folder_to_idx:
                new_idx = folder_to_idx[folder_name]
                self.samples.append((path, new_idx))
                self.targets.append(new_idx)

        # Update class information
        self.classes = self.interesting_labels
        self.class_to_idx = folder_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return {
            'pixel_values': img,
            'label': label
        }

def load_data(data_dir):
    transform = transforms.Compose([
        transforms.Resize((72, 34)), # Resize to the small image size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create custom dataset
    dataset = CustomImageDataset(data_dir=data_dir, transform=transform)
    return dataset

def split_data(dataset, val_size=0.2):
    # Splitting dataset into train and validation sets
    train_size = int((1 - val_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def create_dataloader(train_dataset, val_dataset, batch_size=32):
    # Creating DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def load_model(num_classes, model_name='google/efficientnet-b0'):
    model = CzechRailwayLightNet.from_pretrained()
    return model

# Step 4: Load Pre-trained EfficientNet Model
def load_model_mobilenet(num_classes, model_name="google/mobilenet_v2_1.0_224", img_size=(72, 34)):
    """
    Load and modify a pre-trained MobileNet model for rectangular tensor dimensions

    Args:
        num_classes: Number of output classes for classification
        model_name: Name of the model on Hugging Face hub
        img_size: Tuple of (height, width) for the input image tensor

    Returns:
        model: Modified model ready for fine-tuning or inference
        transform: Preprocessing transform pipeline for images
        processor: Original image processor (may be needed for some operations)
    """
    height, width = img_size

    print(f"Loading model {model_name} and modifying for tensor dimensions {height}x{width} with {num_classes} classes")

    # Load the pre-trained model and processor from Hugging Face hub
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)

        # Configure processor for rectangular images
        if hasattr(processor, 'size'):
            if isinstance(processor.size, dict):
                processor.size['height'] = height
                processor.size['width'] = width
            else:
                # Some processors use a single size parameter for both dimensions
                # We'll need to modify our transform accordingly
                pass

        # Load model with custom config for rectangular input
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

    # Create a custom transform pipeline for the specific input size
    transform = transforms.Compose([
        transforms.Resize((height, width)),  # Resize to the specified dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Deep modification of the model to handle rectangular tensor dimensions

    # First, modify the embedding layer for non-square input if applicable
    if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'patch_embeddings'):
        # Vision Transformer models have patch embeddings with specific dimensions
        patch_size = getattr(model.embeddings.patch_embeddings, 'patch_size', 16)
        if isinstance(patch_size, int):
            # Adjust for rectangular input
            grid_height = height // patch_size
            grid_width = width // patch_size
            model.embeddings.patch_embeddings.num_patches = grid_height * grid_width
            if hasattr(model.embeddings, 'position_embeddings'):
                # May need to interpolate position embeddings for new dimensions
                pass

    # Modify convolutional models (like MobileNet) for rectangular input
    # MobileNetV2 and EfficientNet have specific structures we need to adapt
    if "mobilenet_v2" in model_name.lower() or "efficientnet" in model_name.lower():
        # Need to verify and possibly modify the convolutional backbone
        # For non-square inputs, we need to track feature map dimensions

        # Replace fixed pooling with adaptive pooling throughout the network
        if hasattr(model, 'pool'):
            model.pool = nn.AdaptiveAvgPool2d((1, 1))

        # For models that have pooling operations in Sequential containers
        for module in model.modules():
            if isinstance(module, nn.Sequential):
                for i, layer in enumerate(module):
                    if isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.MaxPool2d):
                        module[i] = nn.AdaptiveAvgPool2d((1, 1))

        # For models with global pooling before classifier
        if hasattr(model, 'avgpool'):
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Find and modify any hard-coded tensor reshaping operations
        # This is model-specific and may require introspection of the forward method

    # Modify classifier head
    # Approach 1: If the model has a standard classifier structure
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        # Keep in_features the same but update out_features
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # Approach 2: If the model has a different classifier structure
    elif hasattr(model, 'classifier') and hasattr(model.classifier, 'fc'):
        model.classifier.fc = nn.Linear(model.classifier.fc.in_features, num_classes)

    # Approach 3: For models with 'classifiers' (plural)
    elif hasattr(model, 'classifiers'):
        model.classifiers = nn.Linear(model.classifiers.in_features, num_classes)

    # Approach 4: Handle MobileNetV2 specific architecture
    elif "mobilenet_v2" in model_name.lower():
        # Most MobileNetV2 variants have this structure
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            last_layer_idx = len(model.classifier) - 1
            in_features = model.classifier[last_layer_idx].in_features
            model.classifier[last_layer_idx] = nn.Linear(in_features, num_classes)

    # Add a custom forward hook to validate and fix tensor dimensions if needed
    def fix_tensor_dimensions(module, input_tensor, output_tensor):
        # This hook can be used to debug and fix dimension issues
        # Uncomment for debugging:
        # print(f"Shape before module: {input_tensor[0].shape if isinstance(input_tensor, tuple) else input_tensor.shape}")
        # print(f"Shape after module: {output_tensor.shape}")
        return output_tensor

    # Apply the hook to critical layers (uncomment for debugging)
    # model.register_forward_hook(fix_tensor_dimensions)

    # For MobileNetV2, we may need to modify the feature extractor
    if "mobilenet_v2" in model_name.lower():
        # Override the forward method to handle rectangular inputs
        original_forward = model.forward

        def custom_forward(self, pixel_values):
            # Get features from the base model
            features = self.mobilenet_v2.features(pixel_values)

            # Global average pooling to handle any input size
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))

            # Flatten
            x = torch.flatten(pooled, 1)

            # Pass through classifier
            logits = self.classifier(x)

            # Return in the format expected by the model
            return logits

        # Only bind if the model has the expected structure
        if hasattr(model, 'mobilenet_v2') and hasattr(model.mobilenet_v2, 'features'):
            # Bind the custom forward method to the model instance
            import types
            model.forward = types.MethodType(custom_forward, model)

    # For EfficientNet, which has a different structure
    elif "efficientnet" in model_name.lower():
        # Similar customization for EfficientNet if needed
        pass

    # Test forward pass with a dummy input to catch any shape issues
    try:
        dummy_input = torch.randn(1, 3, height, width)
        _ = model(dummy_input)
        print("Model successfully handles input shape:", dummy_input.shape)
    except Exception as e:
        print(f"Warning: Forward pass with shape {height}x{width} failed: {e}")
        print("You may need to implement a custom forward method for this model architecture")

    print("Model architecture modification complete")
    return model
def get_training_args(output_dir="./results"):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=30,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.001,
        learning_rate=0.0001,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
    )


# Define custom loss function if needed
class CustomLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels):
        return self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

def train_model(train_loader, val_loader, model, training_args):# Initialize trainer with loss function

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_metrics=compute_metrics,

    )

    # Start training
    trainer.train()

    # Evaluate the fine-tuned model
    return trainer.evaluate()


def save_model_to_pt(model, filepath='model.pt'):
    """
    Save the model to a PyTorch .pt file

    Args:
        model: The trained model to save
        filepath: Path where to save the model
    """
    # Assume `model` is your instance of CzechRailwayLightNet

    # Save the model
    torch.save(model, filepath)
    print(f"Model saved to {filepath}")


def main():
    data_dir = '../reconstructed/czech_railway_light_dataset_roi'
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist!")

    dataset = load_data(data_dir)

    dataset_dist = {}
    for i in os.listdir(data_dir):
        if os.path.isdir(data_dir + "/" + i):
            dataset_dist[i] = len(list(os.listdir(data_dir + "/" + i)))

    print(f"Number of classes in dataset: {len(dataset.classes)}, \n dataset_dist: {dataset_dist}")

    val_size = 0.15
    train_dataset, val_dataset = split_data(dataset, val_size=val_size)
    train_loader, val_loader = create_dataloader(train_dataset, val_dataset)

    num_classes = len(dataset.classes)
    model_name = "czech_railway_lights_net"
    model = load_model(num_classes, model_name=model_name)

    training_args = get_training_args()

    results = train_model(train_loader, val_loader, model, training_args)

    # Save the model to a PyTorch .pt file
    save_model_to_pt(model, filepath='czech_railway_lights_net.pt')

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset_classes": dataset_dist,
            "dataset_val_size": val_size,
            "model_name": model_name,
            "results": results,
            "training_args": {"learning_rate": training_args.learning_rate,
                              "num_train_epochs": training_args.num_train_epochs,
                              "weight_decay": training_args.weight_decay}}, f, indent=2, ensure_ascii=True)

    confusion_matrix_to_pdf(
        confusion_matrix=np.array(results["eval_conf_matrix"]),
        class_names=dataset.classes,
        title='',
        output_path='confusion_matrix_normalized.pdf',
        normalize=True
    )
    confusion_matrix_to_pdf(
        confusion_matrix=np.array(results["eval_conf_matrix"]),
        class_names=dataset.classes,
        title='',
        normalize=False
    )



if __name__ == "__main__":
    main()
