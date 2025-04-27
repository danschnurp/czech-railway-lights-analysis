import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoConfig
from transformers import Trainer, TrainingArguments

from classifiers.crl_model import CzechRailwayLightNet


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
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    conf_matrix = confusion_matrix(labels, preds)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        "conf_matrix": conf_matrix.tolist()
    }


# Custom Dataset Wrapper for Hugging Face Trainer
class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = ImageFolder(root=data_dir)
        self.transform = transform
        self.classes = self.dataset.classes

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
        transforms.Resize((16, 34)),  # Resize to the small image size
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
    model = CzechRailwayLightNet.from_pretrained(model_name, num_labels=num_classes)
    return model

def get_training_args(output_dir="./results"):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.001,
        learning_rate=0.001,
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=3,
    )

def train_model(train_loader, val_loader, model, training_args):
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
    # Save the entire model
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_pt_model(filepath='model.pt', num_classes=None):
    """
    Load a model from a PyTorch .pt file

    Args:
        filepath: Path to the saved model
        num_classes: Number of output classes for the model

    Returns:
        Loaded model
    """
    # Create a new model instance with the same architecture
    config = AutoConfig.from_pretrained('google/efficientnet-b0', num_labels=num_classes)
    model = CzechRailwayLightNet(config)

    # Load the state dict
    model.load_state_dict(torch.load(filepath))

    return model


def main():
    data_dir = '../../reconstructed/czech_railway_lights_dataset_extended_roi'
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist!")

    dataset = load_data(data_dir)

    dataset_dist = {}
    for i in os.listdir(data_dir):
        if os.path.isdir(data_dir + "/" + i):
            dataset_dist[i] = len(list(os.listdir(data_dir + "/" + i)))

    print(f"Number of classes in dataset: {len(dataset.classes)}")

    val_size = 0.15
    train_dataset, val_dataset = split_data(dataset, val_size=val_size)
    train_loader, val_loader = create_dataloader(train_dataset, val_dataset)

    num_classes = len(dataset.classes)
    model_name = "google/efficientnet-b0"
    model = load_model(num_classes, model_name=model_name)

    training_args = get_training_args()

    results = train_model(train_loader, val_loader, model, training_args)

    # Save the model to a PyTorch .pt file
    save_model_to_pt(model, filepath='czech_railway_lights_model.pt')

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
