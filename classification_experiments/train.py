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

from czech_railway_lights_nett import CzechRailwayLightNet


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
        transforms.Resize((74, 34)), # Resize to the small image size
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

def get_training_args(output_dir="./results"):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=30,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=0.001,
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
    model_name = "google/efficientnet-b0"
    model = load_model(num_classes, model_name=model_name)

    training_args = get_training_args()

    results = train_model(train_loader, val_loader, model, training_args)

    # Save the model to a PyTorch .pt file
    save_model_to_pt(model, filepath='czech_railway_lights_nett.pt')

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
