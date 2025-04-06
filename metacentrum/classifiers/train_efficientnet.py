import json
import os
import torch
from sympy.codegen.ast import continue_
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModelForImageClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from PIL import Image

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)

    # Calculate precision, recall, and F1-score
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Custom Dataset Wrapper for Hugging Face Trainer
class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = ImageFolder(root=data_dir)  # Use ImageFolder to load data
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


# Step 1: Load and Preprocess the Dataset
def load_data(data_dir):
    # Define image transformations for preprocessing (resize, normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # EfficientNet typically uses 224x224 input images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create custom dataset
    dataset = CustomImageDataset(data_dir=data_dir, transform=transform)
    return dataset


# Step 2: Split Data into Train and Validation Sets
def split_data(dataset, val_size=0.2):
    # Splitting dataset into train and validation sets
    train_size = int((1 - val_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset


# Step 3: Prepare DataLoaders for Training and Validation
def create_dataloader(train_dataset, val_dataset, batch_size=32):
    # Creating DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# Step 4: Load Pre-trained EfficientNet Model
def load_model(num_classes, model_name):
    # Load the pre-trained EfficientNet model from Hugging Face hub ("google/efficientnet-b0")
    model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=num_classes,
                                                            ignore_mismatched_sizes=True)

    # Manually modify the final layer to have the correct number of output classes (3)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)  # Update the classifier layer

    return model


# Step 5: Define Training Arguments
def get_training_args(output_dir="./results"):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=30,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
    )


# Step 6: Train the Model Using Hugging Face Trainer
def train_model(train_loader, val_loader, model, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        compute_metrics=compute_metrics,
        data_collator=None,

    )

    # Start training
    trainer.train()

    # Evaluate the fine-tuned model
    return trainer.evaluate()


# Step 7: Main Function to Execute the Training
def main():
    # Define the path to your dataset (root directory where your class folders reside)
    data_dir = '../../reconstructed/czech_railway_lights_dataset_extended_roi'  # Ensure this points to the correct directory

    # Ensure that the data directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist!")

    # Load dataset
    dataset = load_data(data_dir)

    dataset_dist = {}
    for i in os.listdir(data_dir):
        if os.path.isdir(data_dir + "/" + i):
            dataset_dist[i] = len(list(os.listdir(data_dir + "/" + i)))


    # Print number of classes for verification
    print(f"Number of classes in dataset: {len(dataset.classes)}")  # Should print 3 classes

    # Split into training and validation datasets
    val_size = 0.5
    train_dataset, val_dataset = split_data(dataset,val_size=val_size)

    # Create DataLoader for training and validation sets
    train_loader, val_loader = create_dataloader(train_dataset, val_dataset)

    # Load EfficientNet model with the number of classes based on the dataset
    num_classes = len(dataset.classes)
    model_name="google/efficientnet-b0"# Automatically detects 3 classes
    model = load_model(num_classes,model_name=model_name)

    # Set up training arguments
    training_args = get_training_args()

    # Train the model
    results = train_model(train_loader, val_loader, model, training_args)
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({
        "dataset_classes": dataset_dist,
            "dataset_val_size": val_size,
        "model_name": model_name,
        "results": results,
        "training_args": {"learning_rate": training_args.learning_rate,
                            "num_train_epochs": training_args.num_train_epochs,
                            "weight_decay": training_args.weight_decay}},f, indent=2, ensure_ascii=True)



# Step 8: Run the Script
if __name__ == "__main__":
    main()
