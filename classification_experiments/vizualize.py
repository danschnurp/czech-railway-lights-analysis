import json
import os

import torch
import torch.nn as nn
import yaml
from torchviz import make_dot
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import networkx as nx
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig

from czech_railway_lights_nett import CzechRailwayLightNet


def print_model_summary(model_path, num_classes):
    """
    Print a summary of the model architecture using torchinfo

    Args:
        model_path: Path to the saved .pt model
        num_classes: Number of output classes
    """
    # Load the model
    config = AutoConfig.from_pretrained('google/efficientnet-b0', num_labels=num_classes)
    model = CzechRailwayLightNet(config)

    # Print model summary with input size
    model_summary = summary(model, input_size=(1, 3, 34, 34),
                            col_names=["input_size", "output_size", "num_params", "kernel_size"],
                            verbose=1)

    return model_summary




def visualize_feature_maps(model_path, image_path, num_classes):
    """
    Visualize feature maps of a specific image through the model

    Args:
        model_path: Path to the saved .pt model
        image_path: Path to an input image
        num_classes: Number of output classes
    """
    from PIL import Image
    import matplotlib.pyplot as plt

    # Load the model
    config = AutoConfig.from_pretrained('google/efficientnet-b0', num_labels=num_classes)
    model = CzechRailwayLightNet(config)
    model.eval()

    # Prepare hooks to capture feature maps
    feature_maps = []

    def hook_fn(module, input, output):
        feature_maps.append(output)

    # Register hooks for each conv layer
    for name, module in model.features.named_modules():
        if isinstance(module, nn.Conv2d):
            module.register_forward_hook(hook_fn)

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((34, 34)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        model(pixel_values=image_tensor)

    # Plot original image
    plt.figure(figsize=(15, 10))
    plt.subplot(1, len(feature_maps) + 1, 1)
    plt.imshow(np.array(image))
    plt.title("Original Image")
    plt.axis('off')

    # Plot feature maps
    for i, fmap in enumerate(feature_maps):

         # Take first image from batch and first few channels
        f_map = fmap[0].detach().cpu().numpy()
        num_channels = min(4, f_map.shape[0])  # Display up to 4 channels

        for j in range(num_channels):
            try:
                plt.subplot(num_channels, len(feature_maps) + 1, len(feature_maps) + 1 + i + j * (len(feature_maps) + 1))
                plt.imshow(f_map[j], cmap='viridis')
                plt.title(f"Layer {i + 1}, Channel {j + 1}")
                plt.axis('off')
            except ValueError:
                break
    plt.tight_layout()
    plt.savefig('feature_maps.png')
    plt.close()
    print("Feature maps saved to feature_maps.png")

def visualize_model_structure(model_path, num_classes, output_file='model_visualization.png'):
    """
    Visualize the model architecture using torchviz

    Args:
        model_path: Path to the saved .pt model
        num_classes: Number of output classes
        output_file: Path to save the visualization
    """
    # Load the model
    config = AutoConfig.from_pretrained('google/efficientnet-b0', num_labels=num_classes)
    model = CzechRailwayLightNet(config)

    model.eval()

    # Create a dummy input
    x = torch.randn(1, 3, 34, 34)

    # Generate model forward pass
    outputs = model(pixel_values=x)

    # Visualize computation graph
    dot = make_dot(outputs['logits'], params=dict(model.named_parameters()))
    dot.render(output_file.replace('.png', ''), format='png')
    print(f"Model visualization saved to {output_file}")

    return output_file



def plot_model_filters(model_path, num_classes):
    """
    Visualize the filters (kernels) of the first convolutional layer

    Args:
        model_path: Path to the saved .pt model
        num_classes: Number of output classes
    """
    # Load the model
    config = AutoConfig.from_pretrained('google/efficientnet-b0', num_labels=num_classes)
    model = CzechRailwayLightNet(config)

    # Get the first convolutional layer's weights
    first_conv_layer = None
    for module in model.features.modules():
        if isinstance(module, nn.Conv2d):
            first_conv_layer = module
            break

    if first_conv_layer is None:
        print("No convolutional layer found in the model")
        return

    # Get the weights and normalize them for visualization
    weights = first_conv_layer.weight.data.clone()
    weights = weights - weights.min()
    weights = weights / weights.max()

    # Plot filters
    plt.figure(figsize=(12, 8))
    num_filters = min(32, weights.shape[0])
    for i in range(num_filters):
        plt.subplot(4, 8, i + 1)

        # For RGB filters
        if weights.shape[1] == 3:
            # Transpose from [channels, height, width] to [height, width, channels]
            filter_img = weights[i].permute(1, 2, 0).cpu().numpy()
            plt.imshow(filter_img)
        else:
            # For grayscale filters
            plt.imshow(weights[i, 0].cpu().numpy(), cmap='gray')

        plt.axis('off')

    plt.tight_layout()
    plt.savefig('model_filters.png')
    plt.close()
    print("First layer filters saved to model_filters.png")


# Example usage
if __name__ == "__main__":
    model_path = 'czech_railway_lights_nett.pt'
    num_classes = 5  # Replace with your actual number of classes

    # Load class mapping
    with open("../metacentrum_experiments/CRL_single_images_less_balanced.yaml") as f:
        interesting_labels = set(list(yaml.load(f, yaml.SafeLoader)["names"].values()))

        num_classes = len(interesting_labels)

    # Choose visualization method:
    visualize_model_structure(model_path, num_classes)
    print_model_summary(model_path, num_classes)


    # For feature maps and filter visualization, you need an image
    image_path = '1131_roi_0.jpg'  # Replace with a path to a sample image
    if os.path.exists(image_path):
        visualize_feature_maps(model_path, image_path, num_classes)
        plot_model_filters(model_path, num_classes)