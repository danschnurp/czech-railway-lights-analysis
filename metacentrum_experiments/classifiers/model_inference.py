import torch
from torchvision import transforms
from transformers import AutoConfig

from classifiers.crl_model import CzechRailwayLightNet


# Simple example of how to use the loaded model for inference
def model_inference(model_path, image, class_mapping):
    """
    Example function to demonstrate model inference

    Args:
        model_path: Path to the saved .pt model
        image_path: Path to an image file for inference
        class_mapping: Dictionary mapping class indices to class names
    """

    # Load the model
    num_classes = len(class_mapping)
    # Create a new model instance with the same architecture
    config = AutoConfig.from_pretrained('google/efficientnet-b0', num_labels=num_classes)
    model = CzechRailwayLightNet(config)

    # Load the state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((16, 34)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)

    # Get prediction
    logits = outputs['logits']
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    predicted_class = class_mapping[str(predicted_class_idx)]

    print(f"Predicted class: {predicted_class}")

    return predicted_class