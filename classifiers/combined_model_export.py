import cv2
import numpy as np
import yaml
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes


class CzechRailwayLightModel(torch.nn.Module):
    def __init__(self, yolo_path="./czech_railway_light_detection_backbone/detection_backbone/weights/best.torchscript",
                 classifier_path="./czech_railway_lights_model.pt",
                 labels_path="../metacentrum_experiments/CRL_single_images_less_balanced.yaml"):
        super(CzechRailwayLightModel, self).__init__()

        # Load YOLO model
        self.yolo_model = torch.jit.load(yolo_path)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Load classifier head
        self.classifier = torch.load(classifier_path, device).to(device)
        self.classifier.eval()  # Set to evaluation mode

        # Load labels
        with open(labels_path) as f:
            self.names = yaml.load(f, yaml.SafeLoader)["names"]

    def preprocess_crop(self, crop):
        """Preprocess cropped image for classification"""
        # Ensure crop is a valid image
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return None

        # Resize the image to (34, 34)
        crop_resized = cv2.resize(crop, (34, 34))

        # Convert to float and normalize to [0, 1]
        img_float = crop_resized.astype(np.float32) / 255.0

        # Change shape to (C, H, W)
        img_transposed = np.transpose(img_float, (2, 0, 1))

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        img_normalized = (img_transposed - mean) / std

        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img_normalized).float()

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def forward(self, images, conf=0.5, iou=0.5, verbose=False):
        """
        Forward pass for detection and classification
        Args:
            images: Input images (either as numpy array or PyTorch tensor)
            conf: Confidence threshold for YOLO detections
            iou: IoU threshold for YOLO detections
            verbose: Whether to print detection info
        """
        # Ensure images is on the same device as the model
        device = next(self.parameters()).device

        # Convert to tensor if needed
        if not isinstance(images, torch.Tensor):
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float()
            else:
                raise TypeError(f"Expected numpy array or torch tensor, got {type(images)}")

        # Move to device
        images = images.to(device)

        # # Add batch dimension if needed
        # if len(images.shape) == 3:
        #     images = images.unsqueeze(0).reshape(1, 3, 1080, 1920)

        # Run YOLO detection
        with torch.no_grad():
            detections = self.yolo_model(images)

        results = []

        # Process each image in the batch
        for i, image_detections in enumerate(detections):
            image = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format

            # Skip if no detections
            if len(image_detections) == 0:
                results.append(image_detections)  # Return empty detections
                continue

            updated_boxes = []

            # Process each detection
            for detection in image_detections:
                boxes = detection.boxes

                # Process each bounding box
                for j in range(len(boxes.xyxy)):
                    x1, y1, x2, y2 = map(int, boxes.xyxy[j])

                    # Handle out-of-bounds coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Crop the detected object
                    crop = image[y1:y2, x1:x2]

                    # Preprocess crop for classification
                    img_tensor = self.preprocess_crop(crop)
                    if img_tensor is None:
                        continue

                    # Run classification
                    with torch.no_grad():
                        classifier_output = self.classifier(img_tensor.to(device))

                    # Get prediction
                    logits = classifier_output['logits']
                    predicted_class_idx = torch.argmax(logits, dim=1).item()
                    confidence = torch.softmax(logits, dim=1)[0, predicted_class_idx].item()

                    # Create new box data with updated class
                    box_data = boxes.data[j].clone()
                    box_data[5] = predicted_class_idx  # Update class index
                    updated_boxes.append(box_data)

            # Create new Results object with updated boxes
            if updated_boxes:
                new_boxes = Boxes(
                    torch.stack(updated_boxes),
                    orig_shape=image_detections[0].boxes.orig_shape
                )

                # Create a new Results object
                updated_result = Results(
                    orig_img=image_detections[0].orig_img,
                    path=image_detections[0].path,
                    names=self.names,
                    boxes=new_boxes
                )

                results.append([updated_result])
            else:
                results.append([])  # No valid detections after processing

        return results

    def export(self, filepath="combined_model.pt"):
        """Export the combined model to a single file"""
        torch.save(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath="combined_model.pt"):
        """Load the combined model from a file"""
        model = torch.load(filepath)
        model.eval()
        return model



# Create and use the model
model = CzechRailwayLightModel()

# For inference
image = cv2.imread("29.733_clean.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
results = model(image)

# Export the model
model.export("czech_railway_combined.pt")

# Later, load the model
loaded_model = CzechRailwayLightModel.load("czech_railway_combined.pt")