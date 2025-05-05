import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import argmax, from_numpy, float32
from torchvision.transforms import transforms
from ultralytics import YOLO


def create_detection_net():
    """Initialize and train the YOLOv5nu detection model"""
    # Load YOLOv5nu model and weights
    yolov5nu_model = YOLO("./30_lights_0_yolov5nu/weights/weights/best.pt")

    yolov5nu_model.train(data="../metacentrum_experiments/CRL_single_images_less_balanced.yaml",
                         project="./czech_railway_light_detection_backbone",
                         name="detection_backbone",
                         epochs=1)

    return yolov5nu_model





class CzechRailwayLightModel(nn.Module):
    """Combined model for Czech railway light detection and classification"""

    def __init__(self, detection_net_path="./classification_experiments/czech_railway_light_detection_backbone/detection_backbone/weights/best.pt",
                 classification_net_path="./czech_railway_lights_net.pt"):
        super(CzechRailwayLightModel, self).__init__()

        print("Loading detection network...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                  #  "mps" if torch.backends.mps.is_available() else
                                   "cpu")

        self.yolo_model = YOLO(detection_net_path)
        self.yolo_model.to(self.device)

        print("Loading classification network...")
        self.classification_head = torch.load(
            classification_net_path, map_location=self.device, weights_only=False)

        self.names = {
            0: 'stop',
            1: 'go',
            2: 'warning',
            3: 'adjust speed and warning',
            4: 'adjust speed and go',
            5: 'lights off'
        }
        self.transform = transforms.Compose([
        transforms.Resize((72, 34)), # Resize to the small image size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def forward(self, image, conf=0.5, iou=0.5, verbose=False):
        """
        Process image through detection and classification pipeline

        Args:
            image: Input image (OpenCV format, BGR)
            conf: Confidence threshold for detection
            iou: IoU threshold for NMS
            verbose: Whether to print detection details

        Returns:
            detections: Raw detection results
            classifications: List of classified light states
            results: Dictionary with full results including bounding boxes and class names
        """
        # Run detection
        detections = self.yolo_model(image, conf=conf, iou=iou, verbose=verbose)

        classifications = []
        results = {
            'boxes': [],
            'class_ids': [],
            'class_names': [],
            'confidences': []
        }

        # Process each detection
        for detection in detections:
            boxes = detection.boxes
            if len(boxes) == 0:
                continue

            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # Ensure valid crop dimensions
                if x1 >= x2 or y1 >= y2:
                    continue
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 > image.shape[1]: x2 = image.shape[1]
                if y2 > image.shape[0]: y2 = image.shape[0]

                # Extract the crop
                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue
                crop_pil = Image.fromarray(crop)
                # Transform and classify
                image_tensor = self.transform(crop_pil).unsqueeze(0)  # Add batch dimension
                image_tensor = image_tensor.to(self.device)

                with torch.no_grad():
                    output = self.classification_head(image_tensor)

                # Get prediction
                logits = output['logits']
                class_id = argmax(logits, dim=1).item()

                classifications.append(class_id)

                # Store results
                results['boxes'].append([x1, y1, x2, y2])
                results['class_ids'].append(class_id)
                results['class_names'].append(self.names[class_id])
                results['confidences'].append(float(box.conf.cpu().numpy()[0]))




        return detections, classifications, results

    def predict(self, image, conf=0.5, iou=0.5, verbose=False):
        """Alias for forward method"""
        return self.forward(image, conf, iou, verbose)

    def visualize_results(self, image, results):
        """
        Draw detection and classification results on the image

        Args:
            image: Original image
            results: Results dictionary from forward method

        Returns:
            Annotated image
        """
        img_copy = image.copy()

        for i, (box, class_name, conf) in enumerate(zip(
                results['boxes'], results['class_names'], results['confidences'])):
            x1, y1, x2, y2 = box

            # Define color based on class (different color for each class)
            class_id = results['class_ids'][i]
            colors = [
                (0, 0, 255),  # stop - red
                (0, 255, 0),  # go - green
                (0, 255, 255),  # warning - yellow
                (255, 0, 255),  # adjust speed and warning - magenta
                (255, 255, 0),  # adjust speed and go - cyan
                (128, 128, 128)  # lights off - gray
            ]
            color = colors[class_id % len(colors)]

            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            text = f"{class_name}: {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

            # Draw text
            cv2.putText(img_copy, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img_copy


# Usage example
if __name__ == "__main__":
    # Initialize model
    model = CzechRailwayLightModel()

    # Load test image
    image_path = "test_image.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
    else:
        # Process image
        detections, classifications, results = model.predict(image, conf=0.5, verbose=True)

        # Visualize results
        output_image = model.visualize_results(image, results)

        # Display results
        print(f"Found {len(classifications)} railway lights:")
        for i, class_id in enumerate(results['class_ids']):
            print(f"  - Light {i + 1}: {results['class_names'][i]} (confidence: {results['confidences'][i]:.2f})")

        # Save output image
        cv2.imwrite("output_image.jpg", output_image)
        print(f"Output image saved to output_image.jpg")