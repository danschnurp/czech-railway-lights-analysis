import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import argmax, cat, from_numpy, float32
from ultralytics import YOLO
from ultralytics.engine.results import Boxes


def create_detection_nett():
    # Load YOLOv5nu model and weights
    yolov5nu_model = YOLO("./30_lights_0_yolov5nu/weights/weights/best.pt")

    yolov5nu_model.train(data="../metacentrum_experiments/CRL_single_images_less_balanced.yaml",
                         project="./czech_railway_light_detection_backbone",
                         name="detection_backbone",
                         epochs=1)


def opencv_transforms(image):
    # Read the image using OpenCV
    # Convert BGR to RGb (OpenCV loads images in BGR format)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to (34, 34)
    image = cv2.resize(image, (34, 34))

    # Convert the image to a tensor (normalize to [0, 1] and change shape)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change shape to (C, H, W)

    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    image = (image - mean) / std

    # Convert to a PyTorch tensor
    image = from_numpy(image)
    image = image.type(float32)
    return image


class CzechRailwayLightModel(nn.Module):

    def __init__(self, detection_nett_path="./czech_railway_light_detection_backbone/detection_backbone/weights/best.pt",
                 classification_nett_path="./czech_railway_lights_model.pt"):
        super(CzechRailwayLightModel, self).__init__()
        print("loading detection nett")
        yolov5nu_model = YOLO(detection_nett_path)
        print("loading classification nett")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # Load classifier head
        czech_railway_head = torch.load(
            classification_nett_path, device, weights_only=False)
        yolov5nu_model.to(device)
        self.names = {0: 'stop', 1: 'go', 2: 'warning', 3: 'adjust speed and warning', 4: 'adjust speed and go', 5: 'lights off'}
        self.yolov5nu_model = yolov5nu_model
        self.czech_railway_head = czech_railway_head

    def forward(self, x, conf=0.5, iou=0.5, verbose=True):
        features = self.yolov5nu_model(x, conf=conf, iou=iou, verbose=verbose)

        item_idx = None
        predicted_classes = []
        for feature_idx, feature in enumerate(features):
            for item_idx, feature_data in enumerate(feature):
                new_datas = []
                for box in feature_data.boxes:
                    x_min, y_min, width, height = map(int, box.xyxy[0])
                    crop = x[y_min:height, x_min:width]

                    image_tensor = opencv_transforms(crop).unsqueeze(1).reshape(1, 3, 34, 34)  # Add batch dimension
                    output = self.czech_railway_head(image_tensor)


                    # Get prediction
                    logits = output['logits']
                    predicted_classes.append(argmax(logits, dim=1).item())

        return features, predicted_classes

    def predict(self, x, conf=0.5, iou=0.5, verbose=True):
        return self.forward(x, conf, iou, verbose)
