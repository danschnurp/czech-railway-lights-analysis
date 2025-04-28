import cv2
import numpy as np
import torch.nn as nn
import yaml
from torch import argmax, load, save, cat, from_numpy, float32
from ultralytics import YOLO


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
    def __init__(self):
        super(CzechRailwayLightModel, self).__init__()
        yolov5nu_model = YOLO(
            "./czech_railway_light_detection_backbone/detection_backbone/weights/best.pt")
        czech_railway_head = load(
            "./czech_railway_lights_model.pt", weights_only=False)
        czech_railway_head.cpu()
        with open("../../metacentrum_experiments/CRL_single_images_less_balanced.yaml") as f:
            interesting_labels = yaml.load(f, yaml.SafeLoader)["names"]
        self.names = interesting_labels
        self.yolov5nu_model = yolov5nu_model
        self.czech_railway_head = czech_railway_head


    def forward(self, x, conf=0.5, iou=0.5, verbose=True):
        features = self.yolov5nu_model(x, conf=conf, iou=iou, verbose=verbose)
        new_datas = []
        item_idx = None
        for feature_idx, feature in enumerate(features):
            for item_idx, i in enumerate(feature):
                x_min, y_min, width, height = map(int, i.boxes[0].xyxy[0])
                crop = x[y_min:height, x_min:width]

                image_tensor = opencv_transforms(crop).unsqueeze(1).reshape(1, 3, 34, 34)  # Add batch dimension
                output = self.czech_railway_head(image_tensor)

                # Get prediction
                logits = output['logits']
                predicted_class_idx = argmax(logits, dim=1).item()

                # Get original box data
                original_box = i.boxes[0]

                # Update the class value directly
                # First clone the original data to avoid modifying it in-place
                new_data = original_box.data.clone()
                # Update the class column (typically index 5 in YOLO format [x1, y1, x2, y2, conf, cls])
                new_data[:, 5] = predicted_class_idx
                new_datas.append(new_data)
        if item_idx is not None:
            # Create a new Boxes object with the updated data
            from ultralytics.engine.results import Boxes

            new_boxes = Boxes(
                        cat(new_datas),
                        orig_shape=original_box.orig_shape if hasattr(original_box, 'orig_shape') else None
                    )

            # Replace the original boxes with the updated ones
            features[feature_idx].boxes = new_boxes

        return features

    def predict(self, x):
        return self.forward(x)


def create_detection_nett():
    # Load YOLOv5nu model and weights
    yolov5nu_model = YOLO("./30_lights_0_yolov5nu/weights/weights/best.pt")

    yolov5nu_model.train(data="../../metacentrum_experiments/CRL_single_images_less_balanced.yaml",
                         project="./czech_railway_light_detection_backbone",
                         name="detection_backbone",
                         epochs=1)




def save_model():
    model = CzechRailwayLightModel()
    model.cpu()
    save(model, "../../dataset_reconstruction_scripts/test_and_preview/two_stage_czech_railway_lights_model.pt",
         )
# create_detection_nett()
save_model()
