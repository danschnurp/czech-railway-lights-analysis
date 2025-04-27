import yaml
from numpy import array
import torch.nn as nn
from torch import Tensor, no_grad, argmax, load, save, cat
from torch.xpu import device
from transformers import AutoConfig
from ultralytics import YOLO as YOLOv5nu  # Assuming you have a YOLOv5nu model implementation
from torchvision import transforms
from PIL import Image
from copy import deepcopy

from classifiers.crl_model import CzechRailwayLightNet


class CombinedModel(nn.Module):
    def __init__(self, yolov5nu_model, czech_railway_head):
        super(CombinedModel, self).__init__()
        self.names = idx_name_dict
        self.backbone = yolov5nu_model
        self.head = czech_railway_head
        self.transform = transforms.Compose([
                    transforms.Resize((16, 34)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    def forward(self, x, conf=0.75, iou=0.45):
        features = self.backbone(x, conf=conf, iou=iou)
        new_datas = []
        item_idx = None
        for feature_idx, feature in enumerate(features):
            for item_idx, i in enumerate(feature):
                x_min, y_min, width, height = map(int, i.boxes[0].xyxy[0])
                crop = x[y_min:height, x_min:width]

                image_tensor = self.transform(Image.fromarray(crop)).unsqueeze(0)  # Add batch dimension
                # Run inference
                with no_grad():
                    output = self.head(image_tensor)

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
# Load YOLOv5nu model and weights
yolov5nu_model = YOLOv5nu("/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection/metacentrum_experiments/30_lights_0_yolov5nu.pt_0.5/weights/CRL_detectron.pt")


# yolov5nu_model = YOLOv5nu("/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection/metacentrum_experiments/30_lights_0_yolov5nu.pt_0.5/weights/best.pt")
# yolov5nu_model.train(data="../metacentrum_experiments/CRL_single_images_less_balanced.yaml", epochs=1)
# yolov5nu_model.eval()
# save(yolov5nu_model, "/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection/metacentrum_experiments/30_lights_0_yolov5nu.pt_0.5/weights/CRL_detectron.pt")
# exit(0)

with open("/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection/metacentrum_experiments/CRL_single_images_less_balanced.yaml") as f:
    idx_name_dict = yaml.load(f, yaml.SafeLoader)["names"]


    # Load the state dict
czech_railway_head =  load("/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection/metacentrum_experiments/classifiers/czech_railway_lights_model.pt")
czech_railway_head.cpu()


# Initialize the combined model
combined_model = CombinedModel(yolov5nu_model, czech_railway_head)



# Fine-tune the combined model
# Define your dataset, loss function, optimizer, and training loop here
