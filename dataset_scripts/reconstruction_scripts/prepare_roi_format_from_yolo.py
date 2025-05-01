import os
import cv2
import numpy as np
from pathlib import Path
import shutil


def extract_crops_by_class(images_dir, labels_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in image_files:
        # Get base filename without extension
        base_name = os.path.splitext(img_file)[0]

        # Find corresponding label file
        label_file = os.path.join(labels_dir, base_name + '.txt')
        if not os.path.exists(label_file):
            print(f"Warning: No label file found for {img_file}")
            continue

        # Load image
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_file}")
            continue

        height, width = img.shape[:2]

        # Read labels
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Process each object in the image
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            bbox_width = float(parts[3]) * width
            bbox_height = float(parts[4]) * height

            # Calculate bbox coordinates
            x1 = int(x_center - bbox_width / 2)
            y1 = int(y_center - bbox_height / 2)
            x2 = int(x_center + bbox_width / 2)
            y2 = int(y_center + bbox_height / 2)

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Crop image
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Create class directory if it doesn't exist
            class_dir = os.path.join(output_dir, f"{class_id}")
            os.makedirs(class_dir, exist_ok=True)

            # Save cropped image
            crop_filename = f"{base_name}_{i}.jpg"
            cv2.imwrite(os.path.join(class_dir, crop_filename), crop)

    print(f"Cropped images extracted and organized by class in {output_dir}")


IMAGES_DIR = "../../reconstructed/czech_railway_light_dataset/train/images/multi_class"
LABELS_DIR = "../../reconstructed/czech_railway_light_dataset/train/labels/multi_class"
output_dir = "../../reconstructed/czech_railway_light_dataset_roi"

extract_crops_by_class(IMAGES_DIR, LABELS_DIR, output_dir)