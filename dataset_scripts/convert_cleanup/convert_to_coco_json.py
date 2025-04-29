import os
import json
from collections import defaultdict

def convert_to_coco(dataset_path, output_path):
    # Initialize COCO format structure
    coco_format = {
        "info": {
            "description": "Converted Dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "",
            "date_created": "2023-10-01"
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # Initialize category IDs
    category_ids = {}

    # Initialize image and annotation IDs
    image_id = 1
    annotation_id = 1

    # Iterate over train and val directories
    for split in ['train', 'val']:
        images_path = os.path.join(dataset_path, split, 'images', 'multi_class')
        labels_path = os.path.join(dataset_path, split, 'labels', 'multi_class')

        # Get list of image files
        image_files = sorted(os.listdir(images_path))

        for image_file in image_files:
            # Read image information
            image_info = {
                "id": image_id,
                "file_name": image_file,
                "height": 1080,  # Placeholder height
                "width": 1920,   # Placeholder width
                "date_captured": "2023-10-01"
            }
            coco_format["images"].append(image_info)

            # Read corresponding label file
            label_file = os.path.join(labels_path, image_file.replace('.jpg', '.txt'))
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        if line == "\n":
                            continue
                        # Parse YOLO format: class x_center y_center width height
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())

                        # Convert YOLO format to COCO format
                        x = x_center * 1920 - (width * 1920) / 2  # Placeholder width
                        y = y_center * 1080 - (height * 1080) / 2  # Placeholder height
                        bbox_width = width * 1920
                        bbox_height = height * 1080

                        # Add category if not already present
                        if int(class_id) not in category_ids:
                            category_ids[int(class_id)] = len(category_ids) + 1
                            coco_format["categories"].append({
                                "id": category_ids[int(class_id)],
                                "name": f"class_{int(class_id)}",
                                "supercategory": "object"
                            })

                        # Create annotation
                        annotation_info = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_ids[int(class_id)],
                            "bbox": [x, y, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "segmentation": [],
                            "iscrowd": 0
                        }
                        coco_format["annotations"].append(annotation_info)
                        annotation_id += 1

            image_id += 1

    # Save COCO format to JSON file
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

# Example usage
dataset_path = './czech_railway_lights_dataset_4_classes'
output_path = './czech_railway_lights_dataset_4_classes.json'
convert_to_coco(dataset_path, output_path)
