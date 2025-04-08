import os
import cv2
from tqdm import tqdm
import argparse



def extract_rois_from_yolo(input_dir, output_dir):
    """
    Extract ROIs from YOLO format dataset and organize them into a class-based folder structure
    for use with ImageFolder.

    Args:
        input_dir: Path to the YOLO dataset directory containing 'images' and 'labels' folders
        output_dir: Path to save the extracted ROIs in class-based folder structure
    """
    # Get paths
    images_dir = os.path.join(input_dir,)
    labels_dir = os.path.join(input_dir.replace("images", "labels"))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to keep track of extracted ROIs per class
    class_counters = {}

    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Processing {len(image_files)} images...")

    for img_file in tqdm(image_files):
        # Get corresponding label file (same name but .txt extension)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # Skip if label file doesn't exist
        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_file}")
            continue

        # Read the image
        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read image {img_file}")
            continue

        # Get image dimensions
        img_height, img_width = image.shape[:2]

        # Read labels
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Process each label (ROI) in the file
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"Warning: Invalid label format in {label_file}, line {i + 1}")
                continue

            class_id = parts[0]

            # Create class directory if it doesn't exist
            class_dir = os.path.join(output_dir, f"class_{class_id}")
            os.makedirs(class_dir, exist_ok=True)

            # Initialize counter for this class if not already done
            if class_id not in class_counters:
                class_counters[class_id] = 0

            # Convert YOLO format (center_x, center_y, width, height) to bounding box coordinates
            # YOLO format values are normalized (0-1)
            center_x = float(parts[1]) * img_width
            center_y = float(parts[2]) * img_height
            bbox_width = float(parts[3]) * img_width
            bbox_height = float(parts[4]) * img_height

            # Calculate bounding box coordinates
            left = int(max(0, center_x - bbox_width / 2))
            top = int(max(0, center_y - bbox_height / 2))
            right = int(min(img_width, center_x + bbox_width / 2))
            bottom = int(min(img_height, center_y + bbox_height / 2))

            # Extract ROI
            roi = image[top:bottom, left:right]
            if roi.size == 0:
                print(f"Warning: Empty ROI extracted from {img_file} with coordinates: {left},{top},{right},{bottom}")
                continue

            # Save the ROI
            class_counters[class_id] += 1
            roi_filename = f"{os.path.splitext(img_file)[0]}_roi_{i}.jpg"
            roi_path = os.path.join(class_dir, roi_filename)
            cv2.imwrite(roi_path, roi)

    print(f"Extraction complete! Summary of extracted ROIs:")
    for class_id, count in class_counters.items():
        print(f"Class {class_id}: {count} ROIs")


def main():
    parser = argparse.ArgumentParser(description='Extract ROIs from YOLO format dataset')
    parser.add_argument('--input',
                        default="../../reconstructed/CRL_extended/train/images/multi_class",
                        help='Input directory with YOLO format dataset (containing images and labels folders)')
    parser.add_argument('--output',
                        default="../../reconstructed/czech_railway_lights_dataset_extended_roi/",
                        help='Output directory for extracted ROIs')
    args = parser.parse_args()

    extract_rois_from_yolo(args.input, args.output)

    print("\nDataset is now ready for use with CustomImageDataset:")
    print(f"dataset = CustomImageDataset(data_dir='{args.output}')")


if __name__ == "__main__":
    main()