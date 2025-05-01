import os
import glob
import shutil
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import albumentations


class YOLODatasetAnalyzer:
    def __init__(self, images_dir, labels_dir):
        """
        Initialize with paths to images and labels directories

        Args:
            images_dir (str): Path to directory containing images
            labels_dir (str): Path to directory containing YOLO label files
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_counts = None
        self.minority_classes = None
        self.class_to_images = {}

    def analyze_dataset(self):
        """
        Analyze the dataset to count instances of each class and 
        identify which images contain which classes

        Returns:
            dict: Counter object with class counts
        """
        label_files = glob.glob(os.path.join(self.labels_dir, "*.txt"))
        all_classes = []

        print(f"Analyzing {len(label_files)} label files...")

        for label_file in tqdm(label_files):
            base_name = os.path.basename(label_file)
            image_name = os.path.splitext(base_name)[0]

            with open(label_file, 'r') as f:
                lines = f.readlines()

            # Extract class labels from each label file
            classes_in_image = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # Ensure valid YOLO format
                    class_id = int(parts[0])
                    classes_in_image.append(class_id)
                    all_classes.append(class_id)

            # Record which classes appear in this image
            for class_id in set(classes_in_image):
                if class_id not in self.class_to_images:
                    self.class_to_images[class_id] = []
                self.class_to_images[class_id].append(os.path.join(self.images_dir, f"{image_name}.jpg"))

        self.class_counts = Counter(all_classes)
        return self.class_counts

    def plot_class_distribution(self):
        """Plot the distribution of classes in the dataset"""
        if self.class_counts is None:
            self.analyze_dataset()

        classes = sorted(self.class_counts.keys())
        counts = [self.class_counts[c] for c in classes]

        plt.figure(figsize=(12, 6))
        plt.bar(classes, counts)
        plt.xlabel('Class ID')
        plt.ylabel('Number of instances')
        plt.title('Class Distribution in Dataset')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(classes)
        plt.show()

        return classes, counts

    def identify_minority_classes(self, threshold_percentage=20):
        """
        Identify minority classes based on a threshold percentage of the most frequent class

        Args:
            threshold_percentage (int): Classes with count below this percentage of the most 
                                     frequent class are considered minority classes

        Returns:
            list: List of class IDs identified as minority classes
        """
        if self.class_counts is None:
            self.analyze_dataset()

        max_count = max(self.class_counts.values())
        threshold = max_count * (threshold_percentage / 100)

        self.minority_classes = [class_id for class_id, count in self.class_counts.items()
                                 if count < threshold]

        print(f"Identified {len(self.minority_classes)} minority classes: {self.minority_classes}")
        print(f"These classes have fewer than {threshold_percentage}% of the instances of the most frequent class")

        return self.minority_classes

    def get_images_with_minority_classes(self, minority_classes=None):
        """
        Get paths to images containing specified minority classes

        Args:
            minority_classes (list, optional): List of class IDs to consider as minority classes.
                                            If None, uses previously identified minority classes.

        Returns:
            dict: Dictionary mapping class IDs to lists of image paths
            list: List of all unique image paths containing any minority class
        """
        if minority_classes is None:
            if self.minority_classes is None:
                self.identify_minority_classes()
            minority_classes = self.minority_classes

        # Filter the class_to_images dict to only include minority classes
        minority_class_images = {class_id: self.class_to_images.get(class_id, [])
                                 for class_id in minority_classes}

        # Get all unique images containing any minority class
        all_images = set()
        for image_list in minority_class_images.values():
            all_images.update(image_list)

        print(f"Found {len(all_images)} images containing minority classes")
        return minority_class_images, list(all_images)


class YOLOAugmenter:
    def __init__(self, images_dir, labels_dir, output_dir, class_map=None):
        """
        Initialize the augmenter

        Args:
            images_dir (str): Path to directory containing original images
            labels_dir (str): Path to directory containing original label files
            output_dir (str): Path to directory where augmented data will be saved
            class_map (dict, optional): Dictionary mapping class IDs to class names
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.output_dir = output_dir
        self.class_map = class_map

        # Create output directories if they don't exist
        self.output_images_dir = os.path.join(output_dir, 'images')
        self.output_labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(self.output_images_dir, exist_ok=True)
        os.makedirs(self.output_labels_dir, exist_ok=True)

        # Define augmentations
        self.augmentations = [
            albumentations.Compose([
                albumentations.HorizontalFlip(p=1.0),
            ], bbox_params=albumentations.BboxParams(format='yolo', label_fields=['class_labels'])),

            albumentations.Compose([
                albumentations.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            ], bbox_params=albumentations.BboxParams(format='yolo', label_fields=['class_labels'])),

            albumentations.Compose([
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightnessContrast(p=0.5),
            ], bbox_params=albumentations.BboxParams(format='yolo', label_fields=['class_labels'])),
        ]

    def augment_images(self, image_paths, augmentations_per_image=3):
        """
        Apply augmentations to the specified images

        Args:
            image_paths (list): List of paths to images to augment
            augmentations_per_image (int): Number of augmentations to apply to each image
        """
        print(f"Augmenting {len(image_paths)} images...")

        for img_path in tqdm(image_paths):
            base_name = os.path.basename(img_path)
            base_name_no_ext = os.path.splitext(base_name)[0]

            # Read the image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read the corresponding label file
            label_path = os.path.join(self.labels_dir, f"{base_name_no_ext}.txt")
            if not os.path.exists(label_path):
                print(f"Warning: No label file found for {img_path}")
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()

            # Parse YOLO format labels
            bboxes = []
            class_labels = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Convert class_id to int, handling potential float values
                    class_id = int(float(parts[0]))
                    # Parse coordinates as floats
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)

            # Copy original image and label to output directory
            shutil.copy(img_path, os.path.join(self.output_images_dir, base_name))
            shutil.copy(label_path, os.path.join(self.output_labels_dir, f"{base_name_no_ext}.txt"))

            # Apply augmentations
            for i in range(augmentations_per_image):
                # Randomly select an augmentation
                aug = random.choice(self.augmentations)

                # Apply augmentation
                try:
                    augmented = aug(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_labels = augmented['class_labels']

                    # Save augmented image
                    aug_image_filename = f"{base_name_no_ext}_aug{i}.jpg"
                    aug_label_filename = f"{base_name_no_ext}_aug{i}.txt"

                    # Convert back to BGR for OpenCV saving
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(self.output_images_dir, aug_image_filename), aug_image_bgr)

                    # Save augmented labels
                    with open(os.path.join(self.output_labels_dir, aug_label_filename), 'w') as f:
                        for box, cls_id in zip(aug_bboxes, aug_class_labels):
                            # YOLO format: class_id x_center y_center width height
                            # Format with 6 decimal places for consistency
                            f.write(f"{int(cls_id)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
                except Exception as e:
                    print(f"Error augmenting {img_path}: {e}")
                    continue

    def display_sample_augmentations(self, image_path, num_samples=5):
        """
        Display sample augmentations for a single image

        Args:
            image_path (str): Path to the image to augment
            num_samples (int): Number of sample augmentations to display
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the corresponding label file
        base_name = os.path.basename(image_path)
        base_name_no_ext = os.path.splitext(base_name)[0]
        label_path = os.path.join(self.labels_dir, f"{base_name_no_ext}.txt")

        if not os.path.exists(label_path):
            print(f"No label file found for {image_path}")
            return

        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Parse YOLO format labels
        bboxes = []
        class_labels = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)

        # Display original image with bounding boxes
        plt.figure(figsize=(20, 4))
        plt.subplot(1, num_samples + 1, 1)
        plt.imshow(image)
        plt.title('Original')

        self._draw_bboxes(image, bboxes, class_labels)

        # Apply and display augmentations
        for i in range(num_samples):
            aug = random.choice(self.augmentations)

            try:
                augmented = aug(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_labels = augmented['class_labels']

                plt.subplot(1, num_samples + 1, i + 2)
                plt.imshow(aug_image)
                plt.title(f'Aug {i + 1}')

                self._draw_bboxes(aug_image, aug_bboxes, aug_class_labels)
            except Exception as e:
                print(f"Error applying augmentation: {e}")

        plt.tight_layout()
        plt.show()

    def _draw_bboxes(self, image, bboxes, class_labels):
        """Helper method to draw bounding boxes on the image"""
        h, w = image.shape[:2]

        for (x_center, y_center, width, height), cls_id in zip(bboxes, class_labels):
            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # Draw rectangle
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw class label
            label = f"Class {cls_id}"
            if self.class_map and cls_id in self.class_map:
                label = self.class_map[cls_id]

            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    # Example usage
    IMAGES_DIR = "../../reconstructed/czech_railway_light_dataset/train_a/images/multi_class"
    LABELS_DIR = "../../reconstructed/czech_railway_light_dataset/train_a/labels/multi_class"
    OUTPUT_DIR = "../../reconstructed/czech_railway_light_dataset/output"

    # Step 1: Analyze the dataset
    analyzer = YOLODatasetAnalyzer(IMAGES_DIR, LABELS_DIR)
    analyzer.analyze_dataset()

    # Step 2: Plot class distribution (optional)
    analyzer.plot_class_distribution()

    # Step 3: Identify minority classes (classes with fewer than 20% of the instances of the most frequent class)
    minority_classes = analyzer.identify_minority_classes(threshold_percentage=20)

    # Step 4: Get images containing minority classes
    _, minority_images = analyzer.get_images_with_minority_classes(minority_classes)

    # Step 5: Augment images with minority classes
    augmenter = YOLOAugmenter(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR)

    # Optional: Display sample augmentations for a single image
    if minority_images:
        augmenter.display_sample_augmentations(minority_images[0])

    # Augment all images containing minority classes
    augmenter.augment_images(minority_images, augmentations_per_image=3)

    print(f"Augmentation complete. Augmented dataset saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()