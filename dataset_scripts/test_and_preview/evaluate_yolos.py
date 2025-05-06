import os
import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Czech Railway Light Detection and Classification with Ultralytics YOLO')
    parser.add_argument('--test-path', type=str,
                        default="../../reconstructed/czech_railway_lights_dataset_extended_TEST/val",
                        help='Path to test dataset (should contain images/ and labels/ folders)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold for detection')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IoU threshold for NMS')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--visualize', action='store_true', default=True, help='Visualize detections on images')
    parser.add_argument('--model', type=str, required=True, help='Path to Ultralytics YOLO model weights (.pt file)')
    return parser.parse_args()


def load_ground_truth_labels(label_path):
    """Load ground truth labels from a YOLO format txt file."""
    if not os.path.exists(label_path):
        return []

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) >= 5:  # class x_center y_center width height
                class_id = int(data[0])
                x_center = float(data[1])
                y_center = float(data[2])
                width = float(data[3])
                height = float(data[4])

                # Convert from YOLO format to [x1, y1, x2, y2, class_id, confidence]
                # Using 1.0 as confidence for ground truth
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                labels.append([x1, y1, x2, y2, class_id, 1.0])

    return labels


def compute_iou(box1, box2):
    """Compute IoU between box1 and box2."""
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def match_detections_to_ground_truth(detections, ground_truth, iou_threshold=0.5):
    """Match detected boxes to ground truth boxes based on IoU."""
    matches = []
    unmatched_gt = list(range(len(ground_truth)))

    for det_idx, detection in enumerate(detections):
        best_iou = iou_threshold
        best_gt_idx = -1

        for gt_idx in unmatched_gt:
            gt = ground_truth[gt_idx]
            iou = compute_iou(detection, gt)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            # We found a match
            matches.append((det_idx, best_gt_idx))
            unmatched_gt.remove(best_gt_idx)

    return matches, unmatched_gt


def calculate_metrics(all_detections, all_ground_truths, iou_threshold=0.5):
    """Calculate precision, recall, F1 score for object detection and classification."""
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives

    class_metrics = {}  # Will store per-class metrics

    # Create a dictionary to store class-specific counts
    class_tp = {}
    class_fp = {}
    class_fn = {}

    for detections, ground_truths in zip(all_detections, all_ground_truths):
        # Skip if no detections or ground truths
        if not detections and not ground_truths:
            continue

        # Match detections to ground truths
        matches, unmatched_gt = match_detections_to_ground_truth(detections, ground_truths, iou_threshold)

        # Count true positives (matched detections)
        tp = len(matches)
        total_tp += tp

        # Count false positives (unmatched detections)
        fp = len(detections) - tp
        total_fp += fp

        # Count false negatives (unmatched ground truths)
        fn = len(unmatched_gt)
        total_fn += fn

        # Update class-specific counts
        for det_idx, gt_idx in matches:
            det_class = int(detections[det_idx][4])
            gt_class = int(ground_truths[gt_idx][4])

            # Count as TP only if the class is correct
            if det_class == gt_class:
                class_tp[gt_class] = class_tp.get(gt_class, 0) + 1
            else:
                # Wrong class prediction counts as both FP and FN
                class_fp[det_class] = class_fp.get(det_class, 0) + 1
                class_fn[gt_class] = class_fn.get(gt_class, 0) + 1

        # Count unmatched detections as FP
        for det_idx in range(len(detections)):
            if not any(det_idx == m[0] for m in matches):
                det_class = int(detections[det_idx][4])
                class_fp[det_class] = class_fp.get(det_class, 0) + 1

        # Count unmatched ground truths as FN
        for gt_idx in unmatched_gt:
            gt_class = int(ground_truths[gt_idx][4])
            class_fn[gt_class] = class_fn.get(gt_class, 0) + 1

    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate per-class metrics
    all_classes = set(list(class_tp.keys()) + list(class_fp.keys()) + list(class_fn.keys()))
    for cls in all_classes:
        tp = class_tp.get(cls, 0)
        fp = class_fp.get(cls, 0)
        fn = class_fn.get(cls, 0)

        cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (
                                                                                          cls_precision + cls_recall) > 0 else 0

        class_metrics[cls] = {
            'precision': cls_precision,
            'recall': cls_recall,
            'f1': cls_f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    return {
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        },
        'per_class': class_metrics
    }


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_evaluation(args, model):
    """Run evaluation on test dataset with Ultralytics YOLO."""
    test_path = Path(args.test_path)
    images_dir = test_path / 'images' / 'multi_class'
    labels_dir = test_path / 'labels' / 'multi_class'

    if not images_dir.exists() or not labels_dir.exists():
        raise ValueError(
            f"Test dataset must contain 'images/multi_class' and 'labels/multi_class' directories at {test_path}")

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get list of image files
    image_files = sorted([f for f in images_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])

    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_dir}")

    print(f"Found {len(image_files)} images for evaluation")

    # Create a visualizations directory if needed
    if args.visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

    # Lists to store results
    all_detections = []
    all_ground_truths = []
    y_true = []  # For classification metrics
    y_pred = []  # For classification metrics

    # Process each image
    for img_path in tqdm(image_files, desc="Evaluating images"):
        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        img_height, img_width = frame.shape[:2]

        # Get corresponding label file path
        label_path = labels_dir / f"{img_path.stem}.txt"

        # Load ground truth labels
        gt_labels = load_ground_truth_labels(str(label_path))

        # Scale normalized coordinates to actual image size
        for i in range(len(gt_labels)):
            gt_labels[i][0] *= img_width
            gt_labels[i][1] *= img_height
            gt_labels[i][2] *= img_width
            gt_labels[i][3] *= img_height

        # Run detection with YOLO
        results = model(frame, conf=args.conf_thres, iou=args.iou_thres, verbose=False)

        # Process detection results
        detected_boxes = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                class_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                # Store detection as [x1, y1, x2, y2, class, confidence]
                detected_boxes.append([x1, y1, x2, y2, class_id, conf])

                # For classification metrics - find matching ground truth box
                if gt_labels:
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx, gt in enumerate(gt_labels):
                        iou = compute_iou([x1, y1, x2, y2], gt[:4])
                        if iou > 0.5 and iou > best_iou:  # Use 0.5 IoU threshold for matching
                            best_iou = iou
                            best_gt_idx = gt_idx

                    # If there's a match, add to classification metrics
                    if best_gt_idx >= 0:
                        y_true.append(int(gt_labels[best_gt_idx][4]))
                        y_pred.append(class_id)

        # Store results for metrics calculation
        all_detections.append(detected_boxes)
        all_ground_truths.append(gt_labels)

        # Visualize if needed
        if args.visualize:
            # Create a copy of the image for visualization
            vis_img = frame.copy()

            # Draw ground truth boxes in green
            for gt in gt_labels:
                x1, y1, x2, y2, cls_id, _ = gt
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_img, f"GT: {model.names[int(cls_id)]}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw detection boxes in blue
            for det in detected_boxes:
                x1, y1, x2, y2, cls_id, conf = det
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis_img, f"{model.names[int(cls_id)]} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Save visualization
            cv2.imwrite(str(vis_dir / f"{img_path.stem}_eval.jpg"), vis_img)

    # Calculate metrics
    metrics = calculate_metrics(all_detections, all_ground_truths, iou_threshold=args.iou_thres)

    # Print and save overall metrics
    print("\nOverall Detection Metrics:")
    print(f"Precision: {metrics['overall']['precision']:.4f}")
    print(f"Recall: {metrics['overall']['recall']:.4f}")
    print(f"F1 Score: {metrics['overall']['f1']:.4f}")
    print(f"TP: {metrics['overall']['tp']}, FP: {metrics['overall']['fp']}, FN: {metrics['overall']['fn']}")

    # Print and save per-class metrics
    print("\nPer-Class Detection Metrics:")
    for cls, cls_metrics in metrics['per_class'].items():
        print(f"Class {cls} ({model.names[cls]}):")
        print(f"  Precision: {cls_metrics['precision']:.4f}")
        print(f"  Recall: {cls_metrics['recall']:.4f}")
        print(f"  F1 Score: {cls_metrics['f1']:.4f}")
        print(f"  TP: {cls_metrics['tp']}, FP: {cls_metrics['fp']}, FN: {cls_metrics['fn']}")

    # Save metrics to file
    with open(output_dir / 'detection_metrics.txt', 'w') as f:
        f.write("Overall Detection Metrics:\n")
        f.write(f"Precision: {metrics['overall']['precision']:.4f}\n")
        f.write(f"Recall: {metrics['overall']['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['overall']['f1']:.4f}\n")
        f.write(f"TP: {metrics['overall']['tp']}, FP: {metrics['overall']['fp']}, FN: {metrics['overall']['fn']}\n\n")

        f.write("Per-Class Detection Metrics:\n")
        for cls, cls_metrics in metrics['per_class'].items():
            f.write(f"Class {cls} ({model.names[cls]}):\n")
            f.write(f"  Precision: {cls_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {cls_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {cls_metrics['f1']:.4f}\n")
            f.write(f"  TP: {cls_metrics['tp']}, FP: {cls_metrics['fp']}, FN: {cls_metrics['fn']}\n\n")

    # Classification metrics
    if y_true and y_pred:
        print("\nClassification Metrics:")
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        print(f"Classification Precision: {precision:.4f}")
        print(f"Classification Recall: {recall:.4f}")
        print(f"Classification F1 Score: {f1:.4f}")

        with open(output_dir / 'classification_metrics.txt', 'w') as f:
            f.write("Classification Metrics:\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")

        # Plot confusion matrix
        class_names = [model.names[i] for i in range(len(model.names))]
        plot_confusion_matrix(y_true, y_pred, class_names, output_dir / 'confusion_matrix.png')

    print(f"\nEvaluation complete. Results saved to {output_dir}")

    # Create and save a summary plot
    plot_performance_summary(metrics, model.names, output_dir / 'performance_summary.png')

    return metrics


def plot_performance_summary(metrics, class_names, output_path):
    """Create a summary plot of the model performance."""
    plt.figure(figsize=(14, 8))

    # Extract metrics
    classes = list(metrics['per_class'].keys())
    classes.sort()

    precision = [metrics['per_class'][cls]['precision'] for cls in classes]
    recall = [metrics['per_class'][cls]['recall'] for cls in classes]
    f1 = [metrics['per_class'][cls]['f1'] for cls in classes]

    # Create plot
    x = np.arange(len(classes))
    width = 0.25

    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1 Score')

    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Detection Performance by Class')
    plt.xticks(x, [class_names[cls] for cls in classes], rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()

    # Load Ultralytics YOLO model
    print(f"Loading YOLO model from {args.model}...")
    model = YOLO(args.model)

    # Display device information
    device = model.device
    print(f"Using device: {device}")

    # Run evaluation
    run_evaluation(args, model)


if __name__ == "__main__":
    main()