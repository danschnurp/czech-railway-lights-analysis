import os
import torch
import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load YOLO model
def load_yolo_model(yolo_model="yolov5s"):
    """
    Load a YOLO model for object detection
    
    Args:
        yolo_model: YOLO model version to use
    
    Returns:
        Loaded YOLO model
    """
    # Using torch hub to load YOLOv5
    model = torch.hub.load('ultralytics/yolov5', yolo_model)
    return model

# Load your custom classifier model
def load_classifier_model(model_path, num_classes):
    """
    Load the custom Czech Railway signal classifier model
    
    Args:
        model_path: Path to the saved .pt model
        num_classes: Number of output classes
    
    Returns:
        Loaded classifier model
    """
    # Create model with the same architecture
    config = AutoConfig.from_pretrained('google/efficientnet-b0', num_labels=num_classes)
    model = CzechRailwayLightNet(config)
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

# Preprocess image for the classifier
def preprocess_for_classifier(image):
    """
    Preprocess an image for the classifier model
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        Preprocessed tensor
    """
    # Ensure image is a PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((16, 34)),  # Resize to the model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension

# Integrated detection and classification pipeline
def detect_and_classify(image_path, yolo_model, classifier_model, class_mapping, 
                       confidence=0.25, target_class="railway_signal", 
                       output_path=None):
    """
    Detect objects with YOLO and classify railway signals with the custom model
    
    Args:
        image_path: Path to the input image
        yolo_model: Loaded YOLO model
        classifier_model: Loaded classifier model
        class_mapping: Dictionary mapping indices to class names
        confidence: Confidence threshold for YOLO detections
        target_class: YOLO class to focus on for classification
        output_path: Path to save the visualization
    
    Returns:
        List of detections with classifications
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run YOLO detection
    yolo_model.conf = confidence  # Set confidence threshold
    results = yolo_model(image_rgb)
    
    # Get detections
    detections = results.pandas().xyxy[0]
    
    # Create visualization image
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Store results
    classified_detections = []
    
    # Process each detection
    for _, detection in detections.iterrows():
        # Check if detection is the target class (if specified)
        if target_class and detection['name'] != target_class:
            continue
            
        # Extract bounding box
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        
        # Extract the region for classification
        roi = image_rgb[y1:y2, x1:x2]
        
        if roi.size == 0:
            continue
            
        # Convert to PIL for preprocessing
        roi_pil = Image.fromarray(roi)
        
        # Preprocess for classifier
        roi_tensor = preprocess_for_classifier(roi_pil)
        
        # Run classifier
        with torch.no_grad():
            outputs = classifier_model(pixel_values=roi_tensor)
            logits = outputs['logits']
            predicted_class_idx = torch.argmax(logits, dim=1).item()
            confidence_score = torch.softmax(logits, dim=1)[0, predicted_class_idx].item()
        
        # Get predicted class name
        predicted_class = class_mapping.get(str(predicted_class_idx), f"Unknown ({predicted_class_idx})")
        
        # Store the result
        result = {
            'bbox': [x1, y1, x2, y2],
            'yolo_confidence': float(detection['confidence']),
            'class': predicted_class,
            'class_confidence': float(confidence_score)
        }
        classified_detections.append(result)
        
        # Draw on image
        color = (255, 0, 0)  # Red
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Label with class
        label = f"{predicted_class} ({confidence_score:.2f})"
        draw.text((x1, y1-15), label, fill=color)
    
    # Save or show the result
    if output_path:
        pil_image.save(output_path)
        print(f"Visualization saved to {output_path}")
    
    return classified_detections

# Process a whole directory of images
def process_directory(image_dir, output_dir, yolo_model, classifier_model, class_mapping,
                     confidence=0.25, target_class="railway_signal"):
    """
    Process all images in a directory
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save output images
        yolo_model: Loaded YOLO model
        classifier_model: Loaded classifier model
        class_mapping: Dictionary mapping indices to class names
        confidence: Confidence threshold for YOLO detections
        target_class: YOLO class to focus on for classification
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_dir) 
                  if os.path.isfile(os.path.join(image_dir, f)) and 
                  os.path.splitext(f.lower())[1] in image_extensions]
    
    results = {}
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        output_path = os.path.join(output_dir, f"detected_{image_file}")
        
        # Run detection and classification
        detections = detect_and_classify(
            image_path, 
            yolo_model, 
            classifier_model, 
            class_mapping, 
            confidence=confidence,
            target_class=target_class, 
            output_path=output_path
        )
        
        # Store results
        results[image_file] = detections
        
        print(f"Processed {image_file}: Found {len(detections)} railway signals")
    
    # Save all results to JSON
    results_path = os.path.join(output_dir, "detection_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

# Real-time webcam detection and classification
def real_time_detection(yolo_model, classifier_model, class_mapping, 
                       confidence=0.25, target_class="railway_signal", 
                       camera_id=0):
    """
    Run real-time detection and classification on webcam feed
    
    Args:
        yolo_model: Loaded YOLO model
        classifier_model: Loaded classifier model
        class_mapping: Dictionary mapping indices to class names
        confidence: Confidence threshold for YOLO detections
        target_class: YOLO class to focus on for classification
        camera_id: Camera/webcam ID
    """
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Convert to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        yolo_model.conf = confidence
        results = yolo_model(frame_rgb)
        
        # Get detections
        detections = results.pandas().xyxy[0]
        
        # Process each detection
        for _, detection in detections.iterrows():
            # Check if detection is the target class (if specified)
            if target_class and detection['name'] != target_class:
                continue
                
            # Extract bounding box
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            
            # Extract the region for classification
            roi = frame_rgb[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
                
            # Convert to PIL for preprocessing
            roi_pil = Image.fromarray(roi)
            
            # Preprocess for classifier
            roi_tensor = preprocess_for_classifier(roi_pil)
            
            # Run classifier
            with torch.no_grad():
                outputs = classifier_model(pixel_values=roi_tensor)
                logits = outputs['logits']
                predicted_class_idx = torch.argmax(logits, dim=1).item()
                confidence_score = torch.softmax(logits, dim=1)[0, predicted_class_idx].item()
            
            # Get predicted class name
            predicted_class = class_mapping.get(str(predicted_class_idx), f"Unknown ({predicted_class_idx})")
            
            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label with class
            label = f"{predicted_class} ({confidence_score:.2f})"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Railway Signal Detection', frame)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Paths
    model_path = 'czech_railway_lights_model.pt'
    class_mapping_path = 'class_mapping.json'
    test_image_path = 'test_image.jpg'  # Replace with your test image
    
    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    num_classes = len(class_mapping)
    
    # Load models
    yolo_model = load_yolo_model("yolov5s")  # Can use s, m, l, or x versions
    classifier_model = load_classifier_model(model_path, num_classes)
    
    # Set YOLO parameters
    yolo_model.conf = 0.25  # Confidence threshold
    
    # Choose one of the following operations:
    
    # 1. Process a single image
    detect_and_classify(
        test_image_path,
        yolo_model,
        classifier_model,
        class_mapping,
        confidence=0.6,
        target_class=None,  # Set to None to detect all objects, or use "traffic light" or similar
        output_path="detected_output.jpg"
    )
    
    # 2. Process a directory of images
    """
    process_directory(
        "test_images/",
        "output_images/",
        yolo_model,
        classifier_model,
        class_mapping,
        confidence=0.25,
        target_class="traffic light"  # Adjust based on YOLO classes
    )
    """
    
    # 3. Real-time webcam detection
    """
    real_time_detection(
        yolo_model,
        classifier_model,
        class_mapping,
        confidence=0.25,
        target_class="traffic light"  # Adjust based on YOLO classes
    )
    """
