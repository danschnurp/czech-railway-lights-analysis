import os
import cv2
import json

from utils.general_utils import download_video
from utils.image_utils import convert_normalized_roi_to_pixels


def extract_crops(data, output_dir, videos_dir):
    """Process each record and extract crops based on ROI."""
    os.makedirs(output_dir, exist_ok=True)

    for record in data:
        # Parse record
        ytlink = record["ytlink"]
        timestamp = record["timestamp in video"]
        roi = record["roi coordinates"]
        class_name = record["color"]
        video_name = record["video name"]

        # Prepare class folder
        class_folder = os.path.join(output_dir, class_name.replace(" ", "_"))
        os.makedirs(class_folder, exist_ok=True)

        # Download and process the video
        video_file = download_video(ytlink, videos_dir)
        cap = cv2.VideoCapture(os.path.join(videos_dir,video_file))

        # Seek to the frame at the specified timestamp
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * timestamp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame at timestamp {timestamp} in {video_name}")
            cap.release()
            continue

        # Extract ROI
        frame_height, frame_width = frame.shape[:2]
        x_min, y_min, width, height = convert_normalized_roi_to_pixels(roi, frame_width, frame_height)
        crop = frame[y_min:height, x_min:width]

        # Save the cropped image
        crop_filename = f"{video_name.replace(' ', '_')}_{int(timestamp)}.jpg"
        crop_path = os.path.join(class_folder, crop_filename)
        cv2.imwrite(crop_path, crop)

        print(f"Saved crop: {crop_path}")

        cap.release()

# Main Program
if __name__ == "__main__":
    input_json = "../railway_datasets/simple_classes/warning_go.json"  # JSON file containing the records
    output_directory = "./output"  # Directory to save cropped images
    videos_dir = "/Volumes/zalohy/dip"
    # Load JSON data
    with open(input_json, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    extract_crops(json_data["data"], output_directory, videos_dir)
