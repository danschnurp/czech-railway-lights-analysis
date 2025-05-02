import os
from PIL import Image
import argparse
from collections import defaultdict


def calculate_average_dimensions(folder_path):
    """
    Calculate the average width and height of all images in a folder and its subfolders.

    Args:
        folder_path (str): Path to the folder containing images

    Returns:
        tuple: (avg_width, avg_height, total_images, dimensions_count)
    """
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    # Variables to track statistics
    total_width = 0
    total_height = 0
    image_count = 0
    failed_files = []
    dimensions_count = defaultdict(int)  # To track frequency of each dimension

    # Walk through all files in folder and subfolders
    for root, _, files in os.walk(folder_path):
        for filename in files:
            # Check if file has a valid image extension
            file_ext = os.path.splitext(filename.lower())[1]
            if file_ext not in valid_extensions:
                continue

            file_path = os.path.join(root, filename)
            try:
                # Open image and get dimensions
                with Image.open(file_path) as img:
                    width, height = img.size

                    # Add to totals
                    total_width += width
                    total_height += height
                    image_count += 1

                    # Record dimension frequency
                    dimensions_count[f"{width}x{height}"] += 1

            except Exception as e:
                failed_files.append((file_path, str(e)))

    # Calculate averages
    avg_width = total_width / image_count if image_count > 0 else 0
    avg_height = total_height / image_count if image_count > 0 else 0

    return avg_width, avg_height, image_count, dimensions_count, failed_files


def main(folder):


    # Calculate statistics
    avg_width, avg_height, image_count, dimensions_count, failed_files = calculate_average_dimensions(folder)

    # Print results
    print(f"\nProcessed {image_count} images:")
    print(f"Average width: {avg_width:.2f} pixels")
    print(f"Average height: {avg_height:.2f} pixels")

    # Print most common dimensions
    print("\nMost common dimensions:")
    sorted_dimensions = sorted(dimensions_count.items(), key=lambda x: x[1], reverse=True)
    for i, (dimension, count) in enumerate(sorted_dimensions[:10]):  # Show top 10
        percentage = (count / image_count) * 100
        print(f"{dimension}: {count} images ({percentage:.1f}%)")

    # Print failed files
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:")
        for path, error in failed_files[:5]:  # Show only first 5
            print(f"- {path}: {error}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")


if __name__ == "__main__":

    folder_path = "/Users/danielschnurpfeil/PycharmProjects/czech-railway-trafic-lights-detection/reconstructed/czech_railway_lights_dataset_extended_roi"
    main(folder_path)
