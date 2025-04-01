


import os
import shutil

def remove_directories_without_results_png(base_path):
    # Walk through the directory tree
    for root, dirs, files in os.walk(base_path, topdown=False):
        # Skip the base directory itself
        if root == base_path:
            continue
        # Check if 'results.png' is in the current directory
        if 'results.png' not in files:
            # Remove the directory if it does not contain 'results.png'
            print(f"Removing directory: {root}")
            shutil.rmtree(root)

# Define the base path
base_path = '../metacentrum/done_experiments/yolo/lights_single_frames/CRTL_multi_2025_01_15_20_06'

# Call the function to remove directories
remove_directories_without_results_png(base_path)
