import os
import re


def process_files(directory):
    # Get all .txt files in the directory
    txt_files = [f for f in os.listdir(directory) if f.endswith('.bak')]

    # Pattern to match the specified format
    pattern = r'([1]) (0\.\d+ 0\.\d+ 0\.\d+ 0\.\d+)'
    replacement = r'0 \2'

    for filename in txt_files:
        filepath = os.path.join(directory, filename)

        os.remove(filepath)

# Usage
directory_path = "."  # Current directory, change this to your directory path
process_files("./czech_railway_dataset_1_class/train/labels/multi_class")
process_files("./czech_railway_dataset_1_class/val/labels/multi_class")