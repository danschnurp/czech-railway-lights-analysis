


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
#
# # Call the function to remove directories
# import os
# import sys
#
#
# def count_lines_in_directory(directory='.'):
#     """
#     Count the number of lines in all .txt files in the specified directory.
#
#     Args:
#         directory (str): Path to the directory to scan. Defaults to current directory.
#
#     Returns:
#         dict: Dictionary with filenames as keys and line counts as values
#         int: Total line count across all text files
#     """
#     if not os.path.isdir(directory):
#         print(f"Error: '{directory}' is not a valid directory")
#         return {}, 0
#
#     file_counts = {}
#     total_lines = 0
#
#     try:
#         for filename in os.listdir(directory):
#             if filename.endswith('.txt'):
#                 file_path = os.path.join(directory, filename)
#
#                 try:
#                     with open(file_path, 'r', encoding='utf-8') as file:
#                         line_count = sum(1 for _ in file)
#                         file_counts[filename] = line_count
#                         total_lines += line_count
#                 except Exception as e:
#                     print(f"Error reading {filename}: {e}")
#     except Exception as e:
#         print(f"Error accessing directory: {e}")
#
#     return file_counts, total_lines
#
#
# def main( directory = "/Users/danielschnurpfeil/Downloads/train"):
#
#     print(f"Counting lines in all .txt files in directory: {directory}")
#     file_counts, total_lines = count_lines_in_directory(directory)
#
#     if file_counts:
#         print("\nResults:")
#         print("-" * 40)
#
#         print(f"Total: {total_lines} lines in {len(file_counts)} text files")
#     else:
#         print("No .txt files found in the directory.")
#
#
# if __name__ == "__main__":
#     main()
#     main("/Users/danielschnurpfeil/Downloads/valid")
