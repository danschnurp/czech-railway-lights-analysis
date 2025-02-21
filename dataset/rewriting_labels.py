import os
import re


def process_files(directory):
    # Get all .txt files in the directory
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    # Pattern to match the specified format
    pattern = r'([23]) (0\.\d+ 0\.\d+ 0\.\d+ 0\.\d+)'
    replacement = r'0 \2'

    for filename in txt_files:
        filepath = os.path.join(directory, filename)

        # Create a backup of the original file
        backup_path = filepath + '.bak'

        try:
            # Read the file content
            with open(filepath, 'r') as file:
                content = file.read()

            # Make a backup
            with open(backup_path, 'w') as file:
                file.write(content)

            # Perform the replacement
            modified_content = re.sub(pattern, replacement, content)

            # Write the modified content back to the original file
            with open(filepath, 'w') as file:
                file.write(modified_content)

            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


# Usage
directory_path = "."  # Current directory, change this to your directory path
process_files("./czech_railway_dataset_4_classes/train/labels/multi_class")
process_files("./czech_railway_dataset_4_classes/val/labels/multi_class")