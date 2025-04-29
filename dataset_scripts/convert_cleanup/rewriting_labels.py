import os
import re


def process_files(directory, old, rep=0, remove=False):
    # Get all .txt files in the directory
    txt_files = [f for f in os.listdir(directory)]

    # Pattern to match the specified format
    string_pattern = '([' + str(old) + r']) (0\.\d+ 0\.\d+ 0\.\d+ 0\.\d+)'
    pattern = r"" + string_pattern
    string_rep =  str(rep) + r'  \2'
    replacement = r''
    if not remove:
        replacement += string_rep
    for filename in txt_files:
        filepath = os.path.join(directory, filename)
        try:
            # Read the file content
            with open(filepath, 'r') as file:
                content = file.read()


            # Perform the replacement
            modified_content = re.sub(pattern, replacement, content)
            if modified_content == "":
                print(f"Deleted label {old} in {filename}")
            elif modified_content != content:
                print(f"Modified label {old} to {rep} in {filename}")

            # Write the modified content back to the original file
            with open(filepath, 'w') as file:
                file.write(modified_content)



        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

# Usage
directory_path = "."  # Current directory, change this to your directory path

paths =[
"../../reconstructed/czech_railway_lights_dataset_extended_1_class/train/labels/multi_class",
"../../reconstructed/czech_railway_lights_dataset_extended_1_class/val/labels/multi_class"

]
to_replace = [1,2,3,4,5,7]

for i in paths:
    for j in to_replace:
            process_files(directory=i, old=j)
    process_files(directory=i, remove=True, old=6)