import os



# Define paths
annotated_dir = "../dataset/czech_railway_lights_dataset_4_classes/train/images/multi_class_annotated"
unannotated_dir = "../dataset/czech_railway_lights_dataset_4_classes/train/images/multi_class"
labels_dir = "../dataset/czech_railway_lights_dataset_4_classes/train/labels/multi_class"

# Get JPG filenames (without extensions) in the annotated directory
annotated_files = {os.path.splitext(f)[0] for f in os.listdir(annotated_dir) if f.endswith(".jpg")}

unannotated_files = {os.path.splitext(f)[0] for f in os.listdir(unannotated_dir) if f.endswith(".jpg")}

# Get label filenames (without extensions) in the labels directory
label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith(".txt")}  # Change to .xml if needed


# Find label files without matching JPGs
files_to_delete = label_files - annotated_files

# Delete unmatched label files
for file in files_to_delete:
    file_path = os.path.join(labels_dir, file + ".txt")  # Change to .xml if needed
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")

print("Label cleanup complete.")

# Find files in unannotated that are NOT in annotated
files_to_delete = unannotated_files - annotated_files

# Delete the unmatched files
for file in files_to_delete:
    for ext in (".jpg", ".png", ".jpeg"):  # Adjust based on your file types
        file_path = os.path.join(unannotated_dir, file + ext)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")

print("Cleanup complete.")

