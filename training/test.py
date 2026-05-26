from pathlib import Path

# Root directory containing the 4 class folders
root_dir = Path(r"C:\Users\LEGION\Desktop\universidade\Mestrado\DAA\AP_Modulo2\training\dataset\train\Stricture")

unique_ids = set()

# Iterate through all image files in subfolders
for img_path in root_dir.rglob("*"):
    if img_path.is_file():
        # Extract filename without extension
        filename = img_path.stem
        
        # Split at first underscore
        if "_" in filename:
            patient_id = filename.split("_")[0]
            unique_ids.add(patient_id)

print(f"Number of unique IDs: {len(unique_ids)}")

# Optional: print all unique IDs
print(sorted(unique_ids))