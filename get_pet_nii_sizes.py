import os
import nibabel as nib
import numpy as np

# Updated path for PET files
root_dir = "/home/prml/RIMA/datasets/ADNI/MRI-PET/AD_PET_130_FIN/ADNI_NII/"
output_file = "AD_PET_130_FIN_Voxel_size.txt"

nii_files = []

print(f"Searching for NII files in {root_dir}...")

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            full_path = os.path.join(root, file)
            nii_files.append(full_path)

print(f"Found {len(nii_files)} NII files.")

results = []
max_vol = -1
min_vol = float('inf')
max_file = None
min_file = None
max_shape = None
min_shape = None

with open(output_file, "w") as f:
    for file_path in sorted(nii_files):
        try:
            img = nib.load(file_path)
            shape = img.shape
            
            vol = np.prod(shape)
            
            results.append((os.path.basename(file_path), shape))
            
            if vol > max_vol:
                max_vol = vol
                max_file = os.path.basename(file_path)
                max_shape = shape
            
            if vol < min_vol:
                min_vol = vol
                min_file = os.path.basename(file_path)
                min_shape = shape
                
            f.write(f"{os.path.basename(file_path)}: {shape}\n")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            f.write(f"{os.path.basename(file_path)}: Error reading file\n")

    f.write("\n")
    f.write(f"Largest size: {max_shape} - {max_file}\n")
    f.write(f"Smallest size: {min_shape} - {min_file}\n")

print(f"Finished processing. Results saved to {output_file}")
