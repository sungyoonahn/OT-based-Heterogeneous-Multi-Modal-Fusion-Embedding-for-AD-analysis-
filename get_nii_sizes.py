import os
import nibabel as nib
import numpy as np


def process_directory(root_dir, output_file):
    """
    Process a single directory to find NII files and extract their sizes.
    
    Args:
        root_dir: Directory to search for NII files
        output_file: Path to output file
    """
    nii_files = []
    
    print(f"\nSearching for NII files in {root_dir}...")
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                full_path = os.path.join(root, file)
                nii_files.append(full_path)
    
    print(f"Found {len(nii_files)} NII files.")
    
    if len(nii_files) == 0:
        print(f"No NII files found in {root_dir}")
        with open(output_file, "w") as f:
            f.write(f"No NII files found in {root_dir}\n")
        return
    
    results = []
    max_vol = -1
    min_vol = float('inf')
    max_file = None
    min_file = None
    max_shape = None
    min_shape = None
    
    with open(output_file, "w") as f:
        f.write(f"NII File Sizes for: {root_dir}\n")
        f.write("=" * 80 + "\n\n")
        
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
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Largest size: {max_shape} - {max_file}\n")
        f.write(f"Smallest size: {min_shape} - {min_file}\n")
        f.write(f"Total NII files: {len(nii_files)}\n")
    
    print(f"Results saved to {output_file}")


def main():
    # Configuration - All MRI dataset directories
    datasets = [
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_AD_MRI_T1_FIN/ADNI", "1204_AD_MRI_T1_FIN_Voxel_size.txt"),
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_AD_MRI_T2_FIN/ADNI", "1204_AD_MRI_T2_FIN_Voxel_size.txt"),
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_MCI_MRI_T1_FIN/ADNI", "1204_MCI_MRI_T1_FIN_Voxel_size.txt"),
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_MCI_MRI_T2_FIN/ADNI", "1204_MCI_MRI_T2_FIN_Voxel_size.txt"),
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_CN_MRI_T1_FIN/ADNI", "1204_CN_MRI_T1_FIN_Voxel_size.txt"),
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_CN_MRI_T2_FIN/ADNI", "1204_CN_MRI_T2_FIN_Voxel_size.txt"),
    ]
    
    output_base_dir = "/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/"
    
    print("=" * 80)
    print("NII File Size Analyzer for ADNI MRI Datasets")
    print("=" * 80)
    
    # Process each dataset
    for root_dir, output_filename in datasets:
        output_file = os.path.join(output_base_dir, output_filename)
        process_directory(root_dir, output_file)
    
    print("\n" + "=" * 80)
    print("All datasets processed!")
    print(f"Output files saved to: {output_base_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
