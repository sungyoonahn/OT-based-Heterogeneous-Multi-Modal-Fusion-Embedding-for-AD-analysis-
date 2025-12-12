import os
import shutil

def cleanup_files():
    # Target directories
    base_dir = "/home/prml/RIMA/datasets/ADNI/MRI-T1-T2"
    target_dirs = [
        os.path.join(base_dir, "1204_AD_MRI_T1_FIN_394"),
        os.path.join(base_dir, "1204_AD_MRI_T2_FIN_394"),
        os.path.join(base_dir, "1204_CN_MRI_T1_FIN_394"),
        os.path.join(base_dir, "1204_CN_MRI_T2_FIN_394")
    ]
    
    deleted_count = 0
    kept_count = 0
    
    for target_dir in target_dirs:
        print(f"Cleaning up {target_dir}...")
        
        if not os.path.exists(target_dir):
            print(f"  Directory not found: {target_dir}")
            continue
            
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check extension
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    kept_count += 1
                else:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        # print(f"  Deleted: {file}")
                    except Exception as e:
                        print(f"  Error deleting {file_path}: {e}")
                        
    print(f"\nCleanup complete.")
    print(f"Total files deleted: {deleted_count}")
    print(f"Total NIfTI files kept: {kept_count}")

if __name__ == "__main__":
    cleanup_files()
