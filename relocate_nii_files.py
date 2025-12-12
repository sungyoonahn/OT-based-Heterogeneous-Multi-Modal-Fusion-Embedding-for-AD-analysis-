import json
import os
import shutil
from pathlib import Path

def relocate_files():
    # Configuration
    json_path = "/home/prml/RIMA/results/ADNI_MRI_T2_3D_RESNET_AD_CN/patient_ids.json"
    base_dir = "/home/prml/RIMA/datasets/ADNI/MRI-T1-T2"
    
    # Load patient IDs
    print(f"Loading patient IDs from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract ID lists
    # Note: The user request implies using the keys in the JSON to identify the groups
    # The keys in the JSON are "1204_AD_MRI_T1_FIN" and "1204_CN_MRI_T1_FIN"
    ad_ids = data.get("1204_AD_MRI_T1_FIN", [])
    cn_ids = data.get("1204_CN_MRI_T1_FIN", [])
    
    print(f"Found {len(ad_ids)} AD IDs and {len(cn_ids)} CN IDs")
    
    # Define mappings (Source Directory -> ID List)
    # We need to handle both T1 and T2 for both AD and CN
    
    mappings = [
        {
            "name": "AD T1",
            "source": os.path.join(base_dir, "1204_AD_MRI_T1_FIN"),
            "ids": ad_ids
        },
        {
            "name": "AD T2",
            "source": os.path.join(base_dir, "1204_AD_MRI_T2_FIN"),
            "ids": ad_ids
        },
        {
            "name": "CN T1",
            "source": os.path.join(base_dir, "1204_CN_MRI_T1_FIN"),
            "ids": cn_ids
        },
        {
            "name": "CN T2",
            "source": os.path.join(base_dir, "1204_CN_MRI_T2_FIN"),
            "ids": cn_ids
        }
    ]
    
    for mapping in mappings:
        source_dir = mapping["source"]
        ids = mapping["ids"]
        count = len(ids)
        
        # Construct destination directory name: {OriginalName}_{Count}
        dir_name = os.path.basename(source_dir)
        dest_dir = os.path.join(base_dir, f"{dir_name}_{count}")
        
        print(f"\nProcessing {mapping['name']}...")
        print(f"Source: {source_dir}")
        print(f"Destination: {dest_dir}")
        
        if not os.path.exists(source_dir):
            print(f"Error: Source directory {source_dir} does not exist. Skipping.")
            continue
            
        # Create destination directory if it doesn't exist
        # We need to maintain the ADNI structure, so we create the base dest dir first
        # The structure is dest_dir/ADNI/{subject_id}
        
        processed_count = 0
        
        for subject_id in ids:
            # Source path: source_dir/ADNI/subject_id
            src_subject_path = os.path.join(source_dir, "ADNI", subject_id)
            
            # Destination path: dest_dir/ADNI/subject_id
            dest_subject_path = os.path.join(dest_dir, "ADNI", subject_id)
            
            if os.path.exists(src_subject_path):
                try:
                    # Copy the directory tree
                    if os.path.exists(dest_subject_path):
                        # print(f"  Subject {subject_id} already exists in destination. Skipping.")
                        pass
                    else:
                        shutil.copytree(src_subject_path, dest_subject_path)
                        # print(f"  Copied {subject_id}")
                    processed_count += 1
                except Exception as e:
                    print(f"  Error copying {subject_id}: {e}")
            else:
                print(f"  Warning: Subject {subject_id} not found in {src_subject_path}")
        
        print(f"Finished {mapping['name']}. Processed {processed_count}/{count} subjects.")

if __name__ == "__main__":
    relocate_files()
