#!/usr/bin/env python3
"""Quick test to verify MRI T1-T2 data loading"""

import os
import json

# Paths
patient_ids_file = "/home/prml/RIMA/results/ADNI_MRI_T2_3D_RESNET_AD_CN/patient_ids.json"
data_dir = "/home/prml/RIMA/datasets/ADNI/MRI-T1-T2"

# Load patient IDs
with open(patient_ids_file, 'r') as f:
    patient_ids = json.load(f)

print("=" * 80)
print("MRI T1-T2 Data Verification")
print("=" * 80)

print(f"\n✓ Patient IDs file loaded: {patient_ids_file}")
print(f"  - Classes: {list(patient_ids.keys())}")
print(f"  - AD patients: {len(patient_ids['1204_AD_MRI_T1_FIN'])}")
print(f"  - CN patients: {len(patient_ids['1204_CN_MRI_T1_FIN'])}")

# Check directories
classes = {
    "AD_T1": "1204_AD_MRI_T1_FIN",
    "AD_T2": "1204_AD_MRI_T2_FIN",
    "CN_T1": "1204_CN_MRI_T1_FIN",
    "CN_T2": "1204_CN_MRI_T2_FIN",
}

print(f"\n✓ Checking directories:")
for name, dir_name in classes.items():
    dir_path = os.path.join(data_dir, dir_name, "ADNI")
    exists = os.path.isdir(dir_path)
    status = "✓" if exists else "✗"
    print(f"  {status} {name}: {dir_path}")
    if exists:
        # Count subdirectories (patients)
        patient_count = len([d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))])
        print(f"     → {patient_count} patient folders found")

# Check for matching T1-T2 pairs
print(f"\n✓ Checking T1-T2 patient matching:")
ad_t1_patients = set(os.listdir(os.path.join(data_dir, "1204_AD_MRI_T1_FIN/ADNI")))
ad_t2_patients = set(os.listdir(os.path.join(data_dir, "1204_AD_MRI_T2_FIN/ADNI")))
cn_t1_patients = set(os.listdir(os.path.join(data_dir, "1204_CN_MRI_T1_FIN/ADNI")))
cn_t2_patients = set(os.listdir(os.path.join(data_dir, "1204_CN_MRI_T2_FIN/ADNI")))

ad_matched = ad_t1_patients & ad_t2_patients
cn_matched = cn_t1_patients & cn_t2_patients

print(f"  AD matched pairs: {len(ad_matched)} (T1: {len(ad_t1_patients)}, T2: {len(ad_t2_patients)})")
print(f"  CN matched pairs: {len(cn_matched)} (T1: {len(cn_t1_patients)}, T2: {len(cn_t2_patients)})")

# Check how many of the patient_ids have matching T1-T2
print(f"\n✓ Checking patient IDs availability:")
ad_ids_in_json = set(patient_ids['1204_AD_MRI_T1_FIN'])
cn_ids_in_json = set(patient_ids['1204_CN_MRI_T1_FIN'])

ad_available = ad_ids_in_json & ad_matched
cn_available = cn_ids_in_json & cn_matched

print(f"  AD: {len(ad_available)}/{len(ad_ids_in_json)} from JSON have matching T1-T2 pairs")
print(f"  CN: {len(cn_available)}/{len(cn_ids_in_json)} from JSON have matching T1-T2 pairs")

# Summary
total_available = len(ad_available) + len(cn_available)
total_expected = len(ad_ids_in_json) + len(cn_ids_in_json)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Expected total samples: {total_expected}")
print(f"Available paired samples: {total_available}")
print(f"Coverage: {total_available/total_expected*100:.1f}%")
print("\n✓ Data verification complete!")
print("=" * 80)
