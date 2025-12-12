#!/usr/bin/env python3
"""Detailed verification of T1-T2 file availability (not just folders)"""

import os
import json

def count_nii_files(directory):
    """Count .nii/.nii.gz files recursively"""
    count = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(('.nii', '.nii.gz')):
                count += 1
    return count

def get_patients_with_nii_files(base_dir):
    """Get set of patient IDs that have .nii files"""
    patients = set()
    if not os.path.isdir(base_dir):
        return patients
    
    for patient_folder in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_folder)
        if os.path.isdir(patient_path):
            # Check if this patient folder has any .nii files
            nii_count = count_nii_files(patient_path)
            if nii_count > 0:
                patients.add(patient_folder)
    return patients

# Paths
patient_ids_file = "/home/prml/RIMA/results/ADNI_MRI_T2_3D_RESNET_AD_CN/patient_ids.json"
data_dir = "/home/prml/RIMA/datasets/ADNI/MRI-T1-T2"

# Load patient IDs from JSON
with open(patient_ids_file, 'r') as f:
    patient_ids = json.load(f)

print("=" * 80)
print("DETAILED T1-T2 FILE VERIFICATION")
print("=" * 80)

# Get patients with actual .nii files
print("\n📁 Scanning for patients with .nii files...")

ad_t1_with_files = get_patients_with_nii_files(os.path.join(data_dir, "1204_AD_MRI_T1_FIN/ADNI"))
ad_t2_with_files = get_patients_with_nii_files(os.path.join(data_dir, "1204_AD_MRI_T2_FIN/ADNI"))
cn_t1_with_files = get_patients_with_nii_files(os.path.join(data_dir, "1204_CN_MRI_T1_FIN/ADNI"))
cn_t2_with_files = get_patients_with_nii_files(os.path.join(data_dir, "1204_CN_MRI_T2_FIN/ADNI"))

print(f"  AD T1: {len(ad_t1_with_files)} patients with .nii files")
print(f"  AD T2: {len(ad_t2_with_files)} patients with .nii files")
print(f"  CN T1: {len(cn_t1_with_files)} patients with .nii files")
print(f"  CN T2: {len(cn_t2_with_files)} patients with .nii files")

# Find matching pairs (patients with BOTH T1 and T2 files)
ad_matched_files = ad_t1_with_files & ad_t2_with_files
cn_matched_files = cn_t1_with_files & cn_t2_with_files

print(f"\n🔗 Patients with BOTH T1 and T2 .nii files:")
print(f"  AD: {len(ad_matched_files)} matched pairs")
print(f"  CN: {len(cn_matched_files)} matched pairs")

# Check against patient_ids.json
ad_ids_in_json = set(patient_ids['1204_AD_MRI_T1_FIN'])
cn_ids_in_json = set(patient_ids['1204_CN_MRI_T1_FIN'])

print(f"\n📋 Patient IDs in JSON file:")
print(f"  AD: {len(ad_ids_in_json)} patients")
print(f"  CN: {len(cn_ids_in_json)} patients")

# Find which patients from JSON have matching T1-T2 files
ad_available = ad_ids_in_json & ad_matched_files
cn_available = cn_ids_in_json & cn_matched_files

print(f"\n✓ Patients from JSON with matching T1-T2 files:")
print(f"  AD: {len(ad_available)}/{len(ad_ids_in_json)} ({len(ad_available)/len(ad_ids_in_json)*100:.1f}%)")
print(f"  CN: {len(cn_available)}/{len(cn_ids_in_json)} ({len(cn_available)/len(cn_ids_in_json)*100:.1f}%)")

# Find missing patients
ad_missing = ad_ids_in_json - ad_matched_files
cn_missing = cn_ids_in_json - cn_matched_files

print(f"\n❌ Patients from JSON WITHOUT matching T1-T2 files:")
print(f"  AD missing: {len(ad_missing)}")
print(f"  CN missing: {len(cn_missing)}")

if len(ad_missing) > 0:
    print(f"\n  First 10 missing AD patients:")
    for i, patient_id in enumerate(sorted(list(ad_missing))[:10]):
        has_t1 = patient_id in ad_t1_with_files
        has_t2 = patient_id in ad_t2_with_files
        print(f"    {patient_id}: T1={'✓' if has_t1 else '✗'} T2={'✓' if has_t2 else '✗'}")

if len(cn_missing) > 0:
    print(f"\n  First 10 missing CN patients:")
    for i, patient_id in enumerate(sorted(list(cn_missing))[:10]):
        has_t1 = patient_id in cn_t1_with_files
        has_t2 = patient_id in cn_t2_with_files
        print(f"    {patient_id}: T1={'✓' if has_t1 else '✗'} T2={'✓' if has_t2 else '✗'}")

# Summary
total_available = len(ad_available) + len(cn_available)
total_expected = len(ad_ids_in_json) + len(cn_ids_in_json)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Expected (from JSON): {total_expected} samples")
print(f"Actually available:   {total_available} samples ({total_available/total_expected*100:.1f}%)")
print(f"  - AD: {len(ad_available)} samples")
print(f"  - CN: {len(cn_available)} samples")
print(f"Missing: {total_expected - total_available} samples")
print("=" * 80)
