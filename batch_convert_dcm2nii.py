#!/usr/bin/env python3
"""
Batch DICOM to NIfTI Converter for ADNI MRI Dataset

This script processes all patient directories in the ADNI MRI dataset and converts
DICOM files to NIfTI format, saving the NIfTI files in the same directory as the DICOM files.
"""

import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm


def find_dicom_directories(base_dir):
    """
    Find all directories containing DICOM files in the ADNI MRI dataset.
    
    The directory structure is:
    base_dir/subject_id/scan_type/date_metadata/patient_id/*.dcm
    
    Returns:
        List of tuples: (subject_id, scan_type, date_metadata, patient_id, dicom_dir_path)
    """
    base_path = Path(base_dir)
    dicom_dirs = []
    
    if not base_path.exists():
        print(f"Warning: Directory does not exist: {base_dir}")
        return dicom_dirs
    
    # Iterate through subject directories
    for subject_dir in sorted(base_path.iterdir()):
        if not subject_dir.is_dir():
            continue
        
        subject_id = subject_dir.name
        
        # Iterate through scan type directories (e.g., MP-RAGE, Axial_T2_STAR)
        for scan_type_dir in subject_dir.iterdir():
            if not scan_type_dir.is_dir():
                continue
            
            scan_type = scan_type_dir.name
            
            # Iterate through date/metadata directories
            for date_metadata_dir in scan_type_dir.iterdir():
                if not date_metadata_dir.is_dir():
                    continue
                
                date_metadata = date_metadata_dir.name
                
                # Iterate through patient_id directories (innermost level before DICOM files)
                for patient_id_dir in date_metadata_dir.iterdir():
                    if not patient_id_dir.is_dir():
                        continue
                    
                    patient_id = patient_id_dir.name
                    
                    # Check if this directory contains .dcm files
                    dcm_files = list(patient_id_dir.glob("*.dcm"))
                    if dcm_files:
                        dicom_dirs.append((subject_id, scan_type, date_metadata, patient_id, patient_id_dir))
    
    return dicom_dirs


def convert_dicom_directory(subject_id, scan_type, date_metadata, patient_id, dicom_dir, no_compress=True):
    """
    Convert a single directory's DICOM files to NIfTI.
    
    Args:
        subject_id: Subject identifier
        scan_type: Scan type (e.g., MP-RAGE, Axial_T2_STAR)
        date_metadata: Date/metadata identifier
        patient_id: Patient identifier (innermost directory)
        dicom_dir: Path to directory containing DICOM files
        no_compress: If True, save as .nii; if False, save as .nii.gz
    
    Returns:
        True if successful, False otherwise
    """
    # Create output filename in the same directory
    extension = ".nii" if no_compress else ".nii.gz"
    output_file = dicom_dir / f"{subject_id}_{scan_type}_{date_metadata}{extension}"
    
    # Skip if already exists
    if output_file.exists():
        return True  # Silent skip
    
    # Build command
    cmd = [
        "python",
        "convert_dcm2nii.py",
        str(dicom_dir),
        str(output_file)
    ]
    
    if no_compress:
        cmd.append("--no-compress")
    
    # Run conversion
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"  ✗ {subject_id}/{scan_type}/{date_metadata}/{patient_id} - Conversion failed")
            print(f"    Error: {result.stderr[:200]}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"  ✗ {subject_id}/{scan_type}/{date_metadata}/{patient_id} - Timeout")
        return False
    except Exception as e:
        print(f"  ✗ {subject_id}/{scan_type}/{date_metadata}/{patient_id} - Error: {e}")
        return False


def process_base_directory(base_dir, dataset_name):
    """
    Process a single base directory and convert all DICOM files found.
    
    Args:
        base_dir: Base directory to process
        dataset_name: Name of the dataset for display purposes
    
    Returns:
        Tuple of (successful_count, failed_count)
    """
    print(f"\nProcessing: {dataset_name}")
    print(f"Directory: {base_dir}")
    
    # Find all DICOM directories
    dicom_dirs = find_dicom_directories(base_dir)
    
    if not dicom_dirs:
        print(f"  No DICOM directories found")
        return 0, 0
    
    print(f"  Found {len(dicom_dirs)} DICOM directories")
    
    successful = 0
    failed = 0
    
    for subject_id, scan_type, date_metadata, patient_id, dicom_dir in tqdm(dicom_dirs, desc=f"  Converting {dataset_name}", leave=False):
        success = convert_dicom_directory(subject_id, scan_type, date_metadata, patient_id, dicom_dir, no_compress=True)
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"  ✓ Successful: {successful}, ✗ Failed: {failed}")
    
    return successful, failed


def main():
    # Configuration - All MRI dataset directories
    datasets = [
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_AD_MRI_T1_FIN/ADNI", "AD MRI T1"),
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_AD_MRI_T2_FIN/ADNI", "AD MRI T2"),
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_MCI_MRI_T1_FIN/ADNI", "MCI MRI T1"),
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_MCI_MRI_T2_FIN/ADNI", "MCI MRI T2"),
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_CN_MRI_T1_FIN/ADNI", "CN MRI T1"),
        ("/home/prml/RIMA/datasets/ADNI/MRI-T1-T2/1204_CN_MRI_T2_FIN/ADNI", "CN MRI T2"),
    ]
    
    print("=" * 70)
    print("ADNI MRI DICOM to NIfTI Batch Converter")
    print("=" * 70)
    print("This script will convert DICOM files to NIfTI format")
    print("NIfTI files will be saved in the same directory as DICOM files")
    print("=" * 70)
    
    total_successful = 0
    total_failed = 0
    
    # Process each dataset
    for base_dir, dataset_name in datasets:
        successful, failed = process_base_directory(base_dir, dataset_name)
        total_successful += successful
        total_failed += failed
    
    # Overall Summary
    total = total_successful + total_failed
    print("\n" + "=" * 70)
    print("OVERALL CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Total directories:  {total}")
    print(f"Successful:         {total_successful}")
    print(f"Failed:             {total_failed}")
    if total > 0:
        print(f"Success rate:       {total_successful/total*100:.1f}%")
    print("=" * 70)
    print("\nNIfTI files saved in the same directories as DICOM files")


if __name__ == '__main__':
    main()
