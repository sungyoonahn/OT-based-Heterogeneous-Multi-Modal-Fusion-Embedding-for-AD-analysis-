#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RIMA

# Train on PET data using the same patient IDs from MRI training
# This loads patient IDs from results/ADNI_MRI_3D_RESNET/patient_ids.json

PATIENT_IDS_FILE="results/ADNI_MRI_3D_RESNET/patient_ids.json"

if [ ! -f "$PATIENT_IDS_FILE" ]; then
    echo "Error: Patient IDs file not found: $PATIENT_IDS_FILE"
    echo "Please run train_mri.sh first to generate the patient IDs file."
    exit 1
fi

echo "========================================="
echo "Training 3D ResNet on PET data"
echo "Using same patient IDs from MRI training"
echo "Loading patient IDs from: $PATIENT_IDS_FILE"
echo "========================================="

python 3D_resnet.py \
    --data-dir /home/prml/RIMA/datasets/ADNI \
    --save-path results/ADNI_PET_3D_RESNET \
    --load-patient-ids "$PATIENT_IDS_FILE" \
    --epochs 30 \
    --batch-size 2 \
    --lr 1e-4 \
    --target-shape 128 128 128 \
    --augment \
    --seed 42

echo ""
echo "========================================="
echo "PET training complete!"
echo "Results saved to: results/ADNI_PET_3D_RESNET/"
echo "========================================="
