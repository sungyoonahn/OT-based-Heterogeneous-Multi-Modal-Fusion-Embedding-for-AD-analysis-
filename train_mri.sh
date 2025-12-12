#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RIMA

# Train on MRI data with 50 samples per class
# This will save the patient IDs to results/ADNI_MRI_3D_RESNET/patient_ids.json

echo "========================================="
echo "Training 3D ResNet on MRI data"
echo "Using 50 samples per class"
echo "========================================="

python 3D_resnet.py \
    --data-dir /home/prml/RIMA/datasets/ADNI \
    --save-path results/ADNI_MRI_3D_RESNET \
    --max-samples-per-class 50 \
    --epochs 30 \
    --batch-size 2 \
    --lr 1e-4 \
    --target-shape 128 128 128 \
    --augment \
    --seed 42

echo ""
echo "========================================="
echo "MRI training complete!"
echo "Patient IDs saved to: results/ADNI_MRI_3D_RESNET/patient_ids.json"
echo "========================================="
