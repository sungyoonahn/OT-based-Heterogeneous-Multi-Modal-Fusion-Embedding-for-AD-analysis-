#!/bin/bash

################################################################################
# Training script for MRI T1-T2 Optimal Transport Model
################################################################################

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rima

# Configuration
DATA_DIR="/home/prml/RIMA/datasets/ADNI/MRI-T1-T2"
PATIENT_IDS="/home/prml/RIMA/results/ADNI_MRI_T1_3D_RESNET_AD_CN/patient_ids.json"
SAVE_PATH="results/ADNI_MRI_T1_T2_OT_AD_CN"
EPOCHS=200
BATCH_SIZE=4
LEARNING_RATE=2e-5
SEED=42
MODEL_DEPTH=101
TARGET_SHAPE="128 128 128"
VAL_FRACTION=0.2
NUM_WORKERS=2
DEVICE="cuda"
MAX_JAX_SAMPLES=100

echo "========================================="
echo "MRI T1-T2 Optimal Transport Training"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Patient IDs: $PATIENT_IDS"
echo "  Save path: $SAVE_PATH"
echo "  Model depth: $MODEL_DEPTH"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Target shape: $TARGET_SHAPE"
echo "  Validation fraction: $VAL_FRACTION"
echo "  Device: $DEVICE"
echo "  Random seed: $SEED"
echo "========================================="
echo ""

python MRI_T1_T2_OT.py \
    --data-dir "$DATA_DIR" \
    --save-path "$SAVE_PATH" \
    --load-patient-ids "$PATIENT_IDS" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --target-shape $TARGET_SHAPE \
    --val-fraction $VAL_FRACTION \
    --num-workers $NUM_WORKERS \
    --device $DEVICE \
    --seed $SEED \
    --model-depth $MODEL_DEPTH \
    --max-jax-samples $MAX_JAX_SAMPLES \
    --augment

if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "Training complete!"
echo "========================================="
echo "Results saved to: $SAVE_PATH/"
echo "Confusion matrix: $SAVE_PATH/confusion_matrix_T1_T2_OT.png"
echo "========================================="
