#!/bin/bash

################################################################################
# EXAMPLE: Quick Test Configuration
# This is a copy of train_both.sh configured for quick testing
################################################################################

# Data settings
DATA_DIR="/home/prml/RIMA/datasets/ADNI"
MAX_SAMPLES_PER_CLASS=10        # Only 10 samples per class for quick test

# Training settings
EPOCHS=5                         # Only 5 epochs for quick test
BATCH_SIZE=2                     # Batch size
LEARNING_RATE=1e-4              # Learning rate
SEED=42                         # Random seed for reproducibility

# Data preprocessing
TARGET_DEPTH=64                 # Smaller size for faster processing
TARGET_HEIGHT=64                # Smaller size for faster processing
TARGET_WIDTH=64                 # Smaller size for faster processing
AUGMENT="--augment"             # Enable augmentation

# Validation settings
VAL_FRACTION=0.2                # Fraction of data for validation

# Hardware settings
NUM_WORKERS=2                   # Number of DataLoader workers
DEVICE="cuda"                   # Device to use (cuda or cpu)

# Output directories
MRI_SAVE_PATH="results/ADNI_MRI_3D_RESNET_TEST"
PET_SAVE_PATH="results/ADNI_PET_3D_RESNET_TEST"

################################################################################
# DO NOT EDIT BELOW THIS LINE
################################################################################

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RIMA

echo "========================================="
echo "QUICK TEST - Master Training Script"
echo "Training 3D ResNet on MRI and PET data"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Samples per class: $MAX_SAMPLES_PER_CLASS"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Target shape: ${TARGET_DEPTH}x${TARGET_HEIGHT}x${TARGET_WIDTH}"
echo "  Augmentation: $([ -n "$AUGMENT" ] && echo "Enabled" || echo "Disabled")"
echo "  Validation fraction: $VAL_FRACTION"
echo "  Device: $DEVICE"
echo "  Random seed: $SEED"
echo "========================================="
echo ""

# Step 1: Train on MRI
echo "Step 1/2: Training on MRI data..."
echo "========================================="

python 3D_resnet.py \
    --data-dir "$DATA_DIR" \
    --save-path "$MRI_SAVE_PATH" \
    --max-samples-per-class $MAX_SAMPLES_PER_CLASS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --target-shape $TARGET_DEPTH $TARGET_HEIGHT $TARGET_WIDTH \
    --val-fraction $VAL_FRACTION \
    --num-workers $NUM_WORKERS \
    --device $DEVICE \
    --seed $SEED \
    $AUGMENT

if [ $? -ne 0 ]; then
    echo "Error: MRI training failed!"
    exit 1
fi

echo ""
echo "MRI training complete!"
echo "Patient IDs saved to: $MRI_SAVE_PATH/patient_ids.json"
echo ""
echo "Waiting 5 seconds before starting PET training..."
sleep 5
echo ""

# Step 2: Train on PET with same patient IDs
echo "Step 2/2: Training on PET data with same patient IDs..."
echo "========================================="

PATIENT_IDS_FILE="$MRI_SAVE_PATH/patient_ids.json"

if [ ! -f "$PATIENT_IDS_FILE" ]; then
    echo "Error: Patient IDs file not found: $PATIENT_IDS_FILE"
    exit 1
fi

echo "Loading patient IDs from: $PATIENT_IDS_FILE"

python 3D_resnet.py \
    --data-dir "$DATA_DIR" \
    --save-path "$PET_SAVE_PATH" \
    --load-patient-ids "$PATIENT_IDS_FILE" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --target-shape $TARGET_DEPTH $TARGET_HEIGHT $TARGET_WIDTH \
    --val-fraction $VAL_FRACTION \
    --num-workers $NUM_WORKERS \
    --device $DEVICE \
    --seed $SEED \
    $AUGMENT

if [ $? -ne 0 ]; then
    echo "Error: PET training failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "All training complete!"
echo "========================================="
echo "MRI results: $MRI_SAVE_PATH/"
echo "PET results: $PET_SAVE_PATH/"
echo "Patient IDs: $PATIENT_IDS_FILE"
echo "========================================="
