#!/bin/bash

################################################################################
# HYPERPARAMETERS - Edit these values to customize training
################################################################################

# Data settings
DATA_DIR_T1="/home/prml/RIMA/datasets/ADNI/MRI-T1-T2"
DATA_DIR_T2="/home/prml/RIMA/datasets/ADNI/MRI-T1-T2"

# Training settings
EPOCHS=200                        # Number of training epochs
BATCH_SIZE=4                     # Batch size
LEARNING_RATE=2e-5              # Learning rate
SEED=42                         # Random seed for reproducibility
MODEL_DEPTH=101                 # ResNet depth (10, 18, 34, 50, 101, 152, 200)

# Data preprocessing
TARGET_DEPTH=128                # Target depth for 3D volumes
TARGET_HEIGHT=128               # Target height for 3D volumes
TARGET_WIDTH=128                # Target width for 3D volumes
AUGMENT="--augment"             # Enable augmentation (set to "" to disable)

# Validation settings
VAL_FRACTION=0.2                # Fraction of data for validation

# Hardware settings
NUM_WORKERS=2                   # Number of DataLoader workers
DEVICE="cuda"                   # Device to use (cuda or cpu)

# Output directories
T1_SAVE_PATH="results/ADNI_MRI_T1_3D_RESNET_AD_CN"
T2_SAVE_PATH="results/ADNI_MRI_T2_3D_RESNET_AD_CN"

# Classes to train on
CLASSES="AD CN"

################################################################################
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
################################################################################

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rima

echo "========================================="
echo "Master Training Script"
echo "Training 3D ResNet on MRI T1 and T2 data"
echo "========================================="
echo ""
echo "Configuration:"
echo "  Model depth: $MODEL_DEPTH"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Target shape: ${TARGET_DEPTH}x${TARGET_HEIGHT}x${TARGET_WIDTH}"
echo "  Augmentation: $([ -n "$AUGMENT" ] && echo "Enabled" || echo "Disabled")"
echo "  Validation fraction: $VAL_FRACTION"
echo "  Device: $DEVICE"
echo "  Random seed: $SEED"
echo "  Classes: $CLASSES"
echo "  Balancing: CN samples will be randomly selected to match AD count"
echo "========================================="
echo ""

# Step 1: Train on MRI T1
echo "Step 1/2: Training on MRI T1 data..."
echo "========================================="
echo "Data directory: $DATA_DIR_T1"
echo "Expected classes: 1204_AD_MRI_T1_FIN, 1204_CN_MRI_T1_FIN"
echo ""

python 3D_resnet.py \
    --data-dir "$DATA_DIR_T1" \
    --save-path "$T1_SAVE_PATH" \
    --classes $CLASSES \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --target-shape $TARGET_DEPTH $TARGET_HEIGHT $TARGET_WIDTH \
    --val-fraction $VAL_FRACTION \
    --num-workers $NUM_WORKERS \
    --device $DEVICE \
    --seed $SEED \
    --model-depth $MODEL_DEPTH \
    --balance-to-minority \
    $AUGMENT

if [ $? -ne 0 ]; then
    echo "Error: MRI T1 training failed!"
    exit 1
fi

echo ""
echo "MRI T1 training complete!"
echo ""
echo "Waiting 5 seconds before starting MRI T2 training..."
sleep 5
echo ""

# Step 2: Train on MRI T2
echo "Step 2/2: Training on MRI T2 data..."
echo "========================================="
echo "Data directory: $DATA_DIR_T2"
echo "Expected classes: 1204_AD_MRI_T2_FIN, 1204_CN_MRI_T2_FIN"
echo ""

python 3D_resnet.py \
    --data-dir "$DATA_DIR_T2" \
    --save-path "$T2_SAVE_PATH" \
    --classes $CLASSES \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --target-shape $TARGET_DEPTH $TARGET_HEIGHT $TARGET_WIDTH \
    --val-fraction $VAL_FRACTION \
    --num-workers $NUM_WORKERS \
    --device $DEVICE \
    --seed $SEED \
    --model-depth $MODEL_DEPTH \
    --balance-to-minority \
    $AUGMENT

if [ $? -ne 0 ]; then
    echo "Error: MRI T2 training failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "All training complete!"
echo "========================================="
echo "MRI T1 results: $T1_SAVE_PATH/"
echo "MRI T2 results: $T2_SAVE_PATH/"
echo ""
echo "Patient IDs and train/val splits saved in:"
echo "  - $T1_SAVE_PATH/patient_ids.json"
echo "  - $T2_SAVE_PATH/patient_ids.json"
echo "========================================="
