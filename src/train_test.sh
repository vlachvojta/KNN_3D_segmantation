#!/bin/bash

PRETRAINED_MODEL_PATH="../../models/InterObject3D_pretrained/weights_exp14_14.pth"
# OUTPUT_DIR="../training_new_new/InterObject3D_downsampled_voxel_0.07_click_0.07"
OUTPUT_DIR="../training_from_scratch/MinkUnet34C_voxel_0.07_click_0.07"
DATASET_DIR="../dataset/S3DIS_converted_downsampled_new"
# DATASET_DIR="../dataset/S3DIS_converted_separated"
mkdir -p $OUTPUT_DIR

# Train the model
python -u train.py \
    -d $DATASET_DIR/train \
    -vd $DATASET_DIR/val \
    --validation_out $OUTPUT_DIR/val_results \
    --stats_path $OUTPUT_DIR \
    --voxel_size 0.07 \
    --click_area 0.07 \
    -o $OUTPUT_DIR \
    --test_step 10 \
    --save_step 50 \
    --saved_loss $OUTPUT_DIR/train_losses.npy \
    --saved_ious_val $OUTPUT_DIR/val_ious.npy \
    --saved_ious_train $OUTPUT_DIR/train_ious.npy \
    -b 12 \
    2>&1 | tee -a $OUTPUT_DIR/train.log
    # -m $PRETRAINED_MODEL_PATH \
