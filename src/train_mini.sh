#!/bin/bash

PRETRAINED_MODEL_PATH="../../models/InterObject3D_pretrained/weights_exp14_14.pth"
OUTPUT_DIR="../training_new_new/InterObject3D_downsampled_voxel_0.04_click_0.05_mini_2"
DATASET_DIR="../dataset/S3DIS_converted_downsampled_new_mini"
# DATASET_DIR="../dataset/S3DIS_converted_separated"
mkdir -p $OUTPUT_DIR

# Train the model
python -u train.py \
    -d $DATASET_DIR/train \
    -vd $DATASET_DIR/val \
    --validation_out $OUTPUT_DIR/val_results \
    --stats_path $OUTPUT_DIR \
    --voxel_size 0.04 \
    --click_area 0.05 \
    -m $PRETRAINED_MODEL_PATH \
    -o $OUTPUT_DIR \
    --test_step 25 \
    --save_step 50 \
    --saved_loss $OUTPUT_DIR/train_losses.npy \
    --saved_ious_val $OUTPUT_DIR/val_ious.npy \
    --saved_ious_train $OUTPUT_DIR/train_ious.npy \
    -b 4 \
    2>&1 | tee -a $OUTPUT_DIR/train.log
