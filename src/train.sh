#!/bin/bash

PRETRAINED_MODEL_PATH="../../models/InterObject3D_pretrained/weights_exp14_14.pth"
OUTPUT_DIR="../training/InterObject3D_BatchTraining_GPU"
mkdir -p $OUTPUT_DIR

# Train the model
python -u train.py \
    -d "../dataset/S3DIS_converted_separated/train" \
    -m $PRETRAINED_MODEL_PATH \
    -o $OUTPUT_DIR \
    -b 20 \
    2>&1 | tee -a $OUTPUT_DIR/train.log
