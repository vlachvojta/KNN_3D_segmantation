#!/bin/bash

for model_dir in InterObject3D_pretrained OUR_click_0.1 OUR_downsampled_click_0.1 OUR_downsampled_click_0.7_voxel_0.7; do
    model=`ls ../training_result_models/$model_dir/*.pth`

    OUTPUT_DIR="../results/compute_iou_downsampled/$model_dir/"
    mkdir -p $OUTPUT_DIR

    echo ""
    echo ""
    echo ""
    echo "=================================================================================================="
    echo "Computing IOU using model: $model"
    echo "Saving results to: $OUTPUT_DIR"
    echo "=================================================================================================="
    echo ""
    echo ""
    echo ""

    python compute_iou.py \
        -m $model \
        -o $OUTPUT_DIR \
        -s ../dataset/S3DIS_converted_downsampled_new/test/ \
        --max_imgs 3 \
        --limit_to_one_object \
        -c 0.07 \
        -v \
        -vs 0.07 \
        2>&1 | tee -a $OUTPUT_DIR/compute_iou.log
done

# InterObject3D_pretrained/weights_exp14_14.pth OUR_click_0.1/model_450.pth OUR_downsampled_click_0.1/model_100.pth OUR_downsampled_click_0.7_voxel_0.7/MinkUNet34C_10.pth