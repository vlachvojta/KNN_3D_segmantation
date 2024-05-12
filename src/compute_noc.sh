#!/bin/bash

for model_dir in InterObject3D_pretrained OUR_click_0.1 OUR_downsampled_click_0.1 OUR_downsampled_click_0.7_voxel_0.7; do
    model=`ls ../training_result_models/$model_dir/*.pth`

    for noc in 80 85 90; do
        OUTPUT_DIR="../results/compute_noc_downsampled/$model_dir/kIOU_$noc"
        mkdir -p $OUTPUT_DIR
        
        echo ""
        echo ""
        echo ""
        echo "=================================================================================================="
        echo "Computing NOC for kIOU=$noc using model: $model"
        echo "Saving results to: $OUTPUT_DIR"
        echo "=================================================================================================="
        echo ""
        echo ""
        echo ""

        python compute_noc.py \
            -m $model \
            -o $OUTPUT_DIR \
            -s ../dataset/S3DIS_converted_downsampled_new/test/ \
            --max_imgs 3 \
            --max_clicks 20 \
            -c 0.07 \
            -vs 0.07 \
            --k_iou $noc \
            2>&1 | tee -a $OUTPUT_DIR/compute_noc.log
    done
done
