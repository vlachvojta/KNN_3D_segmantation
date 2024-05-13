# Meassurements

## Výsledky IOU pro 3 body 
| Model                   | IOU total | board | bookcase | ceiling | clutter | table | wall  | window | ... |
|-------------------------|-----------|-------|----------|---------|---------|-------|-------|--------|-----|
| Pretrained              | 41.87     | 36.70 | 40.36    | 69.70   | 37.22   | 40.25 | 48.08 | 26.39  | ... |
| OUR_voxel_0.5_click_0.1 | 39.50     | 38.15 | 40.71    | 65.60   | 35.24   | 38.96 | 45.54 | 24.72  | ... |
| OUR_voxel_0.7_click_0.7 | 44.10     | 34.31 | 38.76    | 69.14   | 41.54   | 40.73 | 45.78 | 26.46  | ... |


## NoC k IOU
|                          | 80 % IOU | 85 % IOU  |
|--------------------------|----------|-----------|
| InterObject3D_pretrained | 12.96    | 14.76     |
| OUR_voxel_0.5_click_0.1  | 10.29    | 11.77     |
| OUR_voxel_0.7_click_0.7  | **9.5**  | **11.01** |


## IOU from compute_iou - WITH NEW dataloader

### click area 0.1 OUR MODEL InterObject3D_BatchTraining/model_450.pth
- 31.3 % IOU after 650 batches

### click area 0.1 OUR MODEL InterObject3D_downsampled_click_0.5/model_100.pth
- 33.1 % IOU after 650 batches

### click area 0.1 ORIGINAL model
- 30.5 % IOU after 650 batches


### click area 0.05 OUR MODEL InterObject3D_BatchTraining/model_450.pth
- 22.8 % IOU after 650 batches

### click area 0.05 OUR MODEL InterObject3D_downsampled_click_0.5/model_100.pth
- 28.6 % IOU after 650 batches

### click area 0.05 ORIGINAL model
- 32.2 % IOU after 650 batches
