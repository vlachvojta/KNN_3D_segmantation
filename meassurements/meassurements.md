# Meassurements

## IOU from compute_iou - WITH OLD DATALOADER

### click area 0.1
- original dataset: 29.6 % after 1247 batches
- downsampled: 32.2 % after 530 batches

### click area 0.05
- original dataset: 34.6 % after 400 batches
- downsampled: 32.5 % after 1618 batches

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
