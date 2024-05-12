import os, sys
import time

import torch

import compute_iou
import utils


def fake_run_compute_iou(voxel_size, click_area):
    print(f"Running compute_iou with voxel_size={voxel_size} and click_area={click_area}")
    return 42.42

def run_compute_iou(voxel_size, click_area):
    start_time = time.time()
    iou_args = {'src_path': DATASET_DIR, 
                'model_path': MODEL_PATH, 
                'output_dir': f'{OUTPUT_DIR}/voxel_{voxel_size}_click_{click_area}',
                'show_3d': False,
                'inseg_model': inseg_model_class,
                'inseg_global': inseg_global_model,
                'limit_to_one_object': False, #True,
                'verbose': False,
                'max_imgs': 3,
                'click_area': voxel_size,
                'voxel_size': click_area}
    val_iou = compute_iou.main(iou_args)
    print(f'\ncompute_iou took {utils.timeit(start_time)}')
    return val_iou

DATASET_DIR="../dataset/S3DIS_converted_downsampled_new_mini/test"
OUTPUT_DIR="../results/testing_voxel_click_mini"
MODEL_PATH="../../models/InterObject3D_pretrained/weights_exp14_14.pth"
assert os.path.exists(MODEL_PATH), f"Model path does not exist. Choose a valid path to a model. ({MODEL_PATH})"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

inseg_model_class, inseg_global_model = utils.get_model(MODEL_PATH, device)

testing_data = [
    # [0.01, [0.01, 0.015, 0.02]],
    # [0.02, [0.02, 0.03, 0.04]],
    # [0.03, [0.03, 0.045, 0.06]],
    # [0.04, [0.04, 0.06, 0.08]],
    # [0.05, [0.05, 0.075, 0.1]],
    [0.05, [0.062, 0.087]],
    [0.06, [0.06, 0.09, 0.12]],
    [0.07, [0.07, 0.105, 0.14]],
]

for voxel_size, click_areas in testing_data:
    for click_area in click_areas:
        print(f'\n\n\nRunning compute_iou with voxel_size={voxel_size} and click_area={click_area}\n\n\n')
        iou = run_compute_iou(voxel_size, click_area)
        print(f'{voxel_size},{click_area},{iou}', file=open(f'{OUTPUT_DIR}/result_summary.txt', 'a'))

