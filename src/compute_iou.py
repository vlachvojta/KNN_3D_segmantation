import argparse

import torch
import open3d as o3d

from InterObject3D.interactive_adaptation import InteractiveSegmentationModel
from data_loader import DataLoader
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/S3DIS_converted",
                        help="Source path (default: ../dataset/S3DIS_converted")
    parser.add_argument("-m", "--model_path", required=True,
                        help="Model path (required)")
    args = parser.parse_args()

    print('Args:', args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = DataLoader(args.src_path, points_per_object=5)

    # load model from path
    inseg_model_class, inseg_global_model = get_model(args.model_path, device)

    results = []

    i = 0

    while True:
        print(f'\nBatch {i}')
        batch = data_loader.get_batch(1)
        if not batch:
            break
        coords, feats, labels = batch
        coords = coords[0]
        feats = feats[0]
        labels = labels[0].float().long().to(device)
        print(f'Batch: {coords.shape=}, {feats.shape=}, {labels.shape=}')

        print(f'inputs: feats({feats.shape})\n'
              f'        coords({coords.shape})')
        pred, logits = inseg_model_class.prediction(feats, coords, inseg_global_model, device)
        pred = torch.unsqueeze(pred, dim=-1)
        print(f'outputs: pred({pred.shape})\n'
              f'         logits({logits.shape})')
        print(f'labels: labels({labels.shape})')

        iou = inseg_model_class.mean_iou(pred, labels)
        print(f'iou: {iou}')

        utils.visualize_prediction_result(coords, feats, labels, pred, iou, results, i)

        results.append(iou)
        i += 1

    # print result mean
    print(f'Mean IoU: {sum(results) / len(results)}')


def get_model(pretrained_weights_file, device):
    inseg_global = InteractiveSegmentationModel(pretraining_weights=pretrained_weights_file)
    global_model = inseg_global.create_model(device, inseg_global.pretraining_weights_file)
    return inseg_global, global_model


if __name__ == "__main__":
    main()
