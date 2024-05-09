import argparse

import torch
import open3d as o3d

from InterObject3D.interactive_adaptation import InteractiveSegmentationModel
from data_loader import DataLoader
import utils

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/S3DIS_converted_separated/test",
                        help="Source path (default: ../dataset/S3DIS_converted_separated/test)")
    parser.add_argument("-m", "--model_path", required=True,
                        help="Model path (required)")
    parser.add_argument('-o', '--output_dir', type=str, default='../results',
                        help='Where to store testing progress.')
    parser.add_argument("-3", "--show_3d", default=False, action='store_true',
                        help="Show 3D visualization of output models(default: False)")
    parser.add_argument("-i", "--inseg_model", default=None)
    parser.add_argument("-g", "--inseg_global", default=None)
    return parser.parse_args()

def main(args):
    if isinstance(args, dict):
        src_path = args['src_path']
        model_path = args['model_path']
        output_dir = args['output_dir']
        inseg_model = args['inseg_model']
        inseg_global = args['inseg_global']
        show_3d = args['show_3d']
    else:
        src_path = args.src_path
        model_path = args.model_path
        output_dir = args.output_dir
        inseg_model = args.inseg_model
        inseg_global = args.inseg_global
        show_3d = args.show_3d
    
    utils.ensure_folder_exists(output_dir)
    # print('Args:', args) # Debug print only
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = DataLoader(src_path, points_per_object=5, click_area=0.1, normalize_colors=True)


    # load model from path
    if (inseg_model == None):
        inseg_model_class, inseg_global_model = get_model(model_path, device)
    else:
        inseg_model_class = inseg_model
        inseg_global_model = inseg_global
    
    results = []

    i = 0

    while True:
        print(f'\nBatch {i}')
        batch = data_loader.get_random_batch()
        if not batch:
            break
        coords, feats, labels = batch
        coords = torch.tensor(coords).float().to(device)
        feats = torch.tensor(feats).float().to(device)
        labels = torch.tensor(labels).long().to(device)
        print(f'Batch: {coords.shape=}, {feats.shape=}, {labels.shape=}')

        print(f'inputs: feats({feats.shape})\n'
              f'        coords({coords.shape})')
        pred, logits = inseg_model_class.prediction(feats.float(), coords.cpu().numpy(), inseg_global_model, device)
        pred = torch.unsqueeze(pred, dim=-1)
        print(f'outputs: pred({pred.shape})\n'
              f'         logits({logits.shape})')
        print(f'labels: labels({labels.shape})')

        iou = inseg_model_class.mean_iou(pred, labels)
        print(f'iou: {iou}')

        output_point_cloud = get_output_point_cloud(coords, feats, labels, pred)
        if show_3d:
            o3d.visualization.draw_geometries([output_point_cloud])
        utils.save_point_cloud_views(output_point_cloud, iou, i, output_dir)
        results.append(iou)
        print(f'Mean iou so far: {sum(results) / len(results)}')
        i += 1

    # print result mean
    print(f'Mean IoU: {sum(results) / len(results)}')
    return sum(results) / len(results) if len(results) > 0 else 0

def get_model(pretrained_weights_file, device):
    inseg_global = InteractiveSegmentationModel(pretraining_weights=pretrained_weights_file)
    global_model = inseg_global.create_model(device, inseg_global.pretraining_weights_file)
    return inseg_global, global_model

def get_output_point_cloud(coords, feats, labels, pred):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(coords.cpu().numpy())

    colors = feats.cpu().numpy()[:, :3]
    colors[labels.cpu().numpy().reshape(-1) == 1] = [0, 1, 0] # Label GREEN
    colors[pred.cpu().numpy()[:, 0] == 1] = [1, 0, 0] # maskPositive output RED
    colors[feats.cpu().numpy()[:, 3] == 1] = [0, 0, 1] # maskPositive input BLUE
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


if __name__ == "__main__":
    args = parseargs()
    main(args)
