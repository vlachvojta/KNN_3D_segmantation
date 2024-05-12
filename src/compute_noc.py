import argparse

import torch
import open3d as o3d

from InterObject3D.interactive_adaptation import InteractiveSegmentationModel
from data_loader import DataLoader
import utils

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-s", "--src_path", default="../dataset/S3DIS_converted_separated/test",
                        help="Source path (default: ../dataset/S3DIS_converted_separated/test)")
    parser.add_argument("-m", "--model_path", required=True,
                        help="Model path (required)")
    parser.add_argument('-o', '--output_dir', type=str, default='../results',
                        help='Where to store testing progress.')
    parser.add_argument("-3", "--show_3d", default=False, action='store_true',
                        help="Show 3D visualization of output models(default: False)")
    parser.add_argument("-d", "--downsample",  type=int, default=0,
                        help="Downsample value, every k point (default: 0 = no downsampling)")
    parser.add_argument("-l", "--limit_to_one_object", action='store_true',
                        help="Limit objects in one room to one random object (default: False).")
    parser.add_argument("-v", "--verbose", default=True)
    parser.add_argument("-mi", "--max_imgs",  type=int, default=20,
                        help="Number of maximum saved image samples (default: 20)")
    parser.add_argument("-c", "--click_area",  type=float, default=0.1,
                        help="Click area (default: 0.1)")
    parser.add_argument("-mc", "--max_clicks",  type=int, default=15,
                        help="Number of maximum clicks (default: 15)")
    parser.add_argument("-i", "--k_iou",  type=float, default=80.0,
                        help="Minimum IOU treshold (default: 80%)")
    parser.add_argument("-vs", "--voxel_size", default=0.05, type=float,
                        help="The size data points are converting to (default: 0.05)")
    
    args = parser.parse_args()

    utils.ensure_folder_exists(args.output_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_loader = DataLoader(args.src_path, click_area=args.click_area, normalize_colors=True, verbose=args.verbose, downsample=args.downsample, limit_to_one_object=args.limit_to_one_object, n_of_clicks=args.max_clicks)

    print(f'{len(data_loader)} elements in data loader')
    
    inseg_model_class, inseg_global_model = utils.get_model(args.model_path, device)
    
    results = []
    
    i = 0
    
    while True:
        batch = data_loader.get_batch_with_clicks(1)
        if not batch:
            break
        
        for clicks in range(1, args.max_clicks+1):
            # print(f'\nSegmented object no.: {i}')

            if (clicks != 1):
                batch = data_loader.get_batch_with_clicks(clicks)
            coords, feats, labels = batch 
            coords = torch.tensor(coords).float().to(device)
            feats = torch.tensor(feats).float().to(device)
            labels = torch.tensor(labels).long().to(device)

            pred, logits = inseg_model_class.prediction(feats.float(), coords.cpu().numpy(), inseg_global_model, device, voxel_size=args.voxel_size)
            pred = torch.unsqueeze(pred, dim=-1)

            iou = inseg_model_class.mean_iou(pred, labels).cpu()
            # print(f'\niou: {iou}')
            
            if iou >= args.k_iou:
                print(f'Segmented object no.: {i}')
                print(f'iou: {iou}')
                print(f'NOC: {clicks}')
                
                results.append(clicks)
                print(f'Mean NOC so far: {sum(results) / len(results)}\n')
                
                if i < args.max_imgs:
                    output_point_cloud = utils.get_output_point_cloud(coords, feats, labels, pred)
                    if args.show_3d:
                        o3d.visualization.draw_geometries([output_point_cloud])
                    utils.save_point_cloud_views(output_point_cloud, iou, i, args.output_dir, args.verbose)
                    
                data_loader.next_random_batch()
                break
            elif (clicks == args.max_clicks):
                if i < args.max_imgs:
                    output_point_cloud = utils.get_output_point_cloud(coords, feats, labels, pred)
                    if args.show_3d:
                        o3d.visualization.draw_geometries([output_point_cloud])
                    utils.save_point_cloud_views(output_point_cloud, iou, i, args.output_dir, args.verbose)
                
                results.append(clicks)
                data_loader.next_random_batch()
                
                print(f'Segmented object no.: {i}')
                print(f'iou: {iou}')
                
                print(f'Mean NOC so far: {sum(results) / len(results)}\n')        
        i += 1

    print(f'Mean NOC: {sum(results) / len(results)}')
    print(f'{args.k_iou},{sum(results) / len(results):.4f}', file=open(f'{args.output_dir}/../result.txt', 'a'))


if __name__ == "__main__":
    main()
