import argparse
import open3d as o3d
import numpy as np
from data_loader import DataLoader

def main(src_path, force):
    data_loader = DataLoader(src_path, 5, 0.05, force)
    
    def visualize(visualizer):
        coords_batch, feats_batch, label_batch = data_loader.get_batch(1)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_batch.numpy()[0])
        
        colors = feats_batch.numpy()[0][:, :3]/255
        colors[label_batch.numpy()[0].reshape(-1) == 1] = [0, 1, 0] # Label
        colors[feats_batch.numpy()[0][:, 3] == 1] = [1, 0, 0] # maskPositive
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        visualizer.clear_geometries()
        visualizer.add_geometry(pcd)
        visualizer.reset_view_point(True)
    
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window()
    visualize(visualizer)
    visualizer.register_key_callback(o3d.visualization.gui.SPACE, visualize)
    visualizer.run()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/S3DIS_converted",
                        help="Source path (default: ../dataset/S3DIS_converted")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Refresh DataLoader cache")
    args = parser.parse_args()

    main(args.src_path, args.force)
    
    # coords_batch, feats_batch, label_batch = data_loader.get_batch(1)

    # print(f'coords_batch ({type(coords_batch)}): {coords_batch.shape}')
    # print(f'feats_batch ({type(feats_batch)}): {feats_batch.shape}')
    # print(f'label_batch ({type(label_batch)}): {label_batch.shape}')
 