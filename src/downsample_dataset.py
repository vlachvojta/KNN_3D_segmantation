import open3d as o3d
import os
import argparse
import numpy as np


def downsample_dataset(src, dst, value):
    areas = [f.path for f in os.scandir(src) if f.path.endswith(".pcd")]
    for i, area in enumerate(areas):
        print(f"{i+1}/{len(areas)} - {area.split('/')[-1]}")
        
        pcd = o3d.t.io.read_point_cloud(area)
        pcd = pcd.voxel_down_sample(voxel_size=value)
        
        o3d.t.io.write_point_cloud(os.path.join(dst, area.split('/')[-1]), pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/S3DIS_converted",
                        help="Source path (default: ../dataset/S3DIS_converted")
    parser.add_argument("-d", "--dst_path", default="../dataset/S3DIS_converted_downsampled",
                        help="Destination path (default: ../dataset/S3DIS_converted_downsampled)")
    parser.add_argument("-v", "--downsample_value", type=float, default=0.05, help="Value for downsampling (default: 0.05)")
    args = parser.parse_args()

    if not os.path.exists(args.src_path):
        print("Source path does not exist")
        exit(1)
    if not os.path.exists(args.dst_path):
        os.mkdir(args.dst_path)

    downsample_dataset(args.src_path, args.dst_path, args.downsample_value)
