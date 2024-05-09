import open3d as o3d
import os
import argparse
import numpy as np


def downsample_dataset(src, dst, value):
    areas = [f.path for f in os.scandir(src) if f.path.endswith(".pcd")]
    for i, area in enumerate(areas):
        print(f"{i+1}/{len(areas)} - {area.split('/')[-1]}")
        
        pcd = o3d.t.io.read_point_cloud(area)
        pcd = pcd.uniform_down_sample(every_k_points=value)
        
        del pcd.point.normals
        
        area_number = area.split('/')[-1][5]
        if area_number == 5:
            folder = "test"
        elif area_number == 6:
            folder = "val"
        else:
            folder = "train"
        
        o3d.t.io.write_point_cloud(os.path.join(dst, folder, area.split('/')[-1]), pcd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/S3DIS_converted",
                        help="Source path (default: ../dataset/S3DIS_converted")
    parser.add_argument("-d", "--dst_path", default="../dataset/S3DIS_converted_downsampled",
                        help="Destination path (default: ../dataset/S3DIS_converted_downsampled)")
    parser.add_argument("-k", "--downsample_value", type=int, default=5, help="Value for downsampling, every k point (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.src_path):
        print("Source path does not exist")
        exit(1)
    if not os.path.exists(args.dst_path):
        os.mkdir(args.dst_path)
    if not os.path.exists(os.path.join(args.dst_path, "train")):
        os.mkdir(os.path.join(args.dst_path, "train"))
    if not os.path.exists(os.path.join(args.dst_path, "test")):
        os.mkdir(os.path.join(args.dst_path, "test"))
    if not os.path.exists(os.path.join(args.dst_path, "val")):
        os.mkdir(os.path.join(args.dst_path, "val"))

    downsample_dataset(args.src_path, args.dst_path, args.downsample_value)
