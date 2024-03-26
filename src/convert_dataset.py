import open3d as o3d
import os
import argparse
import numpy as np


def process_dataset(src, dst):
    # Create a list containing path of every area from dataset
    areas = [f.path for f in os.scandir(src) if f.is_dir()]
    areas = [f.path for subfolder in areas for f in os.scandir(subfolder) if f.is_dir()]

    # Process each area
    for i, area in enumerate(areas):
        print(f"{i+1}/{len(areas)} - {area.split('/')[-2]}_{area.split('/')[-1]}")

        dst_path = os.path.join(dst, f"{area.split('/')[-2]}_{area.split('/')[-1]}.pcd")

        # Skip if file already exists
        if os.path.exists(dst_path):
            print(f"\tFile already exists, skipping")
            continue

        merged_pcd = o3d.t.geometry.PointCloud()
        group = 0

        # Annotations
        for filename in os.listdir(os.path.join(area, "Annotations")):
            # skip .DS_Store file
            if not filename.endswith(".txt"):
                continue

            # Read pointcloud (it needs to be o3d.io, o3d.t.io doesn't support loading from txt)
            # o3d.t.geometry.PointCloud - [Open3D WARNING] The format of txt is not supported
            pcd = o3d.io.read_point_cloud(os.path.join(area, "Annotations", filename),
                                          format='xyzrgb')

            # Original file has RGB in [0, 255] range
            # it need to be converted to [0, 1] range
            for i in range(len(pcd.points)):
                pcd.colors[i] /= 255

            # Convert it to o3d.t.geometry.PointCloud
            pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)

            # Add group label and masks
            size = len(pcd.point.positions)

            pcd.point.group = o3d.core.Tensor(
                np.full((size, 1), group), o3d.core.uint8, o3d.core.Device("CPU:0"))
            pcd.point.maskPositive = o3d.core.Tensor(
                np.zeros((size, 1)), o3d.core.uint8, o3d.core.Device("CPU:0")).reshape((size, 1))
            pcd.point.maskNegative = o3d.core.Tensor(
                np.zeros((size, 1)), o3d.core.uint8, o3d.core.Device("CPU:0")).reshape((size, 1))

            merged_pcd = pcd if group == 0 else merged_pcd + pcd
            group += 1

        # Calculate normals
        merged_pcd.estimate_normals()

        # Save
        o3d.t.io.write_point_cloud(dst_path, merged_pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/Stanford3dDataset_v1.2",
                        help="Source path (default: ../dataset/Stanford3dDataset_v1.2)")
    parser.add_argument("-d", "--dst_path", default="../dataset/S3DIS_converted",
                        help="Destination path (default: ../dataset/S3DIS_converted)")
    args = parser.parse_args()

    if not os.path.exists(args.src_path):
        print("Source path does not exist")
        exit(1)
    if not os.path.exists(args.dst_path):
        os.mkdir(args.dst_path)

    process_dataset(args.src_path, args.dst_path)
