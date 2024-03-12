import open3d as o3d
import os
import argparse

def process_pointcloud(path):
    pcd = o3d.io.read_point_cloud(path, format='xyzrgb')

    # Original file has RGB in [0, 255] range
    # it need to be converted to [0, 1] range
    for i in range(len(pcd.points)):
        pcd.colors[i] /= 255

    # Calculate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    return pcd

def process_dataset(src, dst):
    # Create a list containing path of every area from dataset
    areas = [f.path for f in os.scandir(src) if f.is_dir()]
    areas = [f.path for subfolder in areas for f in os.scandir(subfolder) if f.is_dir()]
    
    # Process each area
    for i, area in enumerate(areas):
        print(f"{i+1}/{len(areas)} - {area.split('/')[-2]}_{area.split('/')[-1]}")
        
        # Skip if folder already exists
        dst_path = os.path.join(dst, f"{area.split('/')[-2]}_{area.split('/')[-1]}")
        if os.path.exists(dst_path):
            print(f"\tFolder already exists, skipping")
            continue
        os.mkdir(dst_path)

        # Original pointcloud
        pcd = process_pointcloud(os.path.join(area, f"{area.split('/')[-1]}.txt"))
        o3d.io.write_point_cloud(
            filename=os.path.join(dst_path, "original.pcd"),
            pointcloud=pcd)

        # Annotations
        os.mkdir(os.path.join(dst_path, "annotations"))
        os.mkdir(os.path.join(dst_path, "annotations_downsampled"))
        for filename in os.listdir(os.path.join(area, "Annotations")):
            # skip .DS_Store file
            if not filename.endswith(".txt"):
                continue
            
            # Original annotation
            pcd = process_pointcloud(os.path.join(area, "Annotations", filename))
            o3d.io.write_point_cloud(
                filename=os.path.join(dst_path, "annotations", f"{filename.split('.')[0]}.pcd"),
                pointcloud=pcd)

            # Downsampled annotation
            pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.1)
            o3d.io.write_point_cloud(
                filename=os.path.join(dst_path, "annotations_downsampled", f"{filename.split('.')[0]}.pcd"),
                pointcloud=pcd_downsampled)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/Stanford3dDataset_v1.2", help="Source path (default: ../dataset/Stanford3dDataset_v1.2)")
    parser.add_argument("-d", "--dst_path", default="../dataset/S3DIS_converted", help="Destination path (default: ../dataset/S3DIS_converted)")
    args = parser.parse_args()
    
    if not os.path.exists(args.src_path):
        print("Source path does not exist")
        exit(1)
    if not os.path.exists(args.dst_path):
        os.mkdir(args.dst_path)
        
    process_dataset(args.src_path, args.dst_path)
