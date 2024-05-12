import open3d as o3d
import os
import argparse
import numpy as np
import pickle


def process_dataset(src, dst):
    # Create a list containing path of every area from dataset
    areas = [f.path for f in os.scandir(src) if f.is_dir()]
    areas = [f.path for subfolder in areas for f in os.scandir(subfolder) if f.is_dir()]
    
    classes = {}

    # Process each area
    for i, area in enumerate(areas):
        print(f"{i+1}/{len(areas)} - {area.split('/')[-2]}_{area.split('/')[-1]}")

        area_name = f"{area.split('/')[-2]}_{area.split('/')[-1]}.pcd"
        classes[area_name] = {}

        group = 0
        for filename in os.listdir(os.path.join(area, "Annotations")):
            # skip .DS_Store file
            if not filename.endswith(".txt"):
                continue
            
            class_name = filename.split('_')[0]
            # if not class_name in classes[area_name].keys():
            #     classes[area_name][class_name] = []
            # classes[area_name][class_name].append(group)
            classes[area_name][group] = class_name

            group += 1

    # Save classes to pickle file
    with open(os.path.join(dst, "classes.pkl"), 'wb') as f:
        pickle.dump(classes, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/Stanford3dDataset_v1.2",
                        help="Source path (default: ../dataset/Stanford3dDataset_v1.2)")
    parser.add_argument("-d", "--dst_path", default="../dataset/",
                        help="Destination path (default: ../dataset/)")
    args = parser.parse_args()

    if not os.path.exists(args.src_path):
        print("Source path does not exist")
        exit(1)
    if not os.path.exists(args.dst_path):
        os.mkdir(args.dst_path)

    process_dataset(args.src_path, args.dst_path)
