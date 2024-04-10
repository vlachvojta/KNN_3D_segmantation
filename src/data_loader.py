import os
import open3d as o3d
import pickle
import random
import time
from itertools import groupby

import numpy as np
import torch

random.seed(time.time())


class DataLoader:
    def __init__(self, data_path, points_per_object=5, click_area=0.05, force=False):
        assert os.path.exists(data_path), "Data path does not exist. Choose a valid path to a dataset."

        self.cache_path = os.path.join(data_path, "dataloader_cache")
        self.click_area = click_area

        # Load from cache
        if os.path.exists(self.cache_path):
            if force:
                os.remove(self.cache_path)
            else:
                with open(self.cache_path, 'rb') as f:
                    self.data = pickle.load(f)
                return

        self.data = {}

        # Process each area
        for file in [f.path for f in os.scandir(data_path)]:
            print(f"Processing {file}")
            self.data[file] = []

            # Load pointcloud and split into groups (objects)
            pcd = o3d.t.io.read_point_cloud(file)
            groups = list(pcd.point.group.flatten().numpy())
            groups = [list(i) for _, i in groupby(groups)]

            # Simulate clicked points for each group
            offset = 0
            for group in groups:
                # Select every k point from each object
                # First and last point are skipped
                points = [offset + (i * (len(group) // (points_per_object + 2)))
                          for i in range(1, 1+points_per_object)]

                self.data[file].append(points)
                offset += len(group)

        # Save to cache
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.data, f)

    def get_batch(self, batch_size=5):
        coords_batch = []
        feats_batch = []
        label_batch = []

        for _ in range(batch_size):
            # Select random area, object and point
            random_area = random.choice(list(self.data.keys()))
            random_object = random.randint(0, len(self.data[random_area])-1)
            random_point = random.randint(0, len(self.data[random_area][random_object]) - 1)

            print(f"Simulated click - {random_area.split('/')[-1]}/object {random_object}/point {self.data[random_area][random_object][random_point]}")

            # Load pointcloud and simulate positive click in maskPositive
            pcd = o3d.t.io.read_point_cloud(random_area)
        
            # Create a copy of the pointcloud (KDTreeFlann doesn't support o3d.t.geometry.PointCloud or idk)
            # It's ugly but it works
            pcd_tree = o3d.geometry.PointCloud()
            pcd_tree.points = o3d.utility.Vector3dVector(pcd.point.positions.numpy())
            tree = o3d.geometry.KDTreeFlann(pcd_tree)
            [_, idx, _] = tree.search_radius_vector_3d(pcd_tree.points[self.data[random_area][random_object][random_point]], self.click_area)
            for i in idx:
                pcd.point.maskPositive[i] = 1

            # Create a mask with the same group as the clicked point
            group = pcd.point.group[self.data[random_area][random_object][random_point]].numpy()[0]
            label = (pcd.point.group.numpy() == group)
            label = o3d.core.Tensor(label, o3d.core.uint8, o3d.core.Device("CPU:0")).numpy() #(dtype=np.int8)
            del pcd.point.group

            # Add tuple of pointcloud and label to batch
            coords = pcd.point.positions.numpy()
            feats = np.concatenate((pcd.point.colors.numpy(), pcd.point.maskPositive.numpy(), pcd.point.maskNegative.numpy()), axis=1)

            coords_batch.append(coords)
            feats_batch.append(feats)
            label_batch.append(label)

            # Remove already simulated point from data
            del self.data[random_area][random_object][random_point]
            if not self.data[random_area][random_object]:
                del self.data[random_area][random_object]
                if not self.data[random_area]:
                    del self.data[random_area]
                    if not self.data:
                        # Every point has been processed
                        return None

        # Concatenate the lists of numpy arrays
        coords_batch = np.stack(coords_batch, axis=0)
        feats_batch = np.stack(feats_batch, axis=0)
        label_batch = np.stack(label_batch, axis=0)

        feats_batch = torch.tensor(feats_batch, dtype=torch.float32)
        coords_batch = torch.tensor(coords_batch, dtype=torch.float32)
        label_batch = torch.tensor(label_batch, dtype=torch.uint8)

        # Return the concatenated arrays
        return coords_batch, feats_batch, label_batch
