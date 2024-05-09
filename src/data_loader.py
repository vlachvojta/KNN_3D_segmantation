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
    def __init__(self, data_path, points_per_object=5, click_area=0.05, downsample=0, force=False, verbose=True, normalize_colors=False, limit_to_one_object=False):
        self.data_path = data_path
        self.points_per_object = points_per_object
        self.click_area = click_area
        self.downsample = downsample
        self.force = force
        self.verbose = verbose
        self.normalize_colors = normalize_colors

        assert os.path.exists(data_path), "Data path does not exist. Choose a valid path to a dataset."

        # self.cache_path = os.path.join(data_path, "dataloader_cache")
        self.cache_path = os.path.join(data_path, "dataloader_cache_ppo" + str(points_per_object) + "_ca" + str(click_area) + "_ds" + str(downsample) + ".pkl")

        # Load from cache
        if os.path.exists(self.cache_path):
            if force:
                os.remove(self.cache_path)
            else:
                self.data = self.load_from_cache(self.cache_path)
                self.len = self.remaining_unique_elements()
                return

        self.data = {}   

        print(f'\nCreating DataLoader with points_per_object={points_per_object} and click_area={click_area} and downsample={downsample}, processing {len([f for f in os.scandir(data_path)])} files.')
        # Process each area
        for i, file in enumerate([f.path for f in os.scandir(data_path) if f.path.endswith('.pcd')]):
            if verbose:
                print(f"Processing {file}")
            else:
                if i % 50 == 0 and i != 0:
                    print('')
                print(".", end="", flush=True)
            self.data[file] = []

            # Load pointcloud and split into groups (objects)
            pcd = o3d.t.io.read_point_cloud(file)
            
            if downsample != 0:
                pcd = pcd.voxel_down_sample(voxel_size=0.05)
                
            groups = list(pcd.point.group.flatten().numpy())
            groups = [list(i) for _, i in groupby(groups)]

            if limit_to_one_object:
                groups = [random.choice(groups)]

            # Simulate clicked points for each group
            offset = 0
            for group in groups:
                # Select every k point from each object
                # First and last point are skipped
                points = [offset + (i * (len(group) // (points_per_object + 2)))
                          for i in range(1, 1+points_per_object)]

                self.data[file].append(points)
                offset += len(group)
        print('')

        self.len = self.remaining_unique_elements()

        # Save to cache
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.data, f)

    def __getitem__(self, index):
        return self.get_random_batch()

    def get_random_batch(self):
        # coords_list = []
        # feats_list = []
        # label_list = []

        # for _ in range(batch_size):
            # Select random area, object and point
        random_area = random.choice(list(self.data.keys()))
        random_object = random.randint(0, len(self.data[random_area])-1)
        random_point = random.randint(0, len(self.data[random_area][random_object]) - 1)

        if self.verbose:
            print(f"Simulated click - {random_area.split('/')[-1]}/object {random_object}/point {self.data[random_area][random_object][random_point]}")

        # Load pointcloud and simulate positive click in maskPositive
        pcd = o3d.t.io.read_point_cloud(random_area)
        if self.downsample != 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.downsample)
    
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
        feats = np.concatenate((pcd.point.colors.numpy(), pcd.point.maskPositive.numpy(), pcd.point.maskNegative.numpy()), axis=1, dtype=np.float32)

        # coords_list.append(coords)
        # feats_list.append(feats)
        # label_list.append(label)

        # Remove already simulated point from data
        del self.data[random_area][random_object][random_point]
        if not self.data[random_area][random_object]:
            del self.data[random_area][random_object]
            if not self.data[random_area]:
                del self.data[random_area]
                if not self.data:
                    # Every point has been processed
                    print("DataLoader: All points have been processed. Returning None.")
                    return None

        # Concatenate the lists of numpy arrays
        # coords_batch = self.list_to_batch(coords_list, torch.float32)
        # feats_batch = self.list_to_batch(feats_list, torch.float32)
        # label_batch = self.list_to_batch(label_list, torch.uint8)
        
        if self.normalize_colors:
            feats[:, :3] = feats[:, :3] / 255

        # Return the concatenated arrays
        return coords, feats, label

    def list_to_batch(self, clouds, dtype):
        max_len = max([len(cloud) for cloud in clouds])
        clouds = [np.pad(cloud, ((0, max_len - len(cloud)), (0, 0)), 'constant', constant_values=0) 
                  for cloud in clouds]
        batch = np.stack(clouds, axis=0)
        return torch.tensor(batch, dtype=dtype)

    def new_epoch(self):
        assert os.path.exists(self.cache_path), "Cache not found."
        self.data = self.load_from_cache(self.cache_path)

    def remaining_unique_elements(self):
        return sum(len(area) for areas in self.data.values() for area in areas)

    def __len__(self):
        return self.len

    @staticmethod
    def load_from_cache(cache_path):
        print(f'Loading data from cache: {cache_path}')
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        return data

