import os
import open3d as o3d
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch

random.seed(time.time())


class DataLoader:
    def __init__(self, data_path, click_area=0.05, downsample=0, force=False, 
                 verbose=True, normalize_colors=False, limit_to_one_object=False, voxel_size=0, n_of_clicks=0):
        self.data_path = data_path
        self.click_area = click_area
        self.downsample = downsample
        self.force = force
        self.verbose = verbose
        self.normalize_colors = normalize_colors
        self.voxel_size = voxel_size
        self.n_of_clicks = n_of_clicks

        assert os.path.exists(data_path), "Data path does not exist. Choose a valid path to a dataset."
        
        self.selected_object = None
        self.last_class = None

        # self.cache_path = os.path.join(data_path, "dataloader_cache")
        self.cache_path = os.path.join(data_path, "dataloader_cache_ds" + str(downsample) + "_nc" + str(n_of_clicks) + ".pkl")

        # Load from cache
        classes_path = os.path.join("..", "dataset", "classes.pkl")
        if os.path.exists(classes_path):
            self.classes = self.load_from_cache(classes_path)
        else:
            self.classes = None

        if os.path.exists(self.cache_path):
            if force:
                os.remove(self.cache_path)
            else:
                self.data = self.load_from_cache(self.cache_path)
                self.len = self.remaining_unique_elements()
                return

        self.data = {}

        print(f'\nCreating DataLoader with click_area={click_area} and downsample={downsample} and n of clicks={n_of_clicks}, processing {len([f for f in os.scandir(data_path)])} files.')
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
                pcd = pcd.uniform_down_sample(every_k_points=downsample)
                
            groups = list(pcd.point.group.flatten().numpy())
            groups = list(defaultdict(list, {val: [i for i, v in enumerate(groups) if v == val] for val in set(groups)}).values())

            if limit_to_one_object:
                groups = [random.choice(groups)]
                
            # Simulate clicked points for each group
            for group in groups:     
                # Select every k point from each object
                # First and last point are skipped
                
                if self.n_of_clicks == 0:
                    points = [group[(i * (len(group) // 11))] for i in range(1, 10)]
                    random.shuffle(points)
                    # groups of 1,1,2,2,3 clicks
                    points = [points[0:1], points[1:2], points[2:4], points[4:6], points[6:9]]
                    random.shuffle(points)
                else:
                    points = [[group[(i * (len(group) // (self.n_of_clicks + 2)))]] for i in range(1, self.n_of_clicks + 1)]
                    random.shuffle(points)
                self.data[file].append(points)
                
            random.shuffle(self.data[file])
            
        print('')

        self.len = self.remaining_unique_elements()

        # Save to cache
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.data, f)

    def __getitem__(self, index):
        return self.get_random_batch()
    
    def process_click(self, points, area):
        # Load pointcloud
        pcd = o3d.t.io.read_point_cloud(area)
        if self.downsample != 0:
            pcd = pcd.uniform_down_sample(every_k_points=self.downsample)
        
    
        # Create a copy of the pointcloud (KDTreeFlann doesn't support o3d.t.geometry.PointCloud or idk)
        # It's ugly but it works
        pcd_tree = o3d.geometry.PointCloud()
        pcd_tree.points = o3d.utility.Vector3dVector(pcd.point.positions.numpy())
        tree = o3d.geometry.KDTreeFlann(pcd_tree)
    
        # Create a click area for each simulated point
        for point in points:
            [_, idx, _] = tree.search_radius_vector_3d(pcd_tree.points[point], self.click_area)
            for i in idx:
                pcd.point.maskPositive[i] = 1


        # Get group id for label
        group = pcd.point.group[points].numpy()[0]
        
        if self.classes:
            self.last_class = self.classes[area.split('/')[-1]][group[0]]

        # Create a mask with the same group as the clicked point
        label = (pcd.point.group.numpy() == group)
        label = o3d.core.Tensor(label, o3d.core.uint8, o3d.core.Device("CPU:0")).numpy() #(dtype=np.int8)

        # Add tuple of pointcloud and label to batch
        coords = pcd.point.positions.numpy()
        if self.voxel_size > 0:
            coords = coords / self.voxel_size

        feats = np.concatenate((pcd.point.colors.numpy(), pcd.point.maskPositive.numpy(), pcd.point.maskNegative.numpy()), axis=1, dtype=np.float32)
        if self.normalize_colors:
            feats[:, :3] = feats[:, :3] / 255

        if self.verbose:
            print(f"Simulated click - {area.split('/')[-1]}/object {group}/point {points}")

        # Return the concatenated arrays
        return coords, feats, label

    def get_random_batch(self):
        # return random area/object every function call
        if not self.data:
            # Every point has been processed
            print("DataLoader: All points have been processed. Returning None.")
            return None
                
        # Select random area, object and point
        random_area = random.choice(list(self.data.keys()))
        random_object = random.randint(0, len(self.data[random_area])-1)

        random_points = self.data[random_area][random_object].pop(0)

        # Remove already simulated point from data
        if not self.data[random_area][random_object]:
            del self.data[random_area][random_object]
            if not self.data[random_area]:
                del self.data[random_area]
        else:                
            random.shuffle(self.data[random_area])
            
        return self.process_click(random_points, random_area)
    
    def get_batch_with_clicks(self, n_of_clicks):
        # return same area/object every function call
        if self.selected_object == None:
            self.next_random_batch()
        if self.selected_object == None:
            # Every point has been processed
            print("DataLoader: All points have been processed. Returning None.")
            return None
        
        if n_of_clicks > self.n_of_clicks:
            n_of_clicks = self.n_of_clicks
            
        points = [point for points in self.selected_object[:n_of_clicks] for point in points]
        return self.process_click(points, self.selected_area)
    
    def next_random_batch(self):
        if not self.data:
            self.selected_object = None
            return
        
        # Select random area, object and point
        random_area = random.choice(list(self.data.keys()))
        random_object = random.randint(0, len(self.data[random_area])-1)

        self.selected_object = self.data[random_area][random_object]
        self.selected_area = random_area

        del self.data[random_area][random_object]
        if not self.data[random_area]:
            del self.data[random_area]
        else:                
            random.shuffle(self.data[random_area])

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

