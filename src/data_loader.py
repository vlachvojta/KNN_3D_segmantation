import os
import open3d as o3d
import pickle
import random
import time
from itertools import groupby

random.seed(time.time())


class DataLoader:
    def __init__(self, data_path, points_per_object=5, force=False):
        assert os.path.exists(data_path), "Data path does not exist. Choose a valid path to a dataset."

        self.cache_path = os.path.join(data_path, "dataloader_cache")

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
        batch = []
        for _ in range(batch_size):
            # Select random area, object and point
            random_area = random.choice(list(self.data.keys()))
            random_object = random.randint(0, len(self.data[random_area])-1)
            random_point = random.randint(0, len(self.data[random_area][random_object]) - 1)

            print(f"Simulated click - {random_area}/object {random_object}/point {random_point}")

            # Load pointcloud and simulate positive click in maskPositive
            pcd = o3d.t.io.read_point_cloud(random_area)
            pcd.point.maskPositive[self.data[random_area]
                                   [random_object][random_point]] = 1

            # Create a mask with the same group as the clicked point
            group = pcd.point.group[self.data[random_area]
                                    [random_object][random_point]].numpy()[0]
            label = (pcd.point.group.numpy() == group)
            label = o3d.core.Tensor(label, o3d.core.uint8, o3d.core.Device("CPU:0"))
            del pcd.point.group

            # Add tuple of pointcloud and label to batch
            batch.append((pcd, label))

            # Remove already simulated point from data
            del self.data[random_area][random_object][random_point]
            if not self.data[random_area][random_object]:
                del self.data[random_area][random_object]
                if not self.data[random_area]:
                    del self.data[random_area]
                    if not self.data:
                        # Every point has been processed
                        return []

        return batch
