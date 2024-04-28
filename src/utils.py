import os
import time

import open3d as o3d


def ensure_folder_exists(folder_path):
    if not os.path.exists(os.path.dirname(folder_path)):
        os.makedirs(folder_path)


def timeit(start_time) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))


def remove_file_suffix(file_path):
    return os.path.splitext(file_path)[0]


def save_point_cloud_views(point_cloud, iou, i, path):
    save_point_cloud_views_with_window(point_cloud, os.path.join(path, f'point_cloud_{i}_iou_{iou:.0f}'))


def save_point_cloud_views_with_window(point_cloud, file_path):
    ensure_folder_exists(file_path)
    remove_file_suffix(file_path)
    print(f'Saving file to {file_path}')

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    vis.get_render_option().point_size = 3.0
    vis.add_geometry(point_cloud)
    vis.capture_screen_image(f"{file_path}_view_0.png", do_render=True)
    vis.destroy_window()
