import os
import time

import open3d as o3d
import numpy as np
import platform
from PIL import Image
import torch


def ensure_folder_exists(folder_path):
    if not os.path.exists(os.path.dirname(folder_path)):
        os.makedirs(folder_path, exist_ok=True)


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

    # Headless rendering is supported on linux only
    if platform.system() == 'Linux':
        width = 640
        height = 480
        
        render =  o3d.visualization.rendering.OffscreenRenderer(width, height)
        render.scene.add_geometry("pcd", point_cloud, o3d.visualization.rendering.MaterialRecord())
        
        # Get center and corner points of bounding box
        center = point_cloud.get_center()
        bb = render.scene.bounding_box
        bb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bb)
        points = np.asarray(bb.get_box_points())

        # Calculate center of each face
        faces = [(points[4] + points[5] + points[2] + points[7]) / 4,
                 (points[5] + points[3] + points[0] + points[2]) / 4,
                 (points[3] + points[6] + points[0] + points[1]) / 4,
                 (points[6] + points[4] + points[1] + points[7]) / 4,
                 (points[3] + points[4] + points[5] + points[6]) / 4,
                 (points[0] + points[1] + points[2] + points[7]) / 4]
        
    
        # Create image containing every view side by side
        output_img = Image.new('RGB', ((width*len(faces) + 2*(len(faces)-1)), (height*2)+2))
        
        # View from each face to center
        for i in range(len(faces)):
            vector = (faces[i] - center) 
            render.setup_camera(90, center, faces[i] + vector, [0, 0, 1])
            img = render.render_to_image()

            pil_img = Image.fromarray(np.array(img).astype('uint8'), 'RGB')
            output_img.paste(pil_img, (i*width + 2*i, 0))
            
        # View from center to each face
        for i in range(len(faces)):
            render.setup_camera(100, faces[i], center, [0, 0, 1])
            img = render.render_to_image()

            pil_img = Image.fromarray(np.array(img).astype('uint8'), 'RGB')
            output_img.paste(pil_img, (i*width + 2*i, height+2))

        output_img.save(f"{file_path}.png")
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
        vis.get_render_option().point_size = 3.0
        vis.add_geometry(point_cloud)
        vis.capture_screen_image(f"{file_path}_view_0.png", do_render=True)
        vis.destroy_window()

def save_tensor_to_txt(tensor, filename):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    np.savetxt(filename, tensor)
