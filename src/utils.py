import os

import open3d as o3d


def ensure_folder_exists(folder_path):
    if not os.path.exists(os.path.dirname(folder_path)):
        os.makedirs(folder_path)


def remove_file_suffix(file_path):
    return os.path.splitext(file_path)[0]


def visualize_prediction_result(coords, feats, labels, pred, iou, results, i):
    # create point cloud from coords and pred
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(coords.numpy())

    colors = feats.numpy()[:, :3] / 255
    colors[labels.cpu().numpy().reshape(-1) == 1] = [0, 1, 0] # Label
    colors[pred.cpu().numpy()[:, 0] == 1] = [1, 0, 0] # maskPositive output
    colors[feats.cpu().numpy()[:, 3] == 1] = [0, 0, 1] # maskNegative input
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # visualize point cloud
    # o3d.visualization.draw_geometries([point_cloud])
    save_point_cloud_views_with_window(point_cloud, f'../results/test_output_{i}_iou_{iou:.0f}')


def save_point_cloud_views(point_cloud, file_path):
    ensure_folder_exists(file_path)
    remove_file_suffix(file_path)
    print(f'Saving file to {file_path}')

    # Define the four camera viewpoints
    # Each viewpoint is a 4x4 matrix represented as a flat list of 16 elements
    # viewpoints = [
    #     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], # North-West-Up
    #     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1], # South-East-Up
    #     [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], # South-West-Down
    #     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1] # North-East-Down
    # ]

    # # Loop through each viewpoint
    # for i, viewpoint in enumerate(viewpoints):
    #     # Create a camera object with default intrinsic parameters
    #     camera = o3d.camera.PinholeCameraIntrinsic(
    #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    #     # Create a PinholeCameraParameters object and set the extrinsic parameters
    #     camera_params = o3d.camera.PinholeCameraParameters()
    #     camera_params.intrinsic = camera
    #     camera_params.extrinsic = np.array(viewpoint).reshape(4, 4)

    #     # Render the point cloud from the current viewpoint
    #     rendered_image = point_cloud.render_point_cloud_to_image(camera_params)

    #     # Save the rendered image
    #     image_file_path = f"{file_path}_view_{i+1}.png"
    #     o3d.io.write_image(image_file_path, rendered_image)

    # pcd = o3d.io.read_point_cloud("file.pcd", format="pcd")

    # Create a renderer
    renderer = o3d.visualization.rendering.Renderer()

    # Create a scene
    scene = o3d.visualization.rendering.Open3DScene(renderer)

    # Add the point cloud to the scene
    scene.add_geometry("point_cloud", point_cloud)

    # Set up the camera
    camera = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    scene.camera = camera

    # Render the scene to an image
    image = scene.render_to_image()

    # Save the image
    o3d.io.write_image("point_cloud_image.png", image)


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
