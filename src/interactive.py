import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import argparse
import numpy as np
import os


class Interactive:
    def __init__(self, path):
        self.path = path
        self.initGUI()
        self.initApp()

    def initGUI(self):
        self.GUI_App = gui.Application.instance
        self.GUI_App.initialize()

        self.GUI_Window = gui.Application.instance.create_window(
            "Interactive segmentation", 1024, 768)
        self.GUI_Scene = gui.SceneWidget()
        self.GUI_Scene.set_on_mouse(self.onClick)
        self.GUI_Scene.scene = rendering.Open3DScene(self.GUI_Window.renderer)

        self.mainMenu = gui.Vert(0, gui.Margins(10, 10, 10, 10))

        self.file = gui.TextEdit()

        # For testing purposes, delete later
        self.file.text_value = os.path.join(
            self.path, "Area_1_conferenceRoom_1.pcd")

        fileButton = gui.Button("...")
        fileButton.horizontal_padding_em = 0.5
        fileButton.vertical_padding_em = 0
        fileButton.set_on_clicked(self.fileOpen)
        fileLayout = gui.Horiz()
        fileLayout.add_child(gui.Label("File"))
        fileLayout.add_child(self.file)
        fileLayout.add_fixed(5)
        fileLayout.add_child(fileButton)

        self.areaPositive = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.areaPositive.double_value = 0.05
        gridP = gui.VGrid(2, 5)
        gridP.add_child(gui.Label("Click area (positive) \t"))
        gridP.add_child(self.areaPositive)

        self.areaNegative = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.areaNegative.double_value = 0.05
        gridN = gui.VGrid(2, 5)
        gridN.add_child(gui.Label("Click area (negative)\t"))
        gridN.add_child(self.areaNegative)

        run = gui.Button("Run model")
        run.set_on_clicked(self.runModel)

        reload = gui.Button("Reload")
        reload.set_on_clicked(self.loadFile)

        self.mainMenu.add_child(fileLayout)
        self.mainMenu.add_fixed(5)
        self.mainMenu.add_child(gridP)
        self.mainMenu.add_fixed(2)
        self.mainMenu.add_child(gridN)
        self.mainMenu.add_fixed(5)
        self.mainMenu.add_child(run)
        self.mainMenu.add_fixed(5)
        self.mainMenu.add_child(reload)

        self.GUI_Window.set_on_layout(self._on_layout)
        self.GUI_Window.add_child(self.GUI_Scene)
        self.GUI_Window.add_child(self.mainMenu)

    def run(self):
        gui.Application.instance.run()

    def _on_layout(self, layout_context: gui.LayoutContext):
        self.GUI_Scene.frame = self.GUI_Window.content_rect
        width = 300
        height = min(
            self.GUI_Window.content_rect.height,
            self.mainMenu.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height
        )
        self.mainMenu.frame = gui.Rect(self.GUI_Window.content_rect.get_right(
        ) - width, self.GUI_Window.content_rect.y, width, height)

    def initApp(self):
        self.loadFile()

    def fileOpen(self):
        fileDialog = gui.FileDialog(
            gui.FileDialog.OPEN, "Select file", self.GUI_Window.theme)
        fileDialog.set_path(self.path)
        fileDialog.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        fileDialog.set_on_cancel(self.fileCancel)
        fileDialog.set_on_done(self.fileDone)
        self.GUI_Window.show_dialog(fileDialog)

    def fileCancel(self):
        self.GUI_Window.close_dialog()

    def fileDone(self, path):
        self.file.text_value = path
        self.GUI_Window.close_dialog()

        self.loadFile()

    def loadFile(self):
        if not os.path.exists(self.file.text_value):
            self.GUI_Window.show_message_box("Error", "File doesn't exist")
            return

        self.pcd = o3d.t.io.read_point_cloud(self.file.text_value)
        size = len(self.pcd.point.positions)
        self.pcd.point.maskPositive = o3d.core.Tensor(
            np.zeros((size, 1)), o3d.core.uint8, o3d.core.Device("CPU:0")).reshape((size, 1))
        self.pcd.point.maskNegative = o3d.core.Tensor(
            np.zeros((size, 1)), o3d.core.uint8, o3d.core.Device("CPU:0")).reshape((size, 1))

        self.render()

        bounds = self.GUI_Scene.scene.bounding_box
        self.GUI_Scene.setup_camera(60, bounds, bounds.get_center())

    def onClick(self, event):
        def click(depth_image, positive):
            x = event.x - self.GUI_Scene.frame.x
            y = event.y - self.GUI_Scene.frame.y

            depth = np.asarray(depth_image)[y, x]

            if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                return
            else:
                coords = self.GUI_Scene.scene.camera.unproject(
                    x, y, depth, self.GUI_Scene.frame.width,
                    self.GUI_Scene.frame.height)

            pcd_tree = o3d.geometry.PointCloud()
            pcd_tree.points = o3d.utility.Vector3dVector(
                self.pcd.point.positions.numpy())
            tree = o3d.geometry.KDTreeFlann(pcd_tree)

            [_, idx, _] = tree.search_radius_vector_3d(coords, self.areaPositive.double_value if positive
                                                       else self.areaNegative.double_value)
            for i in idx:

                self.pcd.point.colors[i] = [
                    0, 255, 0] if positive else [255, 0, 0]

                if positive:
                    self.pcd.point.maskPositive[i] = 1
                else:
                    self.pcd.point.maskNegative[i] = 1

            self.render()

        # CTRL + Click = positive click
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):
            self.GUI_Scene.scene.scene.render_to_depth_image(
                lambda depth_image: click(depth_image, True))
            return gui.Widget.EventCallbackResult.HANDLED

        # SHIFT + Click = negative click
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.SHIFT):
            self.GUI_Scene.scene.scene.render_to_depth_image(
                lambda depth_image: click(depth_image, False))
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def render(self):
        self.GUI_Scene.scene.clear_geometry()
        self.GUI_Scene.scene.add_geometry(
            "pcd", self.pcd, o3d.visualization.rendering.MaterialRecord())

    def runModel(self):
        # TODO
        pass

    def reload(self):
        self.loadFile()
        self.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", default="../dataset/S3DIS_converted",
                        help="Data path (default: ../dataset/S3DIS_converted")

    args = parser.parse_args()

    interactive = Interactive(args.src_path)
    interactive.run()
