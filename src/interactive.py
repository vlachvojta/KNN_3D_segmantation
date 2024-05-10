import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import argparse
import numpy as np
import os

import torch
from InterObject3D.interactive_adaptation import InteractiveSegmentationModel


class Interactive:
    def __init__(self, path, model_path_ours, model_path_io3d, fix = False):
        self.pcd_path = path
        self.model_path_ours = model_path_ours
        self.model_path_io3d = model_path_io3d
        self.fix = fix
        self.initGUI()
        self.initApp()

    def initGUI(self):
        self.GUI_App = gui.Application.instance
        self.GUI_App.initialize()

        self.GUI_Window = gui.Application.instance.create_window("Interactive segmentation", 1024, 768)
        self.GUI_Scene = gui.SceneWidget()
        self.GUI_Scene.set_on_mouse(self.onClick)
        self.GUI_Scene.set_on_key(self.onKey)
        self.GUI_Scene.scene = rendering.Open3DScene(self.GUI_Window.renderer)

        self.mainMenu = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        

        self.filePcd = gui.TextEdit()
        self.filePcd.enabled = False
        self.filePcd.text_value = self.pcd_path.split('/')[-1]
        fileButton = gui.Button("...")
        fileButton.horizontal_padding_em = 0.5
        fileButton.vertical_padding_em = 0
        fileButton.set_on_clicked(self.fileOpenPcd)
        fileLayout = gui.Horiz()
        fileLayout.add_child(gui.Label("Pcd"))
        fileLayout.add_child(self.filePcd)
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
        
        reload = gui.Button("Reload")
        reload.set_on_clicked(self.loadFile)

        modelOurs = gui.CollapsableVert("Our model", 0, gui.Margins(5, 0, 0, 0))
        self.fileOur = gui.TextEdit()
        self.fileOur.enabled = False
        fileButton = gui.Button("...")
        fileButton.horizontal_padding_em = 0.5
        fileButton.vertical_padding_em = 0
        fileButton.set_on_clicked(self.fileOpenModelOur)
        fileLayoutOur = gui.Horiz()
        fileLayoutOur.add_child(gui.Label("Model"))
        fileLayoutOur.add_child(self.fileOur)
        fileLayoutOur.add_fixed(5)
        fileLayoutOur.add_child(fileButton)
        self.runOur = gui.Button("Run model")
        self.runOur.set_on_clicked(self.runModelOur)
        
        if self.model_path_ours:
            self.fileOur.text_value = self.model_path_ours.split('/')[-1]
            self.modelOur_class, self.modelOur_global = self.loadModel(self.model_path_ours)
        else:
            self.runOur.enabled = False
            
        modelOurs.add_child(fileLayoutOur)
        modelOurs.add_fixed(5)
        modelOurs.add_child(self.runOur)
        
        modelIO3Ds = gui.CollapsableVert("InterObject3D model", 0, gui.Margins(5, 0, 0, 0))
        self.fileIO3D = gui.TextEdit()
        self.fileIO3D.enabled = False
        fileButton = gui.Button("...")
        fileButton.horizontal_padding_em = 0.5
        fileButton.vertical_padding_em = 0
        fileButton.set_on_clicked(self.fileOpenModelIO3D)
        fileLayoutIO3D = gui.Horiz()
        fileLayoutIO3D.add_child(gui.Label("Model"))
        fileLayoutIO3D.add_child(self.fileIO3D)
        fileLayoutIO3D.add_fixed(5)
        fileLayoutIO3D.add_child(fileButton)
        self.runIO3D = gui.Button("Run model")
        self.runIO3D.set_on_clicked(self.runModelIO3D)
        
        if self.model_path_io3d:
            self.fileIO3D.text_value = self.model_path_io3d.split('/')[-1]
            self.modelIO3D_class, self.modelIO3D_global = self.loadModel(self.model_path_io3d)
            
        else:
            self.runIO3D.enabled = False
        
        modelIO3Ds.add_child(fileLayoutIO3D)
        modelIO3Ds.add_fixed(5)
        modelIO3Ds.add_child(self.runIO3D)

        
        # SETTINGS
        settings = gui.CollapsableVert("Settings", 0, gui.Margins(5, 0, 0, 0))
        
        self.downsample = gui.NumberEdit(gui.NumberEdit.INT)
        self.downsample.int_value = 0
        self.downsample.set_on_value_changed(self.onDownsampleChange)
        grid = gui.VGrid(2, 5)
        grid.add_child(gui.Label("Downsample"))
        grid.add_child(self.downsample)
        settings.add_fixed(5)
        settings.add_child(grid)
        
        self.pointSize = gui.Slider(gui.Slider.INT)
        self.pointSize.set_limits(1, 20)
        self.pointSize.set_on_value_changed(self.onPointChange)
        self.pointSize.int_value = 4
        grid = gui.VGrid(2, 5)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self.pointSize)
        settings.add_fixed(5)
        settings.add_child(grid)
        
        self.positiveColor = gui.ColorEdit()
        self.positiveColor.set_on_value_changed(self.onColorChange)
        self.positiveColor.color_value = gui.Color(0, 1, 0)
        grid = gui.VGrid(2, 5)
        grid.add_child(gui.Label("Positive color"))
        grid.add_child(self.positiveColor)
        settings.add_fixed(5)
        settings.add_child(grid)
        
        self.negativeColor = gui.ColorEdit()
        self.negativeColor.set_on_value_changed(self.onColorChange)
        self.negativeColor.color_value = gui.Color(1, 0, 0)
        grid = gui.VGrid(2, 5)
        grid.add_child(gui.Label("Negative color"))
        grid.add_child(self.negativeColor)
        settings.add_fixed(5)
        settings.add_child(grid)
        
        self.predColor = gui.ColorEdit()
        self.predColor.set_on_value_changed(self.onColorChange)
        self.predColor.color_value = gui.Color(0, 0, 1)
        grid = gui.VGrid(2, 5)
        grid.add_child(gui.Label("Prediction Color"))
        grid.add_child(self.predColor)
        settings.add_fixed(5)
        settings.add_child(grid)

        self.mainMenu.add_child(fileLayout)
        self.mainMenu.add_fixed(5)
        self.mainMenu.add_child(gridP)
        self.mainMenu.add_fixed(2)
        self.mainMenu.add_child(gridN)
        self.mainMenu.add_fixed(10)
        self.mainMenu.add_child(modelOurs)
        self.mainMenu.add_fixed(20)
        self.mainMenu.add_child(modelIO3Ds)
        self.mainMenu.add_fixed(10)
        self.mainMenu.add_child(settings)
        self.mainMenu.add_fixed(10)
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
        self.mat = o3d.visualization.rendering.MaterialRecord()
        self.mat.point_size = 4
        
        self.loadFile()

    def fileOpenPcd(self):
        fileDialog = gui.FileDialog(
            gui.FileDialog.OPEN, "Select file", self.GUI_Window.theme)
        fileDialog.set_path(self.pcd_path)
        fileDialog.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        fileDialog.add_filter("", "All files")
        fileDialog.set_on_cancel(self.fileCancel)
        fileDialog.set_on_done(self.fileDonePcd)
        self.GUI_Window.show_dialog(fileDialog)

    def fileCancel(self):
        self.GUI_Window.close_dialog()

    def fileDonePcd(self, path):
        self.pcd_path = path
        self.filePcd.text_value = path.split('/')[-1]
        self.GUI_Window.close_dialog()
        self.loadFile()
        
    def fileOpenModelOur(self):
        fileDialog = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self.GUI_Window.theme)
        fileDialog.add_filter(".pth", "PyTorch model (.pth)")
        fileDialog.add_filter("", "All files")
        fileDialog.set_on_cancel(self.fileCancel)
        fileDialog.set_on_done(self.fileDoneModelOur)
        self.GUI_Window.show_dialog(fileDialog)

    def fileDoneModelOur(self, path):
        self.model_path_ours = path
        self.fileOur.text_value = path.split('/')[-1]
        self.runOur.enabled = True   
        
        self.modelOur_class, self.modelOur_global = self.loadModel(self.model_path_ours)
        
        self.GUI_Window.close_dialog()
        
    def fileOpenModelIO3D(self):
        fileDialog = gui.FileDialog(gui.FileDialog.OPEN, "Select file", self.GUI_Window.theme)
        fileDialog.add_filter(".pth", "PyTorch model (.pth)")
        fileDialog.add_filter("", "All files")
        fileDialog.set_on_cancel(self.fileCancel)
        fileDialog.set_on_done(self.fileDoneModelIO3D)
        self.GUI_Window.show_dialog(fileDialog)

    def fileDoneModelIO3D(self, path):
        self.model_path_io3d = path	
        self.fileIO3D.text_value = path.split('/')[-1]
        self.runIO3D.enabled = True   
        
        self.modelIO3D_class, self.modelIO3D_global = self.loadModel(self.model_path_io3d)
        
        self.GUI_Window.close_dialog()
        
    def loadModel(self, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inseg_model_class = InteractiveSegmentationModel(pretraining_weights=path)
        inseg_global_model = inseg_model_class.create_model(device, inseg_model_class.pretraining_weights_file)
        return inseg_model_class, inseg_global_model

    def loadFile(self):
        if not os.path.exists(self.pcd_path):
            self.GUI_Window.show_message_box("Error", "File doesn't exist")
            return

        self.pcd_original = o3d.t.io.read_point_cloud(self.pcd_path)
        self.maskPositive = []
        self.maskNegative = []
        self.pred = None
        
        size = len(self.pcd_original.point.positions)
        self.pcd_original.point.maskPositive = o3d.core.Tensor(
            np.zeros((size, 1)), o3d.core.uint8, o3d.core.Device("CPU:0")).reshape((size, 1))
        self.pcd_original.point.maskNegative = o3d.core.Tensor(
            np.zeros((size, 1)), o3d.core.uint8, o3d.core.Device("CPU:0")).reshape((size, 1))

        self.pcd_original.point.colors = o3d.core.Tensor(
            self.pcd_original.point.colors.numpy().astype(np.float32) / 255,
            o3d.core.float32,
            o3d.core.Device("CPU:0")
        )

        self.render()

        bounds = self.GUI_Scene.scene.bounding_box
        self.GUI_Scene.setup_camera(60, bounds, bounds.get_center())

    def reload(self):
        self.loadFile()
        self.render()

    def onClick(self, event):
        def click(depth_image):
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
                self.pcd_original.point.positions.numpy())
            tree = o3d.geometry.KDTreeFlann(pcd_tree)
            
            positive = event.is_modifier_down(gui.KeyModifier.CTRL)
            
            [_, idx, _] = tree.search_radius_vector_3d(coords, self.areaPositive.double_value if positive
                                                       else self.areaNegative.double_value)

            if positive:
                self.maskPositive.append(list(idx))
            else:
                self.maskNegative.append(list(idx))
            for i in idx:
                if positive:
                    self.pcd_original.point.maskPositive[i] = 1
                else:
                    self.pcd_original.point.maskNegative[i] = 1
                    
            # Fix for broken rendering (on some systems it doesn't render correctly)
            # When the fix is enabled, you need to re-render manually by presing space 
            if not self.fix:
                self.render()

        # CTRL + Click = positive click
        # SHIFT + Click = negative click
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and (event.is_modifier_down(gui.KeyModifier.CTRL) or event.is_modifier_down(gui.KeyModifier.SHIFT)):
            self.GUI_Scene.scene.scene.render_to_depth_image(click)
            self.render()
            return gui.Widget.EventCallbackResult.HANDLED
 
        return gui.Widget.EventCallbackResult.IGNORED

    def onKey(self, event):
        if event.key == gui.KeyName.SPACE:
            self.render()
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def render(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pcd_original.point.positions.numpy())
        pcd.colors = o3d.utility.Vector3dVector(self.pcd_original.point.colors.numpy())
        
        colors = np.asarray(pcd.colors)
        
        # Paint points
        if self.pred is not None:
            colors[self.pred == 1] = self.getColor(self.predColor)
            
        for i in range(len(self.maskPositive)):
            colors[self.maskPositive[i]] = self.getColor(self.positiveColor)
        for i in range(len(self.maskNegative)):
            colors[self.maskNegative[i]] = self.getColor(self.negativeColor)
        
        pcd.colors = o3d.utility.Vector3dVector(colors)

        self.GUI_Scene.scene.clear_geometry()
        self.GUI_Scene.scene.add_geometry("pcd", pcd, self.mat)

    def getColor(self, color):
        return [color.color_value.red, color.color_value.green, color.color_value.blue]

    def onColorChange(self, _):
        self.render()
    
    def onPointChange(self, value):
        self.mat.point_size = value
        self.render()
        
    def onDownsampleChange(self, value):
        self.reload()
        if value < 0:
            self.downsample.int_value = 0
        elif value > 0:
            self.pcd_original = self.pcd_original.uniform_down_sample(every_k_points=int(value))
        self.render()


    def runModelOur(self):
        self.model_path = self.model_path_ours
        self.runModel(self.modelOur_class, self.modelOur_global)
        
    def runModelIO3D(self):
        self.model_path = self.model_path_io3d
        self.runModel(self.modelIO3D_class, self.modelIO3D_global)

    def runModel(self, inseg_class, inseg_model):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        coords = self.pcd_original.point.positions.numpy()
        feats = np.concatenate((self.pcd_original.point.colors.numpy(), self.pcd_original.point.maskPositive.numpy(), self.pcd_original.point.maskNegative.numpy()), axis=1, dtype=np.float32)
        
        coords = torch.tensor(coords).float().to(device)
        feats = torch.tensor(feats).float().to(device)
        
        pred, _ = inseg_class.prediction(feats.float(), coords.cpu().numpy(), inseg_model, device)

        self.pred = pred.cpu().numpy()
        self.render()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--default_file", default="../dataset/S3DIS_converted_downsampled/test/Area_5_conferenceRoom_1.pcd",
                        help="Data path (default: ../dataset/S3DIS_converted_downsampled/test/Area_5_conferenceRoom_1.pcd")
    parser.add_argument("-m", "--model_path_ours", help="Model path - ours")
    parser.add_argument("-i", "--model_path_interobject3d", help="Model path - InterObject3D")
    parser.add_argument("-f", "--fix", help="Fix for broken rendering", action="store_true")
    args = parser.parse_args()

    interactive = Interactive(args.default_file, args.model_path_ours, args.model_path_interobject3d, args.fix)
    interactive.run()
