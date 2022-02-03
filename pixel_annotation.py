""" Silhouette annotation tool.

This code implements a pixel-level annotation tool for VDAO-2 database using QT5.

To do:

"""
import os
import sys
import cv2
import numpy as np


# PyQt libraries
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui     import QPixmap, QImage
from PyQt5.QtCore    import Qt

# Add internal libs
dir_name = os.path.abspath(os.path.dirname(__file__))
libs_path = os.path.join(dir_name, 'src')
sys.path.insert(0, libs_path)

# Importing layout
from layout import Ui_MainWindow as MainWindow
from utils import newAction, addActions, struct, ToolBar, LabelFileError, to_pixmap
from utils import WindowMenu
from video import VideoPair

__version__ = '0.1'
__appname__ = 'VDAO2 silhouette annotator'


class MainWindow(QMainWindow, MainWindow, WindowMenu):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, parent=None, name=None):

        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        self.basepath = os.path.abspath(os.path.dirname(__file__))
        self.datapath = os.path.join(self.basepath, 'data')
        self.ref_filename = None
        self.tar_filename = None
        self.videopair    = None
        self.settings     = {}

        self.status       = struct(loadedFiles=False,)
        self.frame        = 0
        self.setupUi(self)
        self.setup()

    def setup(self):
        
        # Setting scenes
        self.scene_ref = QGraphicsScene()
        self.scene_tar = QGraphicsScene()
        self.graphicview_ref.setScene(self.scene_ref)
        self.graphicview_tar.setScene(self.scene_tar)

        # Scalers
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH:  self.scaleFitWidth,
            self.MANUAL_ZOOM: lambda: 1,
        }
    def openFile(self):

        filters    = "Video files (*.avi *.mp4);; All files (*)"
        filename   = QFileDialog.getOpenFileName(self, '%s - Choose Reference or target video' % __appname__, self.datapath, filters)
        input_file = filename[0]
        if 'ref' in input_file:
            self.ref_filename = input_file
            self.tar_filename = input_file.replace('ref', 'tar')
        if 'tar' in input_file:
            self.ref_filename = input_file.replace('tar', 'ref')
            self.tar_filename = input_file

        self.error(os.path.exists(self.ref_filename), 
                    'Error opening file {}.'.format(self.ref_filename)+
                    'Make sure it is a valid file.')
        self.error(os.path.exists(self.tar_filename), 
                    'Error opening file\n {}.'.format(self.tar_filename)+
                    '\nMake sure it is a valid file.')
        self.videopair = VideoPair(self.ref_filename, self.tar_filename)
        self.status.loadedFiles = True

        # Temporary
        self.num_frames = self.videopair.tar_video.num_frames
        self.enableWidgets()


    def setFrame(self, idx):
        """Load graphic view and set widgets"""

        if self.status.loadedFiles:

            ref_frame, tar_frame = self.videopair.get_frame(idx-1)
            self.setImages(ref_frame, tar_frame)

    def setImages(self, image_ref, image_tar):

        self.scene_ref.clear()
        self.scene_tar.clear()

        imgmap_ref  = QtGui.QPixmap(to_pixmap(image_ref))
        pixItem_ref = QtWidgets.QGraphicsPixmapItem(imgmap_ref) 
        self.scene_ref.addItem(pixItem_ref)     

        imgmap_tar  = QtGui.QPixmap(to_pixmap(image_tar))
        pixItem_tar = QtWidgets.QGraphicsPixmapItem(imgmap_tar) 
        self.scene_tar.addItem(pixItem_tar)     

        self.scene_ref.update()
        self.scene_tar.update()

    def error(self, test, message):
        if not test:
            return QtWidgets.QMessageBox.critical(self, 'Error', (message))

    def enableWidgets(self):

        if self.status.loadedFiles:
            self.frame_slider.setEnabled(True)
            self.frame_slider.setMinimum(1)
            self.frame_slider.setMaximum(self.num_frames)
            self.frame_slider.setValue(1)

    # def scaleFitWindow(self):

    #     """Figure out the size of the pixmap in order to fit the main widget."""
    #     e = 2.0  # So that no scrollbars are generated.
    #     w1 = self.centralWidget().width()  - e
    #     h1 = self.centralWidget().height() - e
    #     a1 = w1 / h1
    #     # Calculate a new scale value based on the pixmap's aspect ratio.
    #     w2 = self.graphicview_ref.width() - 0.0
    #     h2 = self.graphicview_ref.width() - 0.0
    #     a2 = w2 / h2
    #     return w1 / w2 if a2 >= a1 else h1 / h2

    # def scaleFitWidth(self):

    #     w = self.centralWidget().width() - 2.0
    #     return w / self.graphicview_ref.width()

            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
