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

        self.video        = 0
        self.status       = struct(loadedFiles=False,)
        self.frame        = 0
        self.num_frames   = 0

        self.mask = None
        self.classifier = None
        
        self.setupUi(self)
        self.setup()

    def setup(self):
        
        # Setting scenes
        self.scene_ref = QGraphicsScene()
        self.scene_tar = QGraphicsScene()
        self.graphicview_ref.setScene(self.scene_ref)
        self.graphicview_tar.setScene(self.scene_tar)

        # Toolbar
        self.tools = self.toolbar('Tools')
        

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

        self.video = int(self.tar_filename.split('vid')[-1].split('.')[0])
        dirname = os.path.dirname(self.ref_filename)
        self.tcf_filename   = os.path.join(dirname, 'tcf_lmo')
        self.rf_filename    = os.path.join(dirname, 'rf')
        self.daomc_filename = os.path.join(dirname, 'daomc')
        self.daomc_filename = os.path.join(self.daomc_filename, 
                                           'dissimilarity_video{0:02d}.avi'.format(self.video))

        self.error(os.path.exists(self.ref_filename), 
                    'Error opening file {}.'.format(self.ref_filename)+
                    'Make sure it is a valid file.')
        self.error(os.path.exists(self.tar_filename), 
                    'Error opening file\n {}.'.format(self.tar_filename)+
                    '\nMake sure it is a valid file.')
        self.error(os.path.exists(self.daomc_filename), 
                    'Error opening file\n {}.'.format(self.daomc_filename)+
                    '\nMake sure it is a valid file.')
        self.error(os.path.exists(self.tcf_filename), 
                    'Error opening folder\n {}.'.format(self.tcf_filename)+
                    '\nMake sure it is a valid folder.')
        self.error(os.path.exists(self.rf_filename), 
                    'Error opening folder\n {}.'.format(self.rf_filename)+
                    '\nMake sure it is a valid folder.')

        self.videopair = VideoPair(self.ref_filename, self.tar_filename,
                                   self.rf_filename, self.tcf_filename,
                                   self.daomc_filename)
        self.status.loadedFiles = True

        # Temporary
        self.num_frames = self.videopair.tar_video.num_frames
        self.enableWidgets()


    def setFrame(self, idx):
        """Load graphic view and set widgets"""

        self.frame = idx

        ###### DEBUG ###########
        self.videopair.fold = 1
               
        if self.status.loadedFiles:

            ref_frame, tar_frame, net_frame = self.videopair.get_frame(self.frame)
            self.setImages(net_frame, tar_frame)

    # def setFrameLabel(self):

    #     frame = int(self.frame_edit.toPlainText())

    #     if isinstance(frame, int) :
    #         if frame < self.num_frames and frame>=0:
    #             self.frame = frame
    #             self.setFrame(self.frame)
    #         else:
    #             return QtWidgets.QMessageBox.critical(self, 'Error', ('Value must be integer between 1 and {}'.format(self.num_frames)))
    #     else:
    #         return QtWidgets.QMessageBox.critical(self, 'Error', ('Value must be integer'))

    def setImages(self, image_ref, image_tar):

        self.frame_edit.setText('{}'.format(self.frame+1))
        if self.frame >=0 and self.frame<self.num_frames:
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

            self.graphicview_ref.fitInView(self.scene_ref.sceneRect(),
                                        QtCore.Qt.KeepAspectRatio)
            self.graphicview_tar.fitInView(self.scene_tar.sceneRect(),
                                        QtCore.Qt.KeepAspectRatio)


    def error(self, test, message):
        if not test:
            return QtWidgets.QMessageBox.critical(self, 'Error', (message))

    def enableWidgets(self):

        if self.status.loadedFiles:
            self.frame_slider.setEnabled(True)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(self.num_frames-1)
            self.frame_slider.setValue(0)
            self.pushbutton_next.setEnabled(True)
            self.pushbutton_prev.setEnabled(True)
            self.frame_label.setText('/ {}'.format(self.num_frames))
            self.setFrame(0)
            self.class_groupbox.setEnabled(True)
            self.post_groupbox.setEnabled(True)
            self.vis_groupbox.setEnabled(True)

    def nextFrame(self):
        """Load next frame"""
        if self.frame < self.num_frames:
            self.frame += 1
            self.setFrame(self.frame)

    def prevFrame(self):
        """Load previous frame"""
        if self.frame >0:
            self.frame -= 1
            self.setFrame(self.frame)

    def resizeEvent(self, event):
        if self.status.loadedFiles:
            self.setFrame(self.frame)
        
        super(MainWindow, self).resizeEvent(event)
    

    def change_net(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.videopair.mode = radioButton.mode
            self.setFrame(self.frame)
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
