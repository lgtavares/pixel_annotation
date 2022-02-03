""" Silhouette annotation tool.

This code implements a pixel-level annotation tool for VDAO-2 database using QT5.

To do:

"""
import os
import sys
import cv2
import numpy as np
import pandas as pd


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
from layout import WindowMenu
from utils import newAction, addActions, struct, ToolBar


__version__ = '0.1'
__appname__ = 'VDAO2 silhouette annotator'


class MainWindow(QMainWindow, MainWindow, WindowMenu):

    def __init__(self, parent=None, name=None):

        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        self.setup()

    def setup(self):

        self.setupUi(self)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
