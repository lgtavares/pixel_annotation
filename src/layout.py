from functools import partial
from PyQt5 import QtCore, QtGui, QtWidgets
from utils import newAction, addActions, struct, ToolBar, to_pixmap
from graphicsview import GraphicsView
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        
        # Main Window properties

        # Fonts

        # Icons

        # Central Widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # Layouts
        self.horizontalLayout   = QtWidgets.QHBoxLayout(self.centralwidget)
        self.img_verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)

        # Widgets

        #### Graphic view area
        self.graphicview_ref = GraphicsView(self.centralwidget)
        self.graphicview_tar = GraphicsView(self.centralwidget)
        self.graphicview_ref.setMinimumSize(QtCore.QSize(800, 450))
        self.graphicview_tar.setMinimumSize(QtCore.QSize(800, 450))
        self.img_verticalLayout.addWidget(self.graphicview_ref)
        self.img_verticalLayout.addWidget(self.graphicview_tar)

        #### Frame slide area
        self.frame_slider = QtWidgets.QSlider(self.centralwidget)
        self.frame_slider.setOrientation(QtCore.Qt.Horizontal)   
        self.frame_slider.setEnabled(False)     
        self.img_verticalLayout.addWidget(self.frame_slider)

        # Combining layouts
        self.horizontalLayout.addLayout(self.img_verticalLayout)

        # Actions
        action = partial(newAction, self)
        quit = action('&Quit', self.close,
                      'Ctrl+Q', 'quit', 
                      u'Quit application')

        open = action('&Open', self.openFile,
                      'Ctrl+O', 'open',
                       u'Open image or label file')

        #openNextImg = action('&Next Image', self.openNextImg,
        #                     'd', 'next', u'Open Next')

        # Connections
        self.frame_slider.valueChanged.connect(self.setFrame)

        # Menus
        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles = QtWidgets.QMenu('Open &Recent'),
            labelList   = QtWidgets.QMenu())
        addActions(self.menus.file, (open, None, quit))
        addActions(self.menus.help, (None,))
        addActions(self.menus.view, (None,))

        # Toolbar
        self.tools = self.toolbar('Tools')
        MainWindow.setCentralWidget(self.centralwidget)


    # def resetState(self):

    #     """Reset the Ui state"""
    #     #self.itemsToShapes.clear()
    #     #self.shapesToItems.clear()
    #     #self.labelList.clear()
    #     self.filePath  = None
    #     self.imageData = None
    #     self.labelFile = None
    #     #self.canvas.resetState()


    #     #print(filepath)
    #     #if filePath is None:
    #     #    filePath = self.settings.get('filename')

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Silhouette Annotation tool"))