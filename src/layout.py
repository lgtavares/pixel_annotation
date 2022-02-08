from PyQt5 import QtCore, QtGui, QtWidgets
from utils import newAction, addActions, struct, ToolBar, to_pixmap
from graphicsview import GraphicsView
from PyQt5 import QtCore, QtGui, QtWidgets
from functools import partial


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        
        # Main Window properties

        # Fonts

        # Icons

        # Central Widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # Layouts
        self.horizontalLayout   = QtWidgets.QHBoxLayout(self.centralwidget)
        self.img_verticalLayout = QtWidgets.QVBoxLayout()
        self.slide_horizontalLayout = QtWidgets.QHBoxLayout() 
        self.opt_verticalLayout     = QtWidgets.QVBoxLayout()

        # Widgets

        #### Graphic view area
        self.graphicview_ref = GraphicsView(self.centralwidget)
        self.graphicview_tar = GraphicsView(self.centralwidget)
        self.graphicview_ref.setMinimumSize(QtCore.QSize(400, 225))
        self.graphicview_tar.setMinimumSize(QtCore.QSize(400, 225))
        self.img_verticalLayout.addWidget(self.graphicview_ref)
        self.img_verticalLayout.addWidget(self.graphicview_tar)

        #### Frame slide area
        self.frame_slider = QtWidgets.QSlider(self.centralwidget)
        self.frame_slider.setOrientation(QtCore.Qt.Horizontal)   
        self.frame_slider.setEnabled(False)     
        self.img_verticalLayout.addWidget(self.frame_slider)

        #### Next/Previous buttons
        self.pushbutton_prev = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_next = QtWidgets.QPushButton(self.centralwidget)
        self.pushbutton_prev.setEnabled(False)     
        self.pushbutton_next.setEnabled(False)    
        self.pushbutton_next.setText('\u25b6')
        self.pushbutton_prev.setText('\u25c0')

        self.frame_edit  = QtWidgets.QTextEdit(self.centralwidget)
        self.frame_label = QtWidgets.QLabel(self.centralwidget)
        self.frame_edit.setText('-')
        self.frame_label.setText('/ -')
        self.frame_edit.setMaximumSize(QtCore.QSize(75, 40))
    
        self.slide_horizontalLayout.addWidget(self.pushbutton_prev)
        self.slide_horizontalLayout.addWidget(self.frame_slider)
        self.slide_horizontalLayout.addWidget(self.pushbutton_next)
        self.slide_horizontalLayout.addWidget(self.frame_edit)
        self.slide_horizontalLayout.addWidget(self.frame_label)
        self.img_verticalLayout.addLayout(self.slide_horizontalLayout)

        #GroupBoxes
        self.class_groupbox = QtWidgets.QGroupBox('Classification',self.centralwidget)
        self.post_groupbox  = QtWidgets.QGroupBox('Post-processing',self.centralwidget)
        self.vis_groupbox   = QtWidgets.QGroupBox('Miscellaneous',self.centralwidget)
        self.opt_verticalLayout.addWidget(self.class_groupbox)
        self.opt_verticalLayout.addWidget(self.post_groupbox)
        self.opt_verticalLayout.addWidget(self.vis_groupbox)
        self.class_groupbox.setEnabled(False)
        self.post_groupbox.setEnabled(False)
        self.vis_groupbox.setEnabled(False)


        # Classifier groupbox
        class_layout  = QtWidgets.QHBoxLayout()
        mask_groupbox = QtWidgets.QGroupBox('Mask',self.centralwidget)
        net_groupbox  = QtWidgets.QGroupBox('Classifier',self.centralwidget)
        opt_groupbox  = QtWidgets.QGroupBox('Options',self.centralwidget)
        self.daomc_mask_radio  = QtWidgets.QRadioButton('DAOMC')
        self.tcf_mask_radio    = QtWidgets.QRadioButton('TCF-LMO')
        self.rf_mask_radio     = QtWidgets.QRadioButton('Resnet+RF')
        self.km_mask_radio     = QtWidgets.QRadioButton('K-means')
        self.none_mask_radio   = QtWidgets.QRadioButton('None')
        self.none_mask_radio.setChecked(True)
        maskbox = QtWidgets.QVBoxLayout()
        maskbox.addWidget(self.daomc_mask_radio)
        maskbox.addWidget(self.tcf_mask_radio)
        maskbox.addWidget(self.rf_mask_radio)
        maskbox.addWidget(self.km_mask_radio)
        maskbox.addWidget(self.none_mask_radio)
        mask_groupbox.setLayout(maskbox)
        classbox = QtWidgets.QVBoxLayout()
        self.daomc_net_radio  = QtWidgets.QRadioButton('DAOMC')
        self.tcf_net_radio    = QtWidgets.QRadioButton('TCF-LMO')
        self.rf_net_radio     = QtWidgets.QRadioButton('Resnet+RF')
        self.km_net_radio     = QtWidgets.QRadioButton('K-means')
        self.daomc_net_radio.mode  = 'DAOMC'
        self.tcf_net_radio.mode    = 'TCF-LMO'
        self.rf_net_radio.mode     = 'Resnet+RF'
        self.km_net_radio.mode     = 'K-means'
        classbox.addWidget(self.daomc_net_radio)
        classbox.addWidget(self.tcf_net_radio)
        classbox.addWidget(self.rf_net_radio)
        classbox.addWidget(self.km_net_radio)
        net_groupbox.setLayout(classbox)



        class_layout.addWidget(mask_groupbox)
        class_layout.addWidget(net_groupbox)
        class_layout.addWidget(opt_groupbox)
        self.class_groupbox.setLayout(class_layout)

        # Post-processing groupbox
        post_layout   = QtWidgets.QVBoxLayout()
        thresh_layout = QtWidgets.QHBoxLayout()
        morphlbl_layout  = QtWidgets.QHBoxLayout()
        morph_layout     = QtWidgets.QHBoxLayout()
        self.thresh_slider = QtWidgets.QSlider(self.centralwidget)
        self.thresh_slider.setEnabled(False)
        self.thresh_slider.setMaximum(100)
        self.thresh_slider.setProperty("value", 50)
        self.thresh_slider.setOrientation(QtCore.Qt.Horizontal)
        self.thresh_label =  QtWidgets.QLabel(self.centralwidget)
        self.thresh_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.thresh_label.setText('50')
        thresh_layout.addWidget(self.thresh_slider)
        thresh_layout.addWidget(self.thresh_label)
        self.open_sbox = QtWidgets.QSpinBox(self.centralwidget)
        self.open_sbox.setEnabled(False)
        self.open_sbox.setMinimum(1)
        self.close_sbox = QtWidgets.QSpinBox(self.centralwidget)
        self.close_sbox.setEnabled(False)
        self.close_sbox.setMinimum(1)
        self.erode_sbox = QtWidgets.QSpinBox(self.centralwidget)
        self.erode_sbox.setEnabled(False)
        self.erode_sbox.setMinimum(1)
        morph_layout.addWidget(self.open_sbox)
        morph_layout.addWidget(self.close_sbox)
        morph_layout.addWidget(self.erode_sbox)
        morphlbl_layout.addWidget(QtWidgets.QLabel('Opening'))
        morphlbl_layout.addWidget(QtWidgets.QLabel('Closing'))
        morphlbl_layout.addWidget(QtWidgets.QLabel('Eroding'))
        post_layout.addWidget(QtWidgets.QLabel('Threshold'))
        post_layout.addLayout(thresh_layout)
        post_layout.addLayout(morphlbl_layout)
        post_layout.addLayout(morph_layout)
        self.post_groupbox.setLayout(post_layout)

        # Visualization groupbox




        # Combining layouts
        self.horizontalLayout.addLayout(self.img_verticalLayout)
        self.horizontalLayout.addLayout(self.opt_verticalLayout)

        # Connections
        self.frame_slider.valueChanged.connect(self.setFrame)
        self.pushbutton_next.clicked.connect(self.nextFrame)
        self.pushbutton_prev.clicked.connect(self.prevFrame)
        self.daomc_net_radio.clicked.connect(self.change_net)
        self.tcf_net_radio.clicked.connect(self.change_net)
        self.rf_net_radio.clicked.connect(self.change_net)
        self.km_net_radio.clicked.connect(self.change_net)
        MainWindow.setCentralWidget(self.centralwidget)

        # Actions
        action = partial(newAction, self)
        quit   = action('&Quit', self.close,'Ctrl+Q', 'quit',u'Quit application')
        open   = action('&Open', self.openFile,'Ctrl+O', 'open', u'Open image or label file')
        nextFrame = action('&Next Frame', self.nextFrame,'d', 'next', u'Next Frame')
        prevFrame = action('&Prev Frame', self.prevFrame,'a', 'prev', u'Previous Frame')
        self.actions = struct(quit=quit,open=open, nextFrame=nextFrame, prevFrame=prevFrame)


        # Menus
        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles = QtWidgets.QMenu('Open &Recent'),
            labelList   = QtWidgets.QMenu())
        addActions(self.menus.file, (open, None, quit))
        addActions(self.menus.edit, (None,))
        addActions(self.menus.help, (None,))
        addActions(self.menus.view, (prevFrame, nextFrame, None))

        addActions(self.pushbutton_next, (nextFrame,))
        addActions(self.pushbutton_prev, (prevFrame,))


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