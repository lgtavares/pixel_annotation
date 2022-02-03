from PyQt5 import QtCore, QtGui, QtWidgets
from utils import newAction, addActions, struct, ToolBar

class WindowMenu(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, toolbar)
        return toolbar


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        
        # Widgets

        # Graphic view area
        scrollArea       = QtWidgets.QScrollArea()
        self.graphicview = QtWidgets.QGraphicsView()
        scrollArea.setWidget(self.graphicview)
        scrollArea.setWidgetResizable(True)       

        # Putting widgets together 
        self.setCentralWidget(scrollArea)

        # Actions
        # action = partial(create_action, self)
        # Menus
        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles = QtWidgets.QMenu('Open &Recent'),
            labelList   = QtWidgets.QMenu())
        addActions(self.menus.file, (None,))
        addActions(self.menus.help, (None,))
        addActions(self.menus.view, (None,))

        # Toolbar
        self.tools = self.toolbar('Tools')

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Silhouette Annotation tool"))