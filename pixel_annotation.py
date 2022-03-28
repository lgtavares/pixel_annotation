""" Silhouette annotation tool.

This code implements a pixel-level annotation tool for VDAO-2 database using QT5.

To do:

"""
import os
import sys
import cv2
from cv2 import COLOR_RGB2GRAY
import numpy as np
import pandas as pd
import pickle

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
        self.consistency  = False
        self.video        = 0
        self.status       = struct(loadedFiles=False,)
        self.frame        = 0
        self.num_frames   = 0


        self.mask_method = None
        self.classifier  = None
        self.threshold   = 127
        self.opening     = 1
        self.closing     = 1
        self.erosion     = 1
        self.K            = 2
        self.fold         = 0

        self.anchor_frame = 0
        self.consistency  = False
        self.transform    = None
        self.anchor_silhouette = None
        self.anchor_target     = None

        self.bboxes = []
        self.contour_dc = []
        self.contour_fg = []


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
        self.admult_filename = os.path.join(dirname, 'admult')
        self.admult_filename = os.path.join(self.admult_filename, 
                                           'dissimilarity_video{0:02d}.avi'.format(self.video))
        admult_dilate = os.path.join(dirname, 'admult_dilate')
        self.admult20_filename = os.path.join(admult_dilate, 
                                              'vid{0:02d}_dilate20.avi'.format(self.video)) 
        self.admult40_filename = os.path.join(admult_dilate, 
                                              'vid{0:02d}_dilate40.avi'.format(self.video))           
        self.diss_filename = os.path.join(dirname, 'dissimilarity')
        self.diss_filename = os.path.join(self.diss_filename, 
                                         'dissimilarity_video{0:02d}.avi'.format(self.video))

        self.error(os.path.exists(self.ref_filename), 
                    'Error opening file {}.'.format(self.ref_filename)+
                    'Make sure it is a valid file.')
        self.error(os.path.exists(self.tar_filename), 
                    'Error opening file\n {}.'.format(self.tar_filename)+
                    '\nMake sure it is a valid file.')
        self.error(os.path.exists(self.admult_filename), 
                    'Error opening file\n {}.'.format(self.admult_filename)+
                    '\nMake sure it is a valid file.')
        self.error(os.path.exists(self.admult20_filename), 
                    'Error opening file\n {}.'.format(self.admult_filename)+
                    '\nMake sure it is a valid file.')
        self.error(os.path.exists(self.admult40_filename), 
                    'Error opening file\n {}.'.format(self.admult_filename)+
                    '\nMake sure it is a valid file.')
        self.error(os.path.exists(self.tcf_filename), 
                    'Error opening folder\n {}.'.format(self.tcf_filename)+
                    '\nMake sure it is a valid folder.')
        self.error(os.path.exists(self.rf_filename), 
                    'Error opening folder\n {}.'.format(self.rf_filename)+
                    '\nMake sure it is a valid folder.')
        self.error(os.path.exists(self.diss_filename), 
                    'Error opening folder\n {}.'.format(self.diss_filename)+
                    '\nMake sure it is a valid folder.')

        self.videopair = VideoPair(self.ref_filename, self.tar_filename,
                                   self.rf_filename, self.tcf_filename,
                                   self.admult_filename, self.diss_filename)
        self.status.loadedFiles = True

        # Temporary
        self.num_frames = self.videopair.tar_video.num_frames

        # Create settings
        self.start_settings()

        # Updating
        self.update()



    def update(self):
        """Load graphic view and set widgets"""

        # Enable/disable widgets
        self.enableWidgets()


        if self.status.loadedFiles:

            # setting bounding box
            self.bboxes = self.settings[self.frame]['bbox']
            self.videopair.rect = self.bboxes

            ref_frame, tar_frame, net_frame = self.videopair.get_frame(self.frame)
            self.setImages(ref_frame, tar_frame, net_frame)
            self.annotate()
            self._print_text(self.frame)


    def setImages(self, image_ref, image_tar, net_frames):

        # Checking if the frame is already annotated
        if self.settings[self.frame]['annotated'] :

            self.contour_fg = self.settings[self.frame]['contour_fg']
            self.contour_dc = self.settings[self.frame]['contour_dc']
            img_show_ref =  image_ref
            img_show_tar =  self.draw_contour(image_tar, 
                              self.contour_fg, 
                              self.contour_dc)
        else:
            # Post-processing
            detection, cnt_fg, cnt_dc = self.apply_morphology(net_frames)
            self.contour_fg = cnt_fg
            self.contour_dc = cnt_dc

            img_show_ref = image_ref
            img_show_tar = self.draw_contour(image_tar,cnt_fg, cnt_dc)

            # Saving anchor frame
            if self.frame == self.anchor_frame:
                self.anchor_sil    = detection
                self.anchor_target = image_tar

        self.frame_edit.setText('{}'.format(self.frame+1))
        if self.frame >=0 and self.frame<self.num_frames:
            self.scene_ref.clear()
            self.scene_tar.clear()

            imgmap_ref  = QtGui.QPixmap(to_pixmap(img_show_ref))
            pixItem_ref = QtWidgets.QGraphicsPixmapItem(imgmap_ref) 
            self.scene_ref.addItem(pixItem_ref)     

            imgmap_tar  = QtGui.QPixmap(to_pixmap(img_show_tar))
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
            self.frame_slider.setValue(self.frame)
            self.pushbutton_next.setEnabled(True)
            self.pushbutton_prev.setEnabled(True)
            self.frame_label.setText('/ {}'.format(self.num_frames))
            self.noobject_pushbutton.setEnabled(True)
        
            if self.tcf_net_radio.isChecked() or self.rf_net_radio.isChecked():
                self.fold_combobox.setEnabled(True)
            else:
                self.fold_combobox.setEnabled(False)

            self.ann_text.setEnabled(True)

        
    def setFrame(self):
        """Load frame"""
        frm = self.frame_slider.value()
        if frm < self.num_frames and frm >=0:
           self.frame = frm
        self.change_frame()    

    def setFold(self):
        """Load frame"""
        self.videopair.fold = self.fold_combobox.currentIndex()
        self.update()    

    def nextFrame(self):
        """Load next frame"""

        if self.frame < self.num_frames:
            self.write_setting('annotated', True)
            self.frame += 1
            #if self.settings[self.frame+1]['annotated']:
            #    self.reload_variables()

            self.change_frame()

    def prevFrame(self):
        """Load previous frame"""
        if self.frame >0:
            self.write_setting('annotated', True)
            self.frame -= 1
            self.change_frame()

            self.frame_slider.setValue(self.frame)

    
    def change_frame(self):

        # If frame is annotated, the annotations is turned off and the current contours are loaded
        # as well as the parameters
        if self.settings[self.frame]['annotated']:
            
            # Turning annotation off
            # self.activate_checkbox.setChecked(False)

            # Re-set parameters
            self.reload_variables()

            # Load contours
        else:
            self.propag_settings()
            self.write_setting('annotated', False)
            
        self.update()

            
    def reload_variables(self):

        frame = self.settings[self.frame].copy()
        self.mask_method = frame['mask']
        self.classifier  = frame['algorithm']
        self.threshold   = frame['threshold']
        self.opening     = frame['opening']
        self.closing     = frame['closing']
        self.erosion     = frame['erosion']

        self.anchor_frame = frame['anchor']
        self.consistency  = frame['consistency']
        self.transform    = frame['transform']

        self.bboxes = frame['bbox']
        self.contour_dc = frame['contour_dc']
        self.contour_fg = frame['contour_fg']

        self.fold = frame['fold']
        self.K    = frame['K']


        # reseting widgets
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.frame)
        self.frame_slider.blockSignals(False)

        self.thresh_slider.blockSignals(True)
        self.thresh_slider.setValue(self.threshold)
        self.thresh_label.setText(str(self.threshold))
        self.thresh_slider.blockSignals(False)

        self.open_sbox.blockSignals(True)
        self.open_sbox.setValue(self.opening)
        self.open_sbox.blockSignals(False)

        self.close_sbox.blockSignals(True)
        self.close_sbox.setValue(self.closing)
        self.close_sbox.blockSignals(False)

        self.erode_sbox.blockSignals(True)
        self.erode_sbox.setValue(self.erosion)
        self.erode_sbox.blockSignals(False)

        self.fold_combobox.blockSignals(True)
        self.fold_combobox.setCurrentIndex(self.fold)
        self.fold_combobox.blockSignals(False)

        self.k_sbox.blockSignals(True)
        self.k_sbox.setValue(self.K)
        self.k_sbox.blockSignals(False)

        # self.admult_mask_radio.blockSignals(True)
        # self.admult20_mask_radio.blockSignals(True)
        # self.admult40_mask_radio.blockSignals(True)
        # self.none_mask_radio.blockSignals(True)
        if self.mask_method == 'ADMULT':
            self.admult_mask_radio.setChecked(True)
            self.admult_mask_radio.click()
        elif self.mask_method == 'ADMULT-20':
            self.admult20_mask_radio.setChecked(True)
            self.admult20_mask_radio.click()
        elif self.mask_method == 'ADMULT-40':
            self.admult40_mask_radio.setChecked(True)
            self.admult40_mask_radio.click()
        else:
            self.none_mask_radio.setChecked(True)
            self.none_mask_radio.click()

        # self.admult_mask_radio.blockSignals(False)
        # self.admult20_mask_radio.blockSignals(False)
        # self.admult40_mask_radio.blockSignals(False)  
        # self.none_mask_radio.blockSignals(False)

        #self.tcf_net_radio.blockSignals(True)
        #self.rf_net_radio.blockSignals(True)
        #self.diss_radio.blockSignals(True)
        #self.km_net_radio.blockSignals(True)
        #self.none_net_radio.blockSignals(True)
        if self.classifier == 'TCF-LMO':
            self.tcf_net_radio.setChecked(True)
            self.tcf_net_radio.click()
        elif self.classifier == 'Resnet+RF':
            self.rf_net_radio.setChecked(True)
            self.rf_net_radio.click()
        elif self.classifier == 'Resnet+Dissim':
            self.diss_radio.setChecked(True)
            self.diss_radio.click()
        elif self.classifier == 'K-means':
            self.km_net_radio.setChecked(True)
            self.km_net_radio.click()
        else:
            self.none_net_radio.setChecked(False)
            self.none_net_radio.click()
        #self.tcf_net_radio.blockSignals(False)
        #self.rf_net_radio.blockSignals(False)
        #self.diss_radio.blockSignals(False)
        #self.km_net_radio.blockSignals(False)
        #self.none_net_radio.blockSignals(False)
     
        #self.pushbutton_next.setEnabled(True)
        #self.pushbutton_prev.setEnabled(True)
        #self.frame_label.setText('/ {}'.format(self.num_frames))
        #self.noobject_pushbutton.setEnabled(True)
    
        #if self.tcf_net_radio.isChecked() or self.rf_net_radio.isChecked():
        #    self.fold_combobox.setEnabled(True)
        #else:
        #    self.fold_combobox.setEnabled(False)

    def propag_settings(self):
        if self.frame >0:
            if self.settings[self.frame]['annotated'] == False:
                previous_frame = self.settings[self.frame-1].copy()
                self.settings[self.frame]['mask']      =  previous_frame['mask']
                self.settings[self.frame]['algorithm'] = previous_frame['algorithm']
                self.settings[self.frame]['opening'] = previous_frame['opening']
                self.settings[self.frame]['closing'] = previous_frame['closing']
                self.settings[self.frame]['erosion'] = previous_frame['erosion']
                self.settings[self.frame]['threshold'] = previous_frame['threshold']
                self.settings[self.frame]['anchor']   = previous_frame['anchor']
                self.settings[self.frame]['fold']   = previous_frame['fold']
                self.settings[self.frame]['K']   = previous_frame['K']
         
    def resizeEvent(self, event):
        if self.status.loadedFiles:
            self.update()
        super(MainWindow, self).resizeEvent(event)
    

    def change_net(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.videopair.mode = radioButton.mode
            self.consistency_checkbox.setChecked(False)
            self.k_sbox.setEnabled(self.videopair.mode == 'K-means')
            self.classifier = self.videopair.mode
            self.update()


    def change_mask(self):
        radioButton = self.sender()
        if radioButton.isChecked():
            self.videopair.mask = radioButton.mask
            self.consistency_checkbox.setChecked(False)
            self.mask_method = self.videopair.mask
            self.update()

    def apply_morphology(self, img):

        if self.consistency_checkbox.isChecked():
            
            # set detection
            detection   = self.anchor_sil 

            # warp anchor detection
            warped_detection = self.warpFrame(detection)

            # extracting contours
            gray_det   = cv2.cvtColor(warped_detection, cv2.COLOR_RGB2GRAY) 
            foreground = cv2.cvtColor(255*(gray_det > 200).astype('uint8'), cv2.COLOR_GRAY2RGB) 
            dontcare   = cv2.cvtColor(255*(gray_det > 50).astype('uint8'), cv2.COLOR_GRAY2RGB) 

            cnt_fg, cnt_dc = self.get_contours(foreground, dontcare)

        else:
            

            # structures
            struct_opening  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.opening, self.opening))
            struct_closing  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.closing, self.closing))
            struct_erosion  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.erosion, self.erosion))
            
            # threshold
            _, img = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)

            # opening
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, struct_opening)

            # closing
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, struct_closing)

            # erosion
            foreground = cv2.morphologyEx(img, cv2.MORPH_ERODE,  struct_erosion, borderType = cv2.BORDER_CONSTANT, borderValue = 0) 
            dontcare   = cv2.morphologyEx(img, cv2.MORPH_DILATE, struct_erosion, borderType = cv2.BORDER_CONSTANT, borderValue = 0)         

            cnt_fg, cnt_dc = self.get_contours(foreground, dontcare)

            detection = foreground//2 + dontcare//2
   
        return detection, cnt_fg, cnt_dc


    def get_contours(self, foreground, dontcare):

        foreground = cv2.cvtColor(foreground, cv2.COLOR_RGB2GRAY)
        dontcare   = cv2.cvtColor(dontcare, cv2.COLOR_RGB2GRAY)
        contours_fg, _ = cv2.findContours(foreground.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_dc, _ = cv2.findContours(dontcare.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours_fg, contours_dc
    
    def draw_contour(self, img, contours_fg, contours_dc):

        alpha = 0.5
        if self.contour_checkbox.isChecked():
            cv2.drawContours(img, contours_dc, -1, (0,102, 204, 0.5), 2) 
            cv2.drawContours(img, contours_fg, -1, (191, 64, 64, 0.5), 2) 

        result_img = img.copy()

        if self.fill_checkbox.isChecked():
            cv2.fillPoly(img, pts=contours_dc, color=(0, 102, 204, 0.5))
            cv2.fillPoly(img, pts=contours_fg, color=(191, 64, 64, 0.5))
        
        result_img = cv2.addWeighted(img, alpha, result_img, 1-alpha, 0, result_img)
        return result_img
    
    def noobject(self):
        # mark the current frame as not having an object
        self.write_setting('has_object', False)
        # Go to the next frame
        self.nextFrame()

    def start_settings(self):
        frame_settings = {}
        frame_settings['annotated']  = False
        frame_settings['has_object'] = None
        frame_settings['mask']       = None
        frame_settings['algorithm']  = None
        frame_settings['fold']       = 0
        frame_settings['K']          = 2
        frame_settings['opening']    = 0
        frame_settings['closing']    = 0
        frame_settings['erosion']    = 0
        frame_settings['threshold']  = 127
        frame_settings['contour_fg'] = []
        frame_settings['contour_dc'] = []
        frame_settings['anchor']      = 0
        frame_settings['consistency'] = None
        frame_settings['transform']   = None
        frame_settings['bbox']        = []

        self.settings = {i:frame_settings.copy() for i in range(self.num_frames)}
        self.settings['video'] = self.video

    def annotate(self):
        
        if len(self.contour_dc)+len(self.contour_fg)>0  :
            self.write_setting('has_object', True)
        else:
            self.write_setting('has_object', False)
        self.write_setting('mask', self.videopair.mask)
        self.write_setting('algorithm', self.videopair.mode)
        self.write_setting('fold', self.fold)
        self.write_setting('K', self.K)
        self.write_setting('opening', self.opening)
        self.write_setting('closing', self.closing)
        self.write_setting('erosion', self.erosion)
        self.write_setting('threshold', self.threshold)
        self.write_setting('anchor', self.anchor_frame)
        self.write_setting('consistency', self.consistency)
        self.write_setting('transform', self.transform)
        self.write_setting('contour_fg', self.contour_fg.copy())
        self.write_setting('contour_dc', self.contour_dc.copy())
        self.write_setting('bbox', self.bboxes.copy())


    def write_setting(self, setting, value):
        self.settings[self.frame][setting] = value

    def load_settings(self):

        if os.path.exists(self.tar_filename.replace('.avi','.ann')):
            with open(self.tar_filename.replace('.avi','.ann'), 'rb') as input_file:
                self.settings = pickle.load(input_file)
                last_frame = list(pd.DataFrame(self.settings).T.annotated).index(False)-1
                self.frame_slider.setValue(last_frame)
                self.ann_text.insertPlainText('Settings loaded !!\n')
        else:
            self.save_settings()

    def save_settings(self):
        with open(self.tar_filename.replace('.avi','.ann'), 'wb') as output_file:
            pickle.dump(self.settings, output_file)
            self.ann_text.insertPlainText('Settings saved !!\n')

    def set_morphology(self):
        self.threshold = self.thresh_slider.value()
        self.closing   = self.close_sbox.value()
        self.opening   = self.open_sbox.value()
        self.erosion   = self.erode_sbox.value()
        self.K         = self.k_sbox.value()
        self.fold      = self.fold_combobox.currentIndex()
        self.thresh_label.setText(str(self.threshold))

        self.consistency_checkbox.setChecked(False)
        self.update()
    
    def _print_text(self, frame):

        self.ann_text.clear()
        if frame < 2:
            [self._print_line(k) for k in range(self.frame+3)]
        elif frame > self.num_frames-2:
            [self._print_line(k) for k in range(self.frame-2, self.num_frames)]
        else:
            [self._print_line(k) for k in range(self.frame-2, self.frame+3)]

    def _print_line(self, f):

        frm_set = self.settings[f]
        print_str  = 'Frame {0:>4d}:'.format(f+1)
        print_str += '{0},'.format(frm_set['has_object'])
        print_str += '{0},'.format(frm_set['annotated'])
        print_str += '{0},'.format(frm_set['mask'])
        print_str += '{0},['.format(frm_set['algorithm'])
        print_str += '{0},'.format(frm_set['opening'])
        print_str += '{0},'.format(frm_set['closing'])
        print_str += '{0},'.format(frm_set['erosion'])
        print_str += '{0}],['.format(frm_set['threshold'])
        print_str += '{0},'.format(frm_set['anchor'])
        print_str += '{0}],'.format(frm_set['consistency'])
        print_str += '[{0},'.format(len(frm_set['contour_fg']))
        print_str += '{0}]'.format(len(frm_set['contour_dc']))
        print_str += '[{0}]'.format(len(frm_set['bbox']))
        print_str += '\n'

        self.ann_text.insertPlainText(print_str)


    def activate(self):
        
        if self.activate_checkbox.isChecked():

            self.class_groupbox.setEnabled(True)
            self.post_groupbox.setEnabled(True)
            self.vis_groupbox.setEnabled(True)
            self.thresh_slider.setEnabled(True)
            self.open_sbox.setEnabled(True)
            self.close_sbox.setEnabled(True)
            self.erode_sbox.setEnabled(True)
            self.contour_checkbox.setEnabled(True)
            self.fill_checkbox.setEnabled(True)
            #self.write_setting('annotated', True)

        else:

            self.class_groupbox.setEnabled(False)
            self.post_groupbox.setEnabled(False)
            self.thresh_slider.setEnabled(False)
            self.open_sbox.setEnabled(False)
            self.close_sbox.setEnabled(False)
            self.erode_sbox.setEnabled(False)
            self.contour_checkbox.setEnabled(False)
            self.fill_checkbox.setEnabled(False)
            #self.write_setting('annotated', False)

        self.update()

    def activate_toggle(self):
        self.activate_checkbox.setChecked(not self.activate_checkbox.isChecked())
        
    def activate_consistency(self):

        self.anchor_frame = self.frame

        if self.activate_checkbox.isChecked():
            self.consistency = True
        else:
            self.consistency = False
        self.update()

    def warpFrame(self, detection):

        # Initiate SIFT detector
        sift = cv2.SIFT_create()


        # Get frames
        _, current_frm, _  = self.videopair.get_frame(self.frame)
        anchor_frm = self.anchor_target

        kp1, des1 = sift.detectAndCompute(anchor_frm,None)
        kp2, des2 = sift.detectAndCompute(current_frm,None)

        index_params = dict(algorithm = 1, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        # Select good matched keypoints
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        # Compute homography
        H, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        warped_det = cv2.warpPerspective(detection, H, (detection.shape[1], detection.shape[0]))

        # Save Homography
        self.transform = H
        self.write_setting('transform', self.transform)

        # apply threshold

        return warped_det

    def set_rect(self):
        if -1 in self.graphicview_tar.box_rect.getRect():
            self.bboxes = []
        else:
            box_rect = self.graphicview_tar.box_rect
            box = self.graphicview_tar.mapToScene(box_rect).boundingRect()
            self.bboxes.append([int(i) for i in box.getRect()])
        self.videopair.rect = self.bboxes
        self.update()
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
