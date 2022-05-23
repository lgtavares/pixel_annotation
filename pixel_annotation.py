""" Silhouette annotation tool.

This code implements a pixel-level annotation tool for VDAO-2 database using QT5.

To do:

"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
import pickle
import imutils
import lightgbm as lgb 

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
from settings import Settings

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
        self.frame_settings = Settings()
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
        self.fps_vdao2    = [5, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 5]
        self.offset       = 0
        self.homography   = False

        self.anchor_frame = 0
        self.consistency  = False
        self.transform    = None
        self.anchor_silhouette = None
        self.anchor_target     = None

        self.contour_dc = []
        self.contour_fg = []

        # Status
        self.status_annotating = False
        self.status_comp_inclination = False
        self.status_ann_reference = False
        self.status_propagating = False
        self.status_consistency = False


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

        directory    = QFileDialog.getExistingDirectory(self, '%s - Choose database folder' % __appname__, self.datapath)
        self.dirname = os.path.abspath(directory)

        # Create video object
        self.videopair = VideoPair(self.dirname)
        self.align     = self.videopair.align
        self.video     = self.videopair.video

        self.fps       = self.fps_vdao2[self.video-1]
        self.status.loadedFiles = True

        # Temporary
        self.num_frames = self.videopair.num_frames
        self.frame_settings = Settings(self.num_frames,self.align)
        self.annotate_file  = os.path.join(self.dirname,'tar_vid{0:02d}.ann'.format(self.video))
        # Load angle correction
        self.calculate_angles()

        # Updating
        self.update(static=True)



    def update(self, static=False):
        """Load graphic view and set widgets"""

        # Enable/disable widgets
        self.enableWidgets()


        if self.status.loadedFiles:
            
            ref_frame, tar_frame, net_frame = self.videopair.get_frame(self.frame, offset=self.offset)
            self.setImages(ref_frame, tar_frame, net_frame, static_change=static)
            self.annotate()
            self._print_text(self.frame)


    def setImages(self, image_ref, image_tar, net_frames, static_change=False):

        # Checking if the frame is already annotated
        if self.frame_settings.get_value('annotated') and static_change==False:

            if self.consistency and self.consistency_checkbox.isChecked():
                self.contour_fg = self.frame_settings.get_anchor_value('contour_fg')
                self.contour_dc = self.frame_settings.get_anchor_value('contour_dc')
                transformation  = self.frame_settings.get_value('transform')

                detection   = np.zeros_like(net_frames)
                cv2.fillPoly(detection, pts=self.contour_dc, color=127)
                cv2.fillPoly(detection, pts=self.contour_fg, color=255) 
                warped_det = cv2.warpPerspective(detection, transformation, (detection.shape[1], detection.shape[0]))
                foreground = 255*(warped_det > 200).astype('uint8')
                dontcare   = 255*(warped_det > 50).astype('uint8')
                cnt_fg, cnt_dc = self.get_contours(foreground, dontcare)
                self.contour_fg = cnt_fg
                self.contour_dc = cnt_dc
            else:
                self.contour_fg = self.frame_settings.get_value('contour_fg')
                self.contour_dc = self.frame_settings.get_value('contour_dc')

        elif self.frame_settings.get_value('annotated') and static_change==True:

                self.contour_fg = self.frame_settings.get_anchor_value('contour_fg')
                self.contour_dc = self.frame_settings.get_anchor_value('contour_dc')

                detection   = np.zeros_like(net_frames)
                cv2.fillPoly(detection, pts=self.contour_dc, color=127)
                cv2.fillPoly(detection, pts=self.contour_fg, color=255)    
                foreground = 255*(detection > 200).astype('uint8')
                dontcare   = 255*(detection > 50).astype('uint8')
                cnt_fg, cnt_dc = self.get_contours(foreground, dontcare)
                self.contour_fg = cnt_fg
                self.contour_dc = cnt_dc                         
        else:

            # Post-processing
            detection, cnt_fg, cnt_dc = self.apply_morphology(net_frames)
            self.contour_fg = cnt_fg
            self.contour_dc = cnt_dc


        #  Angle compensation
        if self.angle_checkbox.isChecked():
            image_ref = imutils.rotate_bound(image_ref, self.ref_theta_x[self.frame] )
            image_tar = imutils.rotate_bound(image_tar, self.tar_theta_x[self.frame] )

        if self.mark_checkbox.isChecked():
            img_show_ref = image_tar
            img_show_tar = self.draw_contour(image_ref,self.contour_fg, self.contour_dc)
        else:
            img_show_ref = image_ref
            img_show_tar = self.draw_contour(image_tar,self.contour_fg, self.contour_dc)


        # Printing images on the screen
        if self.frame >=0 and self.frame<self.num_frames:
            self.scene_ref.clear()
            self.scene_tar.clear()
        
            imgmap_ref  = QtGui.QPixmap(to_pixmap(img_show_ref))
            pixItem_ref = QtWidgets.QGraphicsPixmapItem(imgmap_ref) 
            self.scene_ref.addItem(pixItem_ref)     
            self.scene_ref.setSceneRect(0, 0, imgmap_ref.width(), imgmap_ref.height())

            imgmap_tar  = QtGui.QPixmap(to_pixmap(img_show_tar))
            pixItem_tar = QtWidgets.QGraphicsPixmapItem(imgmap_tar) 
            self.scene_tar.addItem(pixItem_tar)     
            self.scene_tar.setSceneRect(0, 0, imgmap_tar.width(), imgmap_tar.height())

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
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.frame)
            self.frame_slider.blockSignals(False)
            self.pushbutton_next.setEnabled(True)
            self.pushbutton_prev.setEnabled(True)
            self.frame_label.setText('{0:>5d} / {1}'.format(self.frame+1, self.num_frames))
        
            if self.tcf_net_radio.isChecked() or self.rf_net_radio.isChecked():
                self.fold_combobox.setEnabled(True)
            else:
                self.fold_combobox.setEnabled(False)

            self.ann_text.setEnabled(True)

            ref_frame = self.frame_settings.get_value('reference_frame')

            # if self.frame < 10:
            #     self.ref_offset_sbox.setMinimum(-ref_frame)
            # if self.frame > self.num_frames - 11:
            #     self.ref_offset_sbox.setMaximum(self.num_frames - ref_frame -1)

        
    def setFrame(self):
        """Load frame"""

        # Setting status
        self.status_propagating = False # When I move to the next frame, I want to turn on the propagation
        self.status_annotating  = False 

        frm = self.frame_slider.value()
        if frm < self.num_frames and frm >=0:
            self.change_frame(frm)  

    def setFold(self):
        """Load frame"""
        self.videopair.fold = self.fold_combobox.currentIndex()
        self.videopair.set_mode(self.videopair.mode)
        self.update(static=True)    

    def nextFrame(self):
        """Load next frame"""

        # Setting status
        self.status_propagating = True # When I move to the next frame, I want to turn on the propagation
        self.status_annotating  = True 

        if self.frame < self.num_frames:
            self.change_frame(self.frame+1)


    def prevFrame(self):
        """Load previous frame"""

        # Setting status
        self.status_propagating = False # When I move to the next frame, I want to turn on the propagation
        self.status_annotating  = False # I want to just read 

        if self.frame >0:
            self.change_frame(self.frame-1)

    
    def change_frame(self, new_frame):

        # Checking if the new frame is the subsequent frame
        subseq = self.frame == new_frame-1

        # Correcting status
        if not self.activate_checkbox.isChecked():
            self.status_annotating  = False
            self.status_consistency = False
            self.status_propagating = False

        # annotating the current frame before changing 
        if self.status_annotating:
            self.frame_settings.set_value('annotated', True)

        # Setting frame label text
        self.frame_label.setText('{0:>5d} / {1}'.format(new_frame+1, self.num_frames))

        # Setting frame slider
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.frame)
        self.frame_slider.blockSignals(False)

        # Setting frame variable
        self.frame = new_frame
        self.frame_settings.set_frame(new_frame)

        # If the new frame is annotated
        if self.frame_settings.get_value('annotated'):
            
            # Reload frame parameters
            self.reload_variables()

        else:

            # if the frame is the subsequent frame, propagate the settings
            if subseq:
                self.frame_settings.propagate_previous()
            
        self.update(static=False)

            
    def reload_variables(self):

        frame = self.frame_settings.get_row(self.frame)

        self.mask_method  = frame['mask'].values[0]
        self.classifier   = frame['algorithm'].values[0]
        self.threshold    = frame['threshold'].values[0]
        self.opening      = frame['opening'].values[0]
        self.closing      = frame['closing'].values[0]
        self.erosion      = frame['erosion'].values[0]
        self.anchor_frame = frame['anchor'].values[0]
        self.consistency  = frame['consistency'].values[0]
        self.transform    = frame['transform'].values[0]
        self.contour_dc   = frame['contour_dc'].values[0]
        self.contour_fg   = frame['contour_fg'].values[0]
        self.fold         = frame['fold'].values[0]
        self.K            = frame['K'].values[0]
        self.offset       = frame['offset'].values[0]
        self.homography   = frame['homography'].values[0]

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

        self.ref_offset_sbox.blockSignals(True)
        self.ref_offset_sbox.setValue(self.offset)
        self.ref_offset_sbox.blockSignals(False)

        self.fold_combobox.blockSignals(True)
        self.fold_combobox.setCurrentIndex(self.fold)
        self.fold_combobox.blockSignals(False)

        self.homography_checkbox.blockSignals(True)
        self.homography_checkbox.setChecked(self.homography)
        self.homography_checkbox.blockSignals(False)

        self.k_sbox.blockSignals(True)
        self.k_sbox.setValue(self.K)
        self.k_sbox.blockSignals(False)

        self.consistency_checkbox.blockSignals(True)
        self.consistency_checkbox.setChecked(self.consistency)
        self.consistency_checkbox.click()
        self.consistency_checkbox.blockSignals(False)

        # if self.mask_method == 'ADMULT':
        #     self.admult_mask_radio.setChecked(True)
        #     # self.admult_mask_radio.click()
        # elif self.mask_method == 'ADMULT-20':
        #     self.admult20_mask_radio.setChecked(True)
        #     # self.admult20_mask_radio.click()
        # elif self.mask_method == 'ADMULT-40':
        #     self.admult40_mask_radio.setChecked(True)
        #     # self.admult40_mask_radio.click()
        # else:
        #     self.none_mask_radio.setChecked(True)
            # self.none_mask_radio.click()

        if self.classifier == 'TCF-LMO':
            self.tcf_net_radio.setChecked(True)
            # self.tcf_net_radio.click()
        elif self.classifier == 'Resnet+RF':
            self.rf_net_radio.setChecked(True)
            # self.rf_net_radio.click()
        elif self.classifier == 'Resnet+Dissim':
            self.diss_radio.setChecked(True)
            # self.diss_radio.click()
        elif self.classifier == 'K-means':
            self.km_net_radio.setChecked(True)
            # self.km_net_radio.click()
        else:
            self.none_net_radio.setChecked(False)
            # self.none_net_radio.click()

        # Correcting status
        self.status_annotating = False
         
    def resizeEvent(self, event):
        if self.status.loadedFiles:
            self.update(static=True)
        super(MainWindow, self).resizeEvent(event)
    

    def change_net(self):

        # Changing status
        self.status_annotating = True
        self.status_consistency = False
        self.status_propagating = False
        
        radioButton = self.sender()
        if radioButton.isChecked():
            self.videopair.set_mode(radioButton.mode)
            self.k_sbox.setEnabled(self.videopair.mode == 'K-means')
            self.classifier = self.videopair.mode
            self.update(static=True)


    def change_mask(self):

        # Changing status
        #self.status_annotating = True
        #self.status_consistency = False
        #self.status_propagating = False

        radioButton = self.sender()
        if radioButton.isChecked():
            self.videopair.mask = radioButton.mask
            self.mask_method = self.videopair.mask
            self.update(static=True)

    def apply_morphology(self, img):

        if self.consistency_checkbox.isChecked():
            
            # Get anchor contours
            anchor_fg, anchor_dc = self.frame_settings.get_anchor_value('contour_fg'), self.frame_settings.get_anchor_value('contour_dc')

            detection   = np.zeros_like(img)
            cv2.fillPoly(detection, pts=anchor_dc, color=127)
            cv2.fillPoly(detection, pts=anchor_fg, color=255) 

            # warp anchor detection
            warped_detection = self.warpFrame(detection)[:,:,0]

            # extracting contours
            foreground = cv2.cvtColor(255*(warped_detection > 200).astype('uint8'), cv2.COLOR_GRAY2RGB) 
            dontcare   = cv2.cvtColor(255*(warped_detection > 50).astype('uint8'), cv2.COLOR_GRAY2RGB) 

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

        if self.angle_checkbox.isChecked():
            if self.mark_checkbox.isChecked(): 
                foreground = imutils.rotate_bound(foreground, self.ref_theta_x[self.align[self.frame]] )
                dontcare   = imutils.rotate_bound(dontcare,   self.ref_theta_x[self.align[self.frame]] )
            else:
                foreground = imutils.rotate_bound(foreground, self.tar_theta_x[self.frame] )
                dontcare   = imutils.rotate_bound(dontcare,   self.tar_theta_x[self.frame] )

        foreground = cv2.cvtColor(foreground, cv2.COLOR_RGB2GRAY)
        dontcare   = cv2.cvtColor(dontcare, cv2.COLOR_RGB2GRAY)
        contours_fg, _ = cv2.findContours(foreground.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_dc, _ = cv2.findContours(dontcare.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if self.angle_checkbox.isChecked():
            foreground = imutils.rotate_bound(foreground, self.tar_theta_x[self.frame] )
            dontcare   = imutils.rotate_bound(dontcare, self.tar_theta_x[self.frame] )

        return list(contours_fg), list(contours_dc)
    
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
    
        

    def annotate(self):
        
        if self.status_annotating:
            if len(self.contour_dc)+len(self.contour_fg)>0  :
                self.frame_settings.set_value('has_object', True)
            else:
                self.frame_settings.set_value('has_object', False)
            self.frame_settings.set_value('mask', self.videopair.mask)
            self.frame_settings.set_value('algorithm', self.videopair.mode)
            self.frame_settings.set_value('fold', self.fold)
            self.frame_settings.set_value('K', self.K)
            self.frame_settings.set_value('opening', self.opening)
            self.frame_settings.set_value('closing', self.closing)
            self.frame_settings.set_value('erosion', self.erosion)
            self.frame_settings.set_value('threshold', self.threshold)
            self.frame_settings.set_value('anchor', self.anchor_frame)
            self.frame_settings.set_value('consistency', self.consistency)
            self.frame_settings.set_value('transform', self.transform)
            self.frame_settings.set_value('contour_fg', self.contour_fg.copy())
            self.frame_settings.set_value('contour_dc', self.contour_dc.copy())
            self.frame_settings.set_value('offset', self.offset)
            self.frame_settings.set_value('homography', self.homography)


    def load_settings(self):

        self.frame_settings.load(self.annotate_file)

        if os.path.exists(self.annotate_file):
            self.ann_text.insertPlainText('Settings loaded !!\n')
            self.frame = self.frame_settings.frame4
            self.frame_slider.setValue(self.frame)


    def save_settings(self):
        self.frame_settings.save(self.annotate_file)

    def set_morphology(self):

        # Setting status
        self.status_annotating = True
        self.status_consistency = False
        self.status_propagating = False

        self.threshold = self.thresh_slider.value()
        self.closing   = self.close_sbox.value()
        self.opening   = self.open_sbox.value()
        self.erosion   = self.erode_sbox.value()
        self.K         = self.k_sbox.value()
        self.offset    = self.ref_offset_sbox.value()
        self.fold      = self.fold_combobox.currentIndex()
        self.thresh_label.setText(str(self.threshold))
        
        self.homography = self.homography_checkbox.isChecked()
        self.videopair.homography =self.homography
        #self.consistency_checkbox.setChecked(False)
        self.update(static=True)
    
    def _print_text(self, frame):

        self.ann_text.clear()
        if frame < 2:
            [self._print_line(k) for k in range(self.frame+3)]
        elif frame > self.num_frames-3:
            [self._print_line(k) for k in range(self.frame-2, self.num_frames)]
        else:
            [self._print_line(k) for k in range(self.frame-2, self.frame+3)]

        self.ann_text.insertPlainText("\n")
        self.ann_text.insertPlainText("Status:\n")
        self.ann_text.insertPlainText("Annotating: {0} - ".format(self.status_annotating))
        self.ann_text.insertPlainText("Consistency: {0} - ".format(self.status_consistency))
        self.ann_text.insertPlainText("Propagating: {0}".format(self.status_propagating))


    def _print_line(self, f):

        print_str = self.frame_settings.get_row_str(f)
        self.ann_text.insertPlainText(print_str)


    def activate(self):


        if self.activate_checkbox.isChecked():

            # Status
            self.status_annotating  = True
            self.status_propagating = True
        
            self.class_groupbox.setEnabled(True)
            self.post_groupbox.setEnabled(True)
            self.vis_groupbox.setEnabled(True)
            self.thresh_slider.setEnabled(True)
            self.open_sbox.setEnabled(True)
            self.close_sbox.setEnabled(True)
            self.erode_sbox.setEnabled(True)
            self.contour_checkbox.setEnabled(True)
            self.fill_checkbox.setEnabled(True)
            self.mark_checkbox.setEnabled(True)
            self.angle_checkbox.setEnabled(True)
            self.ref_offset_sbox.setEnabled(True)
            self.homography_checkbox.setEnabled(True)

        else:
            # Status
            self.status_annotating  = False
            self.status_propagating = False

            self.class_groupbox.setEnabled(False)
            self.post_groupbox.setEnabled(False)
            self.thresh_slider.setEnabled(False)
            self.open_sbox.setEnabled(False)
            self.close_sbox.setEnabled(False)
            self.erode_sbox.setEnabled(False)
            self.contour_checkbox.setEnabled(False)
            self.fill_checkbox.setEnabled(False)
            self.mark_checkbox.setEnabled(False)
            self.angle_checkbox.setEnabled(False)
            self.ref_offset_sbox.setEnabled(False)
            self.homography_checkbox.setEnabled(False)

        self.update(static=True)

    def activate_toggle(self):
        # Setting status
        self.status_annotating = False
        self.status_consistency = False
        self.status_propagating = False

        self.activate_checkbox.setChecked(not self.activate_checkbox.isChecked())
        
    def activate_consistency(self):

        # Setting status
        self.status_annotating = True
        self.status_consistency = True
        self.status_propagating = True # ?

        self.anchor_frame = self.frame

        if self.consistency_checkbox.isChecked():
            self.consistency = True
            self.frame_settings.set_value('anchor', self.anchor_frame)
        else:
            self.consistency = False
            self.frame_settings.set_value('anchor', self.frame)

        self.frame_settings.set_value('consistency', self.consistency)
        self.update(static=True)

    def warpFrame(self, detection):

        if self.anchor_frame != self.frame:

            # Initiate SIFT detector
            sift = cv2.SIFT_create()

            # Get frames
            _, current_frm, _  = self.videopair.get_frame(self.frame)
            _, anchor_frm, _   = self.videopair.get_frame(self.anchor_frame)

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
            self.frame_settings.set_value('transform', self.transform)

            return warped_det
        else:
            return detection

    def set_rect(self):

        # Setting status
        self.status_annotating = True
        self.status_consistency = False
        self.status_propagating = False

        if -1 in self.graphicview_tar.box_rect.getRect():
            bboxes = []
        else:
            bboxes = self.frame_settings.get_value('bbox').copy()
            box_rect = self.graphicview_tar.box_rect
            box = self.graphicview_tar.mapToScene(box_rect).boundingRect()
            bboxes.append([int(i) for i in box.getRect()])
        self.videopair.rect = bboxes
        self.frame_settings.set_value('bbox', bboxes)
        self.update(static=True)
            
    def calculate_angles(self):

        target_df       =  pd.read_csv(self.videopair.target_sei, skiprows=16, sep=r"\s+")
        target_df       = target_df.loc[target_df.index == 'SEI:']
        target_df.index = range(target_df.shape[0])

        reference_df       =  pd.read_csv(self.videopair.reference_sei, skiprows=16, sep=r"\s+")
        reference_df       = reference_df.loc[reference_df.index == 'SEI:']
        reference_df.index = range(reference_df.shape[0])

        # Getting omega columns
        tar_omegax = target_df['OMEGA-X']
        tar_omegaz = target_df['OMEGA-Z']
        ref_omegax = reference_df['OMEGA-X']
        ref_omegaz = reference_df['OMEGA-Z']

        tar_bias_wx = np.median(tar_omegax)
        tar_bias_wz = np.median(tar_omegaz)
        ref_bias_wx = np.median(ref_omegax)
        ref_bias_wz = np.median(ref_omegaz)

        self.tar_theta_x = (np.cumsum(tar_omegax-tar_bias_wx)/self.fps)*180/np.pi
        self.tar_theta_z = (np.cumsum(tar_omegaz-tar_bias_wz)/self.fps)*180/np.pi
        self.ref_theta_x = (np.cumsum(ref_omegax-ref_bias_wx)/self.fps)*180/np.pi
        self.ref_theta_z = (np.cumsum(ref_omegaz-ref_bias_wz)/self.fps)*180/np.pi


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
