import cv2
import os
import sys
import numpy as np
from PIL import Image
import scipy.io
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Add internal libs
dir_name = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, dir_name)

# Importing layout
import torch
import torchvision.transforms as transforms
from resnet import Resnet50_Reduced, MEAN_IMAGENET, STD_IMAGENET


# Transforms
normalize_transform = transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
to_tensor_transform = transforms.ToTensor()
resnet_transform    = transforms.Compose([to_tensor_transform, normalize_transform])



class VideoPair:
    def __init__(self, dirname):

        # Video
        self.video            = int(dirname.split('data/')[1])

        # Folders
        self.reference_folder = os.path.join(dirname, 'reference')
        self.target_folder    = os.path.join(dirname, 'target')
        self.rf_folder        = os.path.join(os.path.dirname(dirname),'rf')

        # Files
        self.alignment        = os.path.join(dirname, 'alignment.mat')
        self.reference_sei    = os.path.join(dirname, 'reference_{0:02d}.sei'.format(self.video))
        self.target_sei       = os.path.join(dirname, 'target_{0:02d}.sei'.format(self.video))


        # self.tcf_filename   = os.path.join(dirname, 'tcf_lmo')
        # self.rf_filename    = os.path.join(dirname, 'rf')
        # self.admult_filename = os.path.join(dirname, 'admult')
        # self.admult_filename = os.path.join(self.admult_filename, 
        #                                    'dissimilarity_video{0:02d}.avi'.format(self.video))
        # admult_dilate = os.path.join(dirname, 'admult_dilate')
        # self.admult20_filename = os.path.join(admult_dilate, 
        #                                       'vid{0:02d}_dilate20.avi'.format(self.video)) 
        # self.admult40_filename = os.path.join(admult_dilate, 
        #                                       'vid{0:02d}_dilate40.avi'.format(self.video))           
        # self.diss_filename = os.path.join(dirname, 'dissimilarity')
        # self.diss_filename = os.path.join(self.diss_filename, 
        #                                  'dissimilarity_video{0:02d}.avi'.format(self.video))

        # self.ref_sei_filename   = os.path.join(dirname, 'reference_{0:02d}.sei'.format(self.video))
        # self.tar_sei_filename   = os.path.join(dirname, 'target_{0:02d}.sei'.format(self.video))

        self.error(os.path.exists(self.reference_folder),'Error opening file\n {}.'.format(self.reference_folder)+'\nMake sure it is a valid folder.')
        self.error(os.path.exists(self.target_folder),'Error opening file\n {}.'.format(self.target_folder)+'\nMake sure it is a valid folder.')
        self.error(os.path.exists(self.reference_sei),'Error opening file\n {}.'.format(self.reference_sei)+'\nMake sure it is a valid file.')
        self.error(os.path.exists(self.target_sei),'Error opening file\n {}.'.format(self.target_sei)+'\nMake sure it is a valid file.')
        self.error(os.path.exists(self.alignment),'Error opening file\n {}.'.format(self.alignment)+'\nMake sure it is a valid file.')


        # Load alignment
        self.align      = (scipy.io.loadmat(self.alignment)['refinedW']-1)[:,0]
        # self.align      = pd.read_csv(self.alignment)['2'].values -1
        self.num_frames = len(self.align)

        # Rect
        self.rect = []

        # self.error(os.path.exists(self.admult_filename), 
        #             'Error opening file\n {}.'.format(self.admult_filename)+
        #             '\nMake sure it is a valid file.')
        # self.error(os.path.exists(self.admult20_filename), 
        #             'Error opening file\n {}.'.format(self.admult_filename)+
        #             '\nMake sure it is a valid file.')
        # self.error(os.path.exists(self.admult40_filename), 
        #             'Error opening file\n {}.'.format(self.admult_filename)+
        #             '\nMake sure it is a valid file.')
        # self.error(os.path.exists(self.tcf_filename), 
        #             'Error opening folder\n {}.'.format(self.tcf_filename)+
        #             '\nMake sure it is a valid folder.')
        # self.error(os.path.exists(self.rf_filename), 
        #             'Error opening folder\n {}.'.format(self.rf_filename)+
        #             '\nMake sure it is a valid folder.')
        # self.error(os.path.exists(self.diss_filename), 
        #             'Error opening folder\n {}.'.format(self.diss_filename)+
        #             '\nMake sure it is a valid folder.')
        # self.error(os.path.exists(self.ref_sei_filename), 
        #             'Error opening folder\n {}.'.format(self.ref_sei_filename)+
        #             '\nMake sure it is a valid folder.')
        # self.error(os.path.exists(self.tar_sei_filename), 
        #             'Error opening folder\n {}.'.format(self.tar_sei_filename)+
        #             '\nMake sure it is a valid folder.')


        
        # admult file paths
        # admult20_filepath = admult_filepath.replace('admult/dissimilarity_video','admult_dilate/vid').replace('.avi','_dilate20.avi')
        # admult40_filepath = admult_filepath.replace('admult/dissimilarity_video','admult_dilate/vid').replace('.avi','_dilate40.avi')

        # # Initialising video Pair
        # self.ref_video      = Video(ref_filepath)
        # self.tar_video      = Video(tar_filepath)
        # self.admult_video   = Video(admult_filepath)
        # self.admult20_video = Video(admult20_filepath)
        # self.admult40_video = Video(admult40_filepath)
        # self.diss_video     = Video(diss_filepath)

        # self.rf_names = [os.path.join(rf_filepath, 'dissimilarity_fold{0:02d}.avi'.format(f)) \
        #                   for f in range(1,10)]
        # self.tcf_names = [os.path.join(tcf_filepath, 'dissimilarity_fold{0:02d}.avi'.format(f)) \
        #                   for f in range(1,10)]
            
        # self.rf_videos     = [Video(v) for v in self.rf_names]
        # self.tcf_videos    = [Video(v) for v in self.tcf_names]
        # self.admult_videos = [self.admult_video, self.admult20_video, self.admult40_video]

        self.mode      = None
        self.mask      = None
        self.K         = 6
        self.fold      = 0
        
        # Operations
        self.transform  = None
        self.net        = None
        self.operation  = None
        self.classifier = None
        
        self.homography = False

    def error(self, test, message):
        if not test:
            return QtWidgets.QMessageBox.critical(self, 'Error', (message))

    def set_mode(self,mode):

        self.mode = mode

        if self.mode == 'Resnet+RF':
            self.transform  = resnet_transform
            self.net        = Resnet50_Reduced('cuda' if torch.cuda.is_available() else 'cpu')
            self.net.freeze()
            self.operation  = None
            self.classifier = joblib.load(os.path.join(self.rf_folder,'testRF_fold{0:02d}.jbl'.format(self.fold)))
        if self.mode == 'Resnet+Dissim':
            self.transform  = resnet_transform
            self.net        = Resnet50_Reduced('cuda' if torch.cuda.is_available() else 'cpu')
            self.net.freeze()
            self.operation  = None
            self.classifier = None
        else:
            self.transform  = None
            self.net        = None
            self.operation  = None
            self.classifier = None            

  
    def get_frame(self, tar_idx, ref_idx=-1, offset=0):

        if ref_idx==-1:
            ref_idx = self.align[tar_idx]

        if ref_idx + offset >= self.num_frames:
            offset  = self.num_frames - ref_idx -1
        if ref_idx + offset < 0:
            offset = -ref_idx

        ref_frame = np.array(Image.open(os.path.join(self.reference_folder, 'frame_{0:05d}.png'.format(ref_idx+offset))))
        tar_frame = np.array(Image.open(os.path.join(self.target_folder, 'frame_{0:05d}.png'.format(tar_idx))))
        ret_frame = np.zeros_like(tar_frame)

        # homography
        if self.homography:
            
             # Initiate SIFT detector
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(ref_frame,None)
            kp2, des2 = sift.detectAndCompute(tar_frame,None)

            index_params  = dict(algorithm = 1, trees = 5)
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
            ref_frame = cv2.warpPerspective(ref_frame, H, (ref_frame.shape[1], ref_frame.shape[0]))

        if self.mode == 'Resnet+RF':

            ref_frame, tar_frame, ret_frame = self.run_resnetRF(ref_frame,tar_frame)

        if self.mode == 'Resnet+Dissim':

            ref_frame, tar_frame, ret_frame = self.run_resnetDissim(ref_frame,tar_frame)

        # frames_tcf = np.array([i.get_frame(idx) for i in self.tcf_videos])
        # frames_rf  = np.array([i.get_frame(idx) for i in self.rf_videos])

        # ref_frame   = self.ref_video.get_frame(idx)
        # tar_frame   = self.tar_video.get_frame(idx)


        # # Masks 
        # if self.mask == 'ADMULT':
        #     mask_frame = self.admult_videos[0].get_frame(idx)       
        #     mask_frame = (mask_frame>127).astype('uint8')
            
        # elif self.mask == 'ADMULT-20':
        #     mask_frame = self.admult_videos[1].get_frame(idx)       
        #     mask_frame = (mask_frame>127).astype('uint8')
  
        # elif self.mask == 'ADMULT-40':
        #     mask_frame = self.admult_videos[2].get_frame(idx)       
        #     mask_frame = (mask_frame>127).astype('uint8')
        # else: 
        #     mask_frame = np.ones_like(tar_frame)

        # if self.mode == 'TCF-LMO':
        #     frames_tcf = (frames_tcf > 127)

        #     if self.fold==0:
        #         class_frame = np.sum(frames_tcf, axis=0)
        #         class_frame = (255*(class_frame/9)).astype('uint8')
        #     elif self.fold in list(range(1,10)):
        #         class_frame = frames_tcf[self.fold-1]
        #         class_frame = 255*class_frame.astype('uint8')
        #     else:
        #         class_frame = np.zeros_like(self.tar_video.get_frame(idx))
  
        # elif self.mode == 'Resnet+RF':
        #     frames_rf = (frames_rf > 127)
        #     if self.fold==0:
        #         class_frame = np.sum(frames_rf, axis=0)
        #         class_frame = (255*(class_frame/9)).astype('uint8')
        #     elif self.fold in list(range(1,10)):
        #         class_frame = frames_rf[self.fold-1]
        #         class_frame = 255*class_frame.astype('uint8')
        #     else:
        #         class_frame = np.zeros_like(self.tar_video.get_frame(idx))
        # elif self.mode == 'K-means':
        #     class_frame = self.kmeans(ref_frame, tar_frame, self.K)
        # elif self.mode == 'Resnet+Dissim':
        #     class_frame = self.diss_video.get_frame(idx)    
        # elif self.mode == None:
        #     class_frame = np.zeros_like(self.tar_video.get_frame(idx))
        # else: 
        #     pass


        # # Bounding box
        if len(self.rect)>0:
            rect_img = np.zeros_like(tar_frame)

            for rect_vec in self.rect:
                rect_img[rect_vec[1]:rect_vec[1]+rect_vec[3],rect_vec[0]:rect_vec[0]+rect_vec[2]] = 1
        else:
            rect_img = np.ones_like(tar_frame)

        # if 0 in mask_frame:
        #     mask_frame = mask_frame*rect_img
        # else:
        #     mask_frame = rect_img

        # Mask multiplication
        ret_frame = rect_img*ret_frame


        # Cleaning bounding box
        self.rect = []
        # if self.mask != None and self.mode == None:
        #     ret_frame = 255*mask_frame
        # elif self.mask == None and self.mode != None:
        #     ret_frame = mask_frame*class_frame
        # elif self.mask == None and self.mode == None:
        #     ret_frame = np.zeros_like(tar_frame)
        # else:
        #     ret_frame = mask_frame*class_frame

        return ref_frame, tar_frame, ret_frame

    def kmeans(self, ref_frame, tar_frame, K):
        h, _, _ = tar_frame.shape
        Z = np.concatenate((ref_frame,tar_frame))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
        _, label, center = cv2.kmeans( np.float32(Z.reshape((-1,3))),K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res    = cv2.cvtColor(center[label.flatten()].reshape((Z.shape)).astype('uint8'), cv2.COLOR_RGB2GRAY)
        diff   = np.abs(res[:h]-res[h:])
        return cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)

    def run_resnetRF(self, ref, tar):

        # Transform
        ref_feat = self.transform(ref)
        tar_feat = self.transform(tar)

        # Net
        ref_feat = self.net(ref_feat[None,:,:,:])
        tar_feat = self.net(tar_feat[None,:,:,:])

        # Operations
        feat = torch.concat((ref_feat,tar_feat), axis=1)[0].T.reshape((-1,512))
        pred = (255 * self.classifier.predict_proba(feat.detach().numpy())[:,1].reshape((200,113)).T)
        pred = cv2.resize(pred, (800,450), interpolation = cv2.INTER_LINEAR)
        pred = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Classifier
        return ref, tar, pred

    def run_resnetDissim(self, ref, tar):

        # Transform
        ref_feat = self.transform(ref)
        tar_feat = self.transform(tar)

        # Net
        diff     = torch.sqrt(torch.sum(( self.net(ref_feat[None,:,:,:])[0] - self.net(tar_feat[None,:,:,:])[0]  )**2, 0))

        diff     = (255*((diff-diff.min())/ (diff.max()-diff.min()))).detach().numpy().astype('uint8')

        thr      = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
        std      = np.std(diff)
        nrm      = (diff.astype(float)-thr)/std
        logistic = 1/(1+np.exp(nrm))

        pred    = 255-(255*logistic).astype('uint8')
        pred    = cv2.resize(pred, (800,450))
        pred    = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Classifier
        return ref, tar, pred

class Video:                                                                                          
                                                                                                                                   
    def __init__(self, filepath):                                                                                                 

        # Get capture
        self.video  = cv2.VideoCapture(filepath)

        self.filepath   = filepath
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width      = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height     = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps        = self.video.get(cv2.CAP_PROP_FPS)

    def get_frame(self, num):

        if (num < self.num_frames) and (num >= 0):

            _ = self.video.set(cv2.CAP_PROP_POS_FRAMES, num)
            _, frame = self.video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

        else:

            self.video.release()
            return None
                                  
    def __str__(self):
        return self.filepath

    def __len__(self):
        return self.num_frames



     