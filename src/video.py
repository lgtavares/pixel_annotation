import cv2
import os
import numpy as np

class VideoPair:
    def __init__(self, ref_filepath, tar_filepath, rf_filepath,
                 tcf_filepath, admult_filepath, diss_filepath):
        
        # admult file paths
        admult20_filepath = admult_filepath.replace('admult/dissimilarity_video','admult_dilate/vid').replace('.avi','_dilate20.avi')
        admult40_filepath = admult_filepath.replace('admult/dissimilarity_video','admult_dilate/vid').replace('.avi','_dilate40.avi')

        # Initialising video Pair
        self.ref_video   = Video(ref_filepath)
        self.tar_video   = Video(tar_filepath)
        self.admult_video   = Video(admult_filepath)
        self.admult20_video = Video(admult20_filepath)
        self.admult40_video = Video(admult40_filepath)
        self.diss_video  = Video(diss_filepath)

        self.rf_names = [os.path.join(rf_filepath, 'dissimilarity_fold{0:02d}.avi'.format(f)) \
                          for f in range(1,10)]
        self.tcf_names = [os.path.join(tcf_filepath, 'dissimilarity_fold{0:02d}.avi'.format(f)) \
                          for f in range(1,10)]
            
        self.rf_videos     = [Video(v) for v in self.rf_names]
        self.tcf_videos    = [Video(v) for v in self.tcf_names]
        self.admult_videos = [self.admult_video, self.admult20_video, self.admult40_video]

        self.mode      = None
        self.mask      = None
        self.K         = 6
        self.fold      = 0
        self.rect      = None
  
    def get_frame(self, idx):

        frames_tcf = np.array([i.get_frame(idx) for i in self.tcf_videos])
        frames_rf  = np.array([i.get_frame(idx) for i in self.rf_videos])

        ref_frame   = self.ref_video.get_frame(idx)
        tar_frame   = self.tar_video.get_frame(idx)
        mask_frame  = np.ones_like(ref_frame)
        class_frame = np.zeros_like(ref_frame)

        # Masks 
        if self.mask == 'ADMULT':
            mask_frame = self.admult_videos[0].get_frame(idx)       
            mask_frame = (mask_frame>127).astype('uint8')
            
        elif self.mask == 'ADMULT-20':
            mask_frame = self.admult_videos[1].get_frame(idx)       
            mask_frame = (mask_frame>127).astype('uint8')
  
        elif self.mask == 'ADMULT-40':
            mask_frame = self.admult_videos[2].get_frame(idx)       
            mask_frame = (mask_frame>127).astype('uint8')
        else: 
            mask_frame = np.ones_like(tar_frame)

        if self.mode == 'TCF-LMO':
            frames_tcf = (frames_tcf > 127)

            if self.fold==0:
                class_frame = np.sum(frames_tcf, axis=0)
                class_frame = (255*(class_frame/9)).astype('uint8')
            elif self.fold in list(range(1,10)):
                class_frame = frames_tcf[self.fold-1]
                class_frame = 255*class_frame.astype('uint8')
            else:
                class_frame = np.zeros_like(self.tar_video.get_frame(idx))
  
        elif self.mode == 'Resnet+RF':
            frames_rf = (frames_rf > 127)
            if self.fold==0:
                class_frame = np.sum(frames_rf, axis=0)
                class_frame = (255*(class_frame/9)).astype('uint8')
            elif self.fold in list(range(1,10)):
                class_frame = frames_rf[self.fold-1]
                class_frame = 255*class_frame.astype('uint8')
            else:
                class_frame = np.zeros_like(self.tar_video.get_frame(idx))
        elif self.mode == 'K-means':
            class_frame = self.kmeans(ref_frame, tar_frame, self.K)
        elif self.mode == 'Resnet+Dissim':
            class_frame = self.diss_video.get_frame(idx)    
        elif self.mode == None:
            class_frame = np.zeros_like(self.tar_video.get_frame(idx))
        else: 
            pass


        # Bounding box
        rect_img = np.ones_like(tar_frame)
        if len(self.rect)>0:
            rect_img = np.zeros_like(tar_frame)

            for rect_vec in self.rect:
                rect_img[rect_vec[1]:rect_vec[1]+rect_vec[3],rect_vec[0]:rect_vec[0]+rect_vec[2]] = 1

        if 0 in mask_frame:
            mask_frame = mask_frame*rect_img

        else:
            mask_frame = rect_img

        # Mask multiplication
        if self.mask != None and self.mode == None:
            ret_frame = 255*mask_frame
        else:
            ret_frame = mask_frame*class_frame

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



     