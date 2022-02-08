import cv2
import os
import numpy as np

class VideoPair:
    def __init__(self, ref_filepath, tar_filepath, rf_filepath,
                 tcf_filepath, daomc_filepath):
        
        # Initialising video Pair
        self.ref_video   = Video(ref_filepath)
        self.tar_video   = Video(tar_filepath)
        self.daomc_video = Video(daomc_filepath)

        self.rf_names = [os.path.join(rf_filepath, 'dissimilarity_fold{0:02d}.avi'.format(f)) \
                          for f in range(1,10)]
        self.tcf_names = [os.path.join(tcf_filepath, 'dissimilarity_fold{0:02d}.avi'.format(f)) \
                          for f in range(1,10)]
            
        self.rf_videos  = [Video(v) for v in self.rf_names]
        self.tcf_videos = [Video(v) for v in self.tcf_names]

        self.mode      = None
        self.fold      = 0

  
    def get_frame(self, idx):

        if self.mode == None:
            return self.ref_video.get_frame(idx), \
                   self.tar_video.get_frame(idx), \
                   np.zeros_like(self.tar_video.get_frame(idx))

        elif self.mode == 'DAOMC':

            return self.ref_video.get_frame(idx), \
                    self.tar_video.get_frame(idx), \
                    self.daomc_video.get_frame(idx)       

        elif self.mode == 'TCF-LMO':

            frames_arr = np.array([i.get_frame(idx) for i in self.tcf_videos])
            frames_arr = (frames_arr > 127)
            if self.fold==0:
                net_img = np.sum(frames_arr, axis=0)
                net_img = (255*(net_img/9)).astype('uint8')
            elif self.fold in list(range(1,10)):
                net_img = frames_arr[self.fold-1]
                net_img = 255*net_img.astype('uint8')
            else:
                net_img = np.zeros_like(self.tar_video.get_frame(idx))
            return  self.ref_video.get_frame(idx), \
                    self.tar_video.get_frame(idx), \
                    net_img          

        elif self.mode == 'Resnet+RF':
            frames_arr = np.array([i.get_frame(idx) for i in self.rf_videos])
            frames_arr = (frames_arr > 127)
            if self.fold==0:
                net_img = np.sum(frames_arr, axis=0)
                net_img = (255*(net_img/9)).astype('uint8')
            elif self.fold in list(range(1,10)):
                net_img = frames_arr[self.fold-1]
                net_img = 255*net_img.astype('uint8')
            else:
                net_img = np.zeros_like(self.tar_video.get_frame(idx))
            return  self.ref_video.get_frame(idx), \
                    self.tar_video.get_frame(idx), \
                    net_img    
        else:   
            return self.ref_video.get_frame(idx), \
                   self.tar_video.get_frame(idx), \
                   np.zeros_like(self.tar_video.get_frame(idx))

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



     