import cv2

class VideoPair:
    def __init__(self, ref_filepath, tar_filepath):
        
        # Initialising video Pair
        self.ref_video = Video(ref_filepath)
        self.tar_video = Video(tar_filepath)
    
    def get_frame(self, idx):

        return self.ref_video.get_frame(idx), self.tar_video.get_frame(idx)


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



     