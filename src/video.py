import cv2
from frame import Frame

class Video:                                                                                          
                                                                                                                                   
    def __init__(self, filepath):                                                                                                 

        # Get capture
        video  = cv2.VideoCapture(filepath)

        self.filepath   = filepath
        self.num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width      = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height     = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps        = video.get(cv2.CAP_PROP_FPS)

    def get_frame(self, num):
        video  = cv2.VideoCapture(self.filepath)

        if (num < self.num_frames) and (num >= 0):

            _ = video.set(cv2.CAP_PROP_POS_FRAMES, num)
            _, frame_array = video.read()
            frame = Frame(frame_array, num)
            video.release()

            return frame.to_rgb()

        else:

            video.release()
            return None
                                  
    def __str__(self):
        return self.filepath

    def __len__(self):
        return self.num_frames



     