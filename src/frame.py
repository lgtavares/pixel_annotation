import cv2
from video import Video

class Frame(Video):
 
    def __init__(self, array, index):   

        self.array = array
        self.index = index
        self.colorspace = 'bgr'
        self.width  = Video.width
        self.height = Video.height

    def to_rgb(self):
        return cv2.cvtColor(self.array, cv2.COLOR_BGR2RGB)