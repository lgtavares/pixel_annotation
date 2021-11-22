from video import Video

class Annotation():

    """Class to handle the annotations.

    This class is designed to handle the properties and status of the annotation process.

    Attributes:
        ref_video (Video): Reference video without anomalies.
        tar_video (Video): Target video to be annotated.

    """    

    def __init__(self, ref_video, tar_video):

        # Loading objectis
        self.ref_video = Video(ref_video)       
        self.tar_video = Video(tar_video)
