import os
import pickle
import pandas as pd
import numpy as np

class Settings():

    def __init__(self, num_frames=0):

        self.names = ['annotated', 'has_object', 'mask', 'algorithm',
                      'fold', 'K', 'opening', 'closing', 'erosion', 'threshold',
                      'contour_fg', 'contour_dc', 'anchor', 'consistency', 'transform', 'bbox']

        self.num_frames  = num_frames
        self.settings_df = pd.DataFrame(index=np.arange(self.num_frames),columns=self.names)
        
        # populate the frames
        self.__zero_condition = [False, None, None, None, 0, 2, 0, 0, 0, 127, [], [], 0, None, None, []]
        for k in range(self.num_frames):
            self.settings_df.loc[k] = self.__zero_condition 

        # Actual position
        self.frame = 0

    def set_frame(self, frame):
        self.frame = frame

    def set_value(self, setting, value):
        self.settings_df.loc[self.frame, setting] = value

    def set_row(self, row):
        if len(row) == len(self.names):
            self.settings_df.iloc[self.frame,:] = np.array(row)

    def get_value(self, setting):
        return self.settings_df.loc[self.frame, setting]

    def get_row(self):
        return self.settings_df.loc[self.frame]

    def get_anchor_value(self, setting):
        anchor_frame = self.settings_df.loc[self.frame]['anchor']
        return self.settings_df.loc[anchor_frame, setting]

    def get_anchor_row(self):
        anchor_frame = self.settings_df.loc[self.frame]['anchor']
        return self.settings_df.loc[anchor_frame]

    def propagate_previous(self):
        self.set_frame(self.frame+1)
        if self.settings_df.loc[self.frame-1,'annotated']:
            self.settings_df.loc[self.frame,:] = self.settings_df.loc[self.frame-1,:].copy()

    def get_row_str(self, frame):
        row = self.settings_df.loc[frame]
        print_str  = 'Frame {0:>4d}:'.format(frame+1)
        print_str += '{0},'.format(row['has_object'])
        print_str += '{0},'.format(row['annotated'])
        print_str += '{0},'.format(row['mask'])
        print_str += '{0},['.format(row['algorithm'])
        print_str += '{0},'.format(row['opening'])
        print_str += '{0},'.format(row['closing'])
        print_str += '{0},'.format(row['erosion'])
        print_str += '{0}],['.format(row['threshold'])
        print_str += '{0},'.format(row['anchor'])
        print_str += '{0}],'.format(row['consistency'])
        print_str += '[{0},'.format(len(row['contour_fg']))
        print_str += '{0}]'.format(len(row['contour_dc']))
        print_str += '[{0}]'.format(len(row['bbox']))
        print_str += '\n'

        return print_str

    def to_dict(self,d):
        return d.T.to_dict()

    def to_df(self,d):
        self.settings_df = pd.DataFrame.from_dict(d).T

    def load(self,filename):

        if os.path.exists(filename):
            with open(filename, 'rb') as input_file:
                self.settings_df = self.to_df(pickle.load(input_file))
                last_frame = list(self.settings_df.annotated).index(False)-1
                self.set_frame(last_frame)
        else:
            self.save(filename)

    def save(self, filename):
        with open(filename, 'wb') as output_file:
            pickle.dump(self.to_dict(self.settings_df), output_file)

    def save_settings(self):
        with open(self.tar_filename.replace('.avi','.ann'), 'wb') as output_file:
            pickle.dump(self.settings, output_file)
            self.ann_text.insertPlainText('Settings saved !!\n')
