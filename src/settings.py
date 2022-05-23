import os
import pickle
import pandas as pd
import numpy as np

class Settings():

    def __init__(self, num_frames=0, align=[]):

        self.names = ['target_frame', 'reference_frame', 'annotated', 'has_object', 'mask', 'algorithm',
                      'fold', 'K', 'opening', 'closing', 'erosion', 'threshold',
                      'contour_fg', 'contour_dc', 'anchor', 'consistency', 'transform', 'bbox', 'offset', 'homography', 'H_mat']

        self.num_frames  = num_frames
        self.align       = align
        self.df = pd.DataFrame(index=np.arange(self.num_frames),columns=self.names)
        
        # populate the frames
        self.__zero_condition = [0, 0, False, None, None, None, 0, 2, 0, 0, 0, 127, [], [], 0, None, None, [], 0, False, None]

        for k in range(self.num_frames):
            self.df.iloc[k] =  np.array(self.__zero_condition, dtype=np.object).copy()
        if len(self.align) > 0:
            self.df['target_frame']    = np.arange(self.num_frames)
            self.df['reference_frame'] = self.align

        # Actual position
        self.frame = 0

    def set_frame(self, frame):
        self.frame = frame

    def set_value(self, setting, value):

        self.df.at[self.frame, setting] = value

    def set_row(self, row):
        if len(row) == len(self.names):
            self.df.loc[self.frame,:] = np.array(row)

    def get_value(self, setting):
        return self.df.loc[self.df['target_frame']==self.frame, setting].values[0]

    def get_row(self,frame):
        return self.df.loc[self.df['target_frame']==frame, :]

    def get_anchor_value(self, setting):
        anchor_frame = self.df.loc[self.df['target_frame']==self.frame, 'anchor'].values[0]
        return self.df.loc[self.df['target_frame']==anchor_frame, setting].values[0]

    def get_anchor_row(self):
        anchor_frame = self.df.loc[self.df['target_frame']==self.frame, 'anchor'].values[0]
        return self.df.loc[self.df['target_frame']==anchor_frame, :]

    def propagate_previous(self):
        propag_columns = ['mask', 'algorithm', 'opening', 'closing', 'erosion', 'threshold', 'anchor', 'consistency']
        #self.set_frame(self.frame+1)
        if self.frame > 0:
            if self.df.loc[self.df['target_frame']==self.frame-1,'annotated'].values[0] &  ~ self.df.loc[self.df['target_frame']==self.frame,'annotated'].values[0] :
                self.df.loc[self.df['target_frame']==self.frame,propag_columns] = self.df.loc[self.df['target_frame']==self.frame-1,propag_columns].copy().values
                self.df.loc[self.df['target_frame']==self.frame-1,'annotated']  = True


    def get_row_str(self, frame):
        row = self.get_row(frame)
        
        print_str  = 'Frame [{0:>4d}/{1:>4d}]:'.format(frame+1,row['reference_frame'].values[0]+1)
        print_str += '{0},'.format(row['annotated'].values[0])
        print_str += '{0},'.format(row['mask'].values[0])
        print_str += '{0},['.format(row['algorithm'].values[0])
        print_str += '{0},'.format(row['opening'].values[0])
        print_str += '{0},'.format(row['closing'].values[0])
        print_str += '{0},'.format(row['erosion'].values[0])
        print_str += '{0}],['.format(row['threshold'].values[0])
        print_str += '{0},'.format(row['anchor'].values[0]+1)
        print_str += '{0}],'.format(row['consistency'].values[0])
        print_str += '[{0},'.format(len(row['contour_fg'].values[0]))
        print_str += '{0}]'.format(len(row['contour_dc'].values[0]))
        print_str += '[{0}]'.format(len(row['bbox'].values[0]))
        print_str += '[{0}, {1}]'.format(row['offset'].values[0], row['homography'].values[0])
        print_str += '\n'

        return print_str

    def to_dict(self,d):
        return d.T.to_dict()

    def to_df(self,d):
        return pd.DataFrame.from_dict(d).T

    def load(self,filename):
        load_columns = ['target_frame', 'reference_frame','annotated', 'has_object', 'mask', 'algorithm',
                      'fold', 'K', 'opening', 'closing', 'erosion', 'threshold',
                      'contour_fg', 'contour_dc', 'anchor', 'consistency', 'transform', 
                      'bbox', 'offset', 'homography', 'H_mat']
        if os.path.exists(filename):
            input_df =  pickle.load(open(filename, 'rb'))  
            self.df[load_columns] = self.to_df(input_df)
            last_frame = list(self.df.annotated).index(False)-1
            self.set_frame(last_frame)
        else:
            self.save(filename)

    def save(self, filename):
        out_df = self.df.copy()
        out_df.at[self.frame, 'annotated'] = True
        with open(filename, 'wb') as output_file:
            pickle.dump(self.to_dict(out_df), output_file)

    def save_settings(self):
        with open(self.tar_filename.replace('.avi','.ann'), 'wb') as output_file:
            pickle.dump(self.settings, output_file)
            self.ann_text.insertPlainText('Settings saved !!\n')
