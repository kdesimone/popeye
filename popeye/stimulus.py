"""

Base-classes for voxel-wise encoding models and voxel-wise encoding model fits.

All the models in the reconst module follow the same template: a Model object
is used to represent the abstract properties of the model, that are independent
of the specifics of the data . These properties are reused whenver fitting a
particular set of data (different voxels, for example).


"""
from __future__ import division

import numpy as np
from scipy.misc import imresize
from scipy.io import loadmat

import nibabel


class Stimulus(object):
    """ Abstract class for stimulus models
    """
    def __init__(self, stim_path, viewing_distance, screen_width, scale_factor):
        """
        
        """
        
        # absorb the vars
        self.stim_path = stim_path
        self.viewing_distance = viewing_distance
        self.screen_width = screen_width
        self.scale_factor = scale_factor
        
        # load the stimulus assuming npy format
        self.stim_arr = self.load_stimulus_file(self.stim_path)
        
        # ascertain stimulus features
        self.pixels_across = np.shape(self.stim_arr)[1]
        self.pixels_down = np.shape(self.stim_arr)[0]
        self.run_length = np.shape(self.stim_arr)[2]
        self.ppd = np.pi*self.pixels_across/np.arctan(self.screen_width/self.viewing_distance/2.0)/360.0 # degrees of visual angle
        
        # generate the coordinate matrices
        self.deg_x, self.deg_y = self.generate_coordinate_matrices(1.0)
        
        # generate the coarse arrays
        self.deg_x_coarse, deg_y_coarse = self.generate_coordinate_matrices(self.scale_factor)
        self.stim_arr_coarse = self.resample_stimulus(self.scale_factor)
        
    def generate_coordinate_matrices(self, scale_factor=1):
        
        [X,Y] = np.meshgrid(np.arange(np.round(self.pixels_across*scale_factor)),
                            np.arange(np.round(self.pixels_down*scale_factor)))
                            
                            
        deg_x = (X-np.round(self.pixels_across*scale_factor)/2)/(self.ppd*scale_factor)
        deg_y = np.flipud((Y-np.round(self.pixels_down*scale_factor)/2)/(self.ppd*scale_factor))
        
        deg_x += 0.5/(self.ppd*scale_factor)
        deg_y += 0.5/(self.ppd*scale_factor)
        
        return deg_x, deg_y
        
    def resample_stimulus(self,scale_factor=0.05):
        
        dims = np.shape(self.stim_arr)
        
        resampled_arr = np.zeros((dims[0]*scale_factor, dims[1]*scale_factor, dims[2]))
        
        for tr in np.arange(self.run_length):
            resampled_arr[:,:,tr] = imresize(self.stim_arr[:,:,tr], scale_factor)
            
        resampled_arr[resampled_arr > 0] = 1
        
        return resampled_arr
        
    ### Do I return self.stim_arr instead of stim_arr?
    def load_stimulus_file(self,stim_path):
        
        if self.stim_path.endswith('.npy'):
            stim_arr = np.load(self.stim_path)
        elif self.stim_path.endswith('.mat'):
            file_name = self.stim_path.split("/")[-1]
            stim_arr = loadmat(file_name)[file_name]
        elif self.stim_path.endswith('.nii.gz'):
            stim_arr = nibabel.load(self.stim_path).get_data()
        
        return stim_arr
            
            
