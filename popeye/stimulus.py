"""

First pass at a stimulus model for abstracting the qualities and functionality of a stimulus
into an abstract class.  For now, we'll assume the stimulus model only pertains to visual 
stimuli on a visual display over time (i.e., 3D).  Hopefully this can be extended to other stimuli
with an arbitrary number of dimensions (e.g., auditory stimuli).

"""
from __future__ import division
import ctypes

from multiprocessing import Process, Queue, Array
from multiprocessing.managers import BaseManager

import numpy as np
from scipy.misc import imresize
from scipy.io import loadmat

import nibabel

def generate_coordinate_matrices(pixels_across, pixels_down, ppd, scale_factor=1):
    
    """Creates coordinate matrices for representing the visual field in terms
       of degrees of visual angle.
       
    This function takes the screen dimensions, the pixels per degree, and a
    scaling factor in order to generate a pair of ndarrays representing the
    horizontal and vertical extents of the visual display in degrees of visual
    angle.
    
    Parameters
    ----------
    pixels_across : int
        The number of pixels along the horizontal extent of the visual display.
    pixels_down : int
        The number of pixels along the vertical extent of the visual display.
    ppd: float
        The number of pixels that spans 1 degree of visual angle.  This number
        is computed using the display width and the viewing distance.  See the
        config.init_config for details. 
    scale_factor : float
        The scale factor by which the stimulus is resampled.  The scale factor
        must be a float, and must be greater than 0.
        
    Returns
    -------
    deg_x : ndarray
        An array representing the horizontal extent of the visual display in
        terms of degrees of visual angle.
    deg_y : ndarray
        An array representing the vertical extent of the visual display in
        terms of degrees of visual angle.
    """
    
    [X,Y] = np.meshgrid(np.arange(np.round(pixels_across*scale_factor)),
                        np.arange(np.round(pixels_down*scale_factor)))
                        
                        
    deg_x = (X-np.round(pixels_across*scale_factor)/2)/(ppd*scale_factor)
    deg_y = (Y-np.round(pixels_down*scale_factor)/2)/(ppd*scale_factor)
    
    deg_x += 0.5/(ppd*scale_factor)
    deg_y += 0.5/(ppd*scale_factor)
    
    return deg_x, deg_y

def resample_stimulus(stim_arr, scale_factor=0.05):
    
    """Resamples the visual stimulus
    
    The function takes an ndarray `stim_arr` and resamples it by the user
    specified `scale_factor`.  The stimulus array is assumed to be a three
    dimensional ndarray representing the stimulus, in screen pixel coordinates,
    over time.  The first two dimensions of `stim_arr` together represent the
    exent of the visual display (pixels) and the last dimensions represents
    time (TRs).
    
    Parameters
    ----------
    stim_arr : ndarray
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.
    scale_factor : float
        The scale factor by which the stimulus is resampled.  The scale factor
        must be a float, and must be greater than 0.
        
    Returns
    -------
    resampledStim : ndarray
        An array that is resampled according to the user-specified scale factor.
    """
    
    dims = np.shape(stim_arr)
    
    resampled_arr = np.zeros((dims[0]*scale_factor, dims[1]*scale_factor, dims[2]))
    
    for tr in np.arange(dims[-1]):
        resampled_arr[:,:,tr] = imresize(stim_arr[:,:,tr], scale_factor)
    
    return resampled_arr.astype('short')
    
def load_stimulus_file(stim_path):
    
    if stim_path.endswith('.npy'):
        stim_arr = np.load(stim_path)
    elif stim_path.endswith('.mat'):
        file_name = stim_path.split("/")[-1]
        stim_arr = loadmat(file_name)[file_name]
    elif stim_path.endswith('.nii.gz'):
        stim_arr = nibabel.load(stim_path).get_data()
    
    return stim_arr.astype('short')

def gaussian_2D(X, Y, x0, y0, sigma_x, sigma_y, degrees, amplitude=1):
    
    theta = degrees*2*np.pi/360
    
    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    
    Z = amplitude*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    
    return Z

def simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, 
                         thetas, num_steps, stim_ecc, blanks=True, threshold = 0.33):
    
    # visuotopic stuff
    ppd = np.pi*pixels_across/np.arctan(screen_width/viewing_distance/2.0)/360.0 # degrees of visual angle
    deg_x, deg_y = generate_coordinate_matrices(pixels_across, pixels_down, ppd, 1.0)
    
    # initialize a counter
    tr_num = 0
    
    # insert blanks
    if blanks:
        thetas = list(thetas)
        thetas.insert(0,-1)
        thetas.append(-1)
        
    # initialize the stimulus array
    bar_stimulus = np.zeros((pixels_down, pixels_across, len(thetas)*num_steps))
    
    # main loop
    for theta in thetas:
        
        if theta != -1:
            
            theta_rad = theta * np.pi / 180
            
            # get the starting point and trajectory
            start_pos = np.array([-np.cos(theta_rad)*stim_ecc, np.sin(theta_rad)*stim_ecc])
            end_pos = np.array([np.cos(theta_rad)*stim_ecc, -np.sin(theta_rad)*stim_ecc])
            run_and_rise = end_pos - start_pos;
            
            # step through each position along the trajectory
            for step in np.arange(0,num_steps):
                
                # get the position of the bar at each step
                xy0 = run_and_rise * step/num_steps + start_pos
                
                # generate the gaussian
                Z = gaussian_2D(deg_x,deg_y,xy0[0],xy0[1],1,100,theta)
                
                # store and iterate
                bar_stimulus[:,:,tr_num] = Z
                tr_num += 1
                
        else:
            for step in np.arange(0,num_steps):
                tr_num += 1
                
    
    bar_digital = np.zeros_like(bar_stimulus)
    bar_digital[bar_stimulus > threshold] = 1
    bar_digital = np.short(bar_digital)
    
    return bar_digital


# This should eventually be VisualStimulus, and there would be an abstract class layer
# above this called Stimulus that would be generic for n-dimentional feature spaces.
class Stimulus(object):
    
    """ Abstract class for stimulus model """
    
    def __init__(self, stim_arr, viewing_distance, screen_width, scale_factor, clip_number=0, roll_number=0):
        
        # absorb the vars
        self.viewing_distance = viewing_distance
        self.screen_width = screen_width
        self.scale_factor = scale_factor
        self.clip_number = clip_number
        self.roll_number = roll_number
        
        # trim and rotate stimulus is specified
        if self.clip_number != 0:
            stim_arr = stim_arr[:, :, self.clip_number::]
        if self.roll_number != 0:
            stim_arr = np.roll(stim_arr, self.roll_number, axis=-1)
        
        # ascertain stimulus features
        self.pixels_across = np.shape(stim_arr)[1]
        self.pixels_down = np.shape(stim_arr)[0]
        self.run_length = np.shape(stim_arr)[2]
        self.ppd = np.pi*self.pixels_across/np.arctan(self.screen_width/self.viewing_distance/2.0)/360.0 # degrees of visual angle
        
        # create down-sampled stimulus
        stim_arr_coarse = resample_stimulus(stim_arr,self.scale_factor)
        
        # generate the coordinate matrices
        deg_x, deg_y = generate_coordinate_matrices(self.pixels_across, self.pixels_down, self.ppd)
        deg_x_coarse, deg_y_coarse = generate_coordinate_matrices(self.pixels_across, self.pixels_down, self.ppd, self.scale_factor)
        
        # share the arrays via memmap to reduce size
        self.deg_x = np.memmap('%s%s.npy' %('/tmp/','deg_x'),dtype = np.double, mode = 'w+',shape = np.shape(deg_x))
        self.deg_x[:] = deg_x[:]
        
        self.deg_y = np.memmap('%s%s.npy' %('/tmp/','deg_y'),dtype = ctypes.c_double, mode = 'w+',shape = np.shape(deg_y))
        self.deg_y[:] = deg_y[:]
        
        self.deg_x_coarse = np.memmap('%s%s.npy' %('/tmp/','deg_x_coarse'),dtype = ctypes.c_double, mode = 'w+',shape = np.shape(deg_x_coarse))
        self.deg_x_coarse[:] = deg_x_coarse[:]
        
        self.deg_y_coarse = np.memmap('%s%s.npy' %('/tmp/','deg_y_coarse'),dtype = ctypes.c_double, mode = 'w+',shape = np.shape(deg_y_coarse))
        self.deg_y_coarse[:] = deg_y_coarse[:]
        
        self.stim_arr = np.memmap('%s%s.npy' %('/tmp/','stim_arr'),dtype = ctypes.c_short, mode = 'w+',shape = np.shape(stim_arr))
        self.stim_arr[:] = stim_arr[:]
        
        self.stim_arr_coarse = np.memmap('%s%s.npy' %('/tmp/','stim_arr_coarse'),dtype = ctypes.c_short, mode = 'w+',shape = np.shape(stim_arr_coarse))
        self.stim_arr_coarse[:] = stim_arr_coarse[:]
        
        
