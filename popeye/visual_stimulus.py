"""

First pass at a stimulus model for abstracting the qualities and functionality of a stimulus
into an abstract class.  For now, we'll assume the stimulus model only pertains to visual 
stimuli on a visual display over time (i.e., 3D).  Hopefully this can be extended to other stimuli
with an arbitrary number of dimensions (e.g., auditory stimuli).

"""

from __future__ import division
import ctypes
import gc
import sys  

import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.io import loadmat
from scipy.signal import square

from popeye.base import StimulusModel
import popeye.utilities as utils

def pixels_per_degree(pixels_across, screen_width, viewing_distance):

    return np.pi*pixels_across/np.arctan(screen_width/viewing_distance/2.0)/360.0
    
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
    
    return deg_x, np.flipud(deg_y)

def resample_stimulus(stim_arr, scale_factor=0.05, mode='nearest', dtype='uint8'):
    
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
    
    mode : str, optional
        Points outside the boundaries of the input are filled according
        to the given mode ('constant', 'nearest', 'reflect' or 'wrap').
        Default is 'nearest'.
        
        
    Returns
    -------
    resampled_arr : ndarray
        An array that is resampled according to the user-specified scale factor.
    """
    
    dims = np.shape(stim_arr)
    
    resampled_arr = np.zeros((dims[0]*scale_factor, dims[1]*scale_factor, dims[2]),dtype=dtype)
    
    # loop
    for tr in np.arange(dims[-1]):
        
        # resize it
        f = zoom(stim_arr[:,:,tr], scale_factor, mode=mode)
        
        # insert it
        resampled_arr[:,:,tr] = f
    
    return resampled_arr

def gaussian_2D(X, Y, x0, y0, sigma_x, sigma_y, degrees, amplitude=1):
    
    """
    A utility function for creating a two-dimensional Gaussian.
    
    This function served as the model for the Cython implementation of a
    generator of two-dimensional Gaussians.
    
    X : ndarray
        The coordinate array for the horizontal dimension of the display.
        
    Y : ndarray
        The coordinate array for the the vertical dimension of the display.
    
    x0 : float
        The location of the center of the Gaussian on the horizontal axis.
        
    y0 : float
        The location of the center of the Gaussian on the verticala axis.
    
    sigma_x : float
        The dispersion of the Gaussian over the horizontal axis.
    
    sigma_y : float
        The dispersion of the Gaussian over the vertical axis.
    
    degrees : float
        The orientation of the two-dimesional Gaussian (degrees).
    
    amplitude : float
        The amplitude of the two-dimensional Gaussian.
    
    Returns
    -------
    gauss : ndarray
        The two-dimensional Gaussian.
    
    """
    theta = degrees*np.pi/180
        
    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    
    Z = amplitude*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    
    return Z

def simulate_sinflicker_bar(pixels_across, pixels_down, 
                            viewing_distance, screen_width, 
                            thetas, sweep_steps, bar_width,
                            ecc, tr_length, flicker_hz, projector_hz):
    
    # get number of frames per volume
    frames_per_vol = tr_length*projector_hz
    total_trs = len(thetas)*sweep_steps
    total_frames = frames_per_vol * total_trs
    total_secs = total_trs*tr_length
    
    # flicker
    t = np.linspace(0,total_secs,total_secs*projector_hz)
    full_run = np.sin(2 * np.pi * flicker_hz * t)
    full_run = np.uint8((full_run + 1) * 128)
    
    # visuotopic stuff
    ppd = np.pi*pixels_across/np.arctan(screen_width/viewing_distance/2.0)/360.0 # degrees of visual angle
    deg_x, deg_y = generate_coordinate_matrices(pixels_across, pixels_down, ppd, 1.0)
    
    # initialize the bar array
    bar_stimulus = np.uint8(np.ones((pixels_down, pixels_across, total_frames)))
    
    # counter
    tr_num = 0
    
    # main loop
    for theta in thetas:
        
        if theta != -1:
            
            # convert to radians
            theta_rad = theta * np.pi / 180
            
            # get the starting point and trajectory
            start_pos = np.array([-np.cos(theta_rad)*ecc, -np.sin(theta_rad)*ecc])
            end_pos = np.array([np.cos(theta_rad)*ecc, np.sin(theta_rad)*ecc])
            run_and_rise = end_pos - start_pos;
            
            if np.mod(theta,90) == 0:
                sigma_x = bar_width/2
                sigma_y = 500
            else:
                sigma_x = 500 # pragma: no cover
                sigma_y = bar_width/2 # pragma: no cover
            
            # step through each position along the trajectory
            for step in np.arange(0,sweep_steps):
                
                # get the position of the bar at each step
                xy0 = run_and_rise * step/sweep_steps + start_pos
                
                # generate the gaussian
                Z = gaussian_2D(deg_x,deg_y,xy0[0],xy0[1],sigma_x,sigma_y,theta)
                Z_mask = np.zeros_like(Z)
                Z_mask[Z>0.33] = 1
                
                # loop over each flip
                for fr in np.arange(frames_per_vol):
                    
                    # get the frame number
                    f_num = tr_num * frames_per_vol + fr
                    
                    # set the frame to mean lum
                    bar_stimulus[:,:,f_num] = 128
                    
                    # get the bar pixels
                    xx,yy = np.nonzero(Z_mask)
                    
                    # set the amp modulation
                    bar_stimulus[xx,yy,f_num] = full_run[f_num]
                    
                # iterate TR
                tr_num += 1
                
        else:
            
            # step through each volume
            for step in np.arange(0,sweep_steps):
                                
                # step through each flip
                for f in np.arange(frames_per_vol):
                    
                    # get the frame number
                    f_num = tr_num * frames_per_vol + f
                    
                    # set the frame to mean luminance
                    bar_stimulus[:,:,f_num] = 128
                    
                # iterate
                tr_num += 1
       
    return bar_stimulus

def simulate_bar_stimulus(pixels_across, pixels_down, 
                          viewing_distance, screen_width, 
                          thetas, num_bar_steps, num_blank_steps, 
                          ecc, clip = 0.33):

    """
    A utility function for creating a sweeping bar stimulus in memory.
    
    This function creates a standard retinotopic mapping stimulus, the 
    sweeping bar. The user specifies some of the display and stimulus
    parameters. This is particularly useful for writing tests and simulating
    responses of visually driven voxels.
    
    pixels_across : int
        The number of pixels along the horizontal dimension of the display.
        
    pixels_down : int
        The number of pixels along the vertical dimension of the display.
    
    viewing_distance : float
        The distance between the participant and the display (cm).
    
    screen_width : float
        The width of the display (cm). This is used to compute the visual angle
        for determining the pixels per degree of visual angle.
    
    thetas : array-like
        An array containing the orientations of the bars that will sweep
        across the display.  For example `thetas = np.arange(0,360,360/8)`.
    
    num_steps : int
        The number of steps the bar makes on each sweep through the visual field.
    
    ecc : float
        The distance from fixation each bar sweep begins and ends (degees).
    
    blanks : bool
        A flag determining whether there are blank periods inserted at the beginning
        and the end of the stimulus run.
    
    clip : float
        The bar stimulus is created by cliping a very oblong two-dimensional
        Gaussian oriented orthogonally to the direction of the sweep.
        
    Returns
    -------
    bar : ndarray
        An array containing the bar stimulus. The array is three-dimensional, with the
        first two dimensions representing the size of the display and the third
        dimension representing time.
    
    
    """
    
    
    # visuotopic stuff
    ppd = np.pi*pixels_across/np.arctan(screen_width/viewing_distance/2.0)/360.0 # degrees of visual angle
    deg_x, deg_y = generate_coordinate_matrices(pixels_across, pixels_down, ppd, 1.0)
    
    # initialize the stimulus array
    total_trs = len(thetas[thetas==-1])*num_blank_steps + len(thetas[thetas!=-1])*num_bar_steps
    bar_stimulus = np.zeros((pixels_down, pixels_across, total_trs))
    
    # initialize a counter
    tr_num = 0
    
    # main loop
    for theta in thetas:
        
        if theta != -1:  # pragma: no cover
            
            # convert to radians
            theta_rad = theta * np.pi / 180
            
            # get the starting point and trajectory
            start_pos = np.array([-np.cos(theta_rad)*ecc, -np.sin(theta_rad)*ecc])
            end_pos = np.array([np.cos(theta_rad)*ecc, np.sin(theta_rad)*ecc])
            run_and_rise = end_pos - start_pos;
            
            if np.mod(theta,90) == 0:
                sigma_x = 1
                sigma_y = 100
            else:
                sigma_x = 100
                sigma_y = 1
            
            # step through each position along the trajectory
            for step in np.arange(0,num_bar_steps):
                
                # get the position of the bar at each step
                xy0 = run_and_rise * step/num_bar_steps + start_pos
                
                # generate the gaussian
                Z = gaussian_2D(deg_x,deg_y,xy0[0],xy0[1],sigma_x,sigma_y,theta)
                                
                # store and iterate
                bar_stimulus[:,:,tr_num] = Z
                tr_num += 1
                
        else:  # pragma: no cover
            for step in np.arange(0,num_blank_steps):
                tr_num += 1
                
    
    # digitize the bar stimulus
    bar = np.zeros_like(bar_stimulus)
    bar[bar_stimulus > clip] = 1
    bar = np.short(bar)
    
    return bar


# This should eventually be VisualStimulus, and there would be an abstract class layer
# above this called Stimulus that would be generic for n-dimentional feature spaces.
class VisualStimulus(StimulusModel):
    
    
    def __init__(self, stim_arr, viewing_distance, screen_width,
                 scale_factor, tr_length, dtype, interp='nearest'):
        
        """
        
        A child of the StimulusModel class for visual stimuli.
        
        
        Paramaters
        ----------
        
        stim_arr : ndarray
            An array containing the visual stimulus at the native resolution. The 
            visual stimulus is assumed to be three-dimensional (x,y,time).
        
        viewing_distance : float
            The distance between the participant and the display (cm).
            
        screen_width : float
            The width of the display (cm). This is used to compute the visual angle
            for determining the pixels per degree of visual angle.
        
        scale_factor : float
            The downsampling rate for ball=parking a solution. The `stim_arr` is
            downsampled so as to speed up the fitting procedure.  The final model
            estimates will be derived using the non-downsampled stimulus.
            
        """
        
        StimulusModel.__init__(self, stim_arr, dtype, tr_length)
        
        # absorb the vars
        self.viewing_distance = viewing_distance
        self.screen_width = screen_width
        self.scale_factor = scale_factor
        self.interp = interp
        
        # ascertain stimulus features
        self.pixels_across = self.stim_arr.shape[1]
        self.pixels_down = self.stim_arr.shape[0]
        self.run_length = self.stim_arr.shape[2]
        self.ppd = pixels_per_degree(self.pixels_across, self.screen_width, self.viewing_distance)
        
        # generate coordinate matrices
        deg_x, deg_y = generate_coordinate_matrices(self.pixels_across, self.pixels_down, self.ppd)
        
        # share coordinate matrices
        self.deg_x = utils.generate_shared_array(deg_x, ctypes.c_double)
        self.deg_y = utils.generate_shared_array(deg_y, ctypes.c_double)
        self.stim_arr = utils.generate_shared_array(stim_arr, dtype)
        
        if self.scale_factor == 1.0:
            
            
            self.stim_arr0 = self.stim_arr
            self.deg_x0 = self.deg_x
            self.deg_y0 = self.deg_y
            
        else:
            
            # create downsampled stimulus
            stim_arr0 = resample_stimulus(self.stim_arr, self.scale_factor)
            
            # generate the coordinate matrices
            deg_x0, deg_y0 = generate_coordinate_matrices(self.pixels_across, self.pixels_down, self.ppd, self.scale_factor)
            
            # share the arrays
            self.deg_x0 = utils.generate_shared_array(deg_x0, ctypes.c_double)
            self.deg_y0 = utils.generate_shared_array(deg_y0, ctypes.c_double)
            self.stim_arr0 = utils.generate_shared_array(stim_arr0, dtype)
        
