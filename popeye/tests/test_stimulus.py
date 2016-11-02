from __future__ import division

import os, ctypes

import popeye.utilities as utils
import numpy as np
import numpy.testing as npt

import nose.tools as nt

from popeye.visual_stimulus import pixels_per_degree, generate_coordinate_matrices, resample_stimulus, simulate_sinflicker_bar, simulate_bar_stimulus, VisualStimulus

def test_pixels_per_degree():
    
    pixels_across = 1
    screen_width = 1
    viewing_distance = 1
    ppd = pixels_per_degree(pixels_across, screen_width, viewing_distance)
    
def test_generate_coordinate_matrices():
    
    # set some dummy display parameters
    pixels_across = 100
    pixels_down = 100
    ppd = 1.0
    scale_factor = 1.0
    
    # generate coordinates
    deg_x, deg_y = generate_coordinate_matrices(pixels_across,pixels_down,ppd,scale_factor)
    
    # assert
    nt.assert_true(np.sum(deg_x[0,0:50]) == np.sum(deg_x[0,50::])*-1)
    nt.assert_true(np.sum(deg_y[0:50,0]) == np.sum(deg_y[50::,0])*-1)
    
    # try the same with an odd number of pixels
    pixels_across = 101
    pixels_down = 101
    
    # generate coordinates
    deg_x, deg_y = generate_coordinate_matrices(pixels_across,pixels_down,ppd,scale_factor)
    
    # assert
    nt.assert_true(np.sum(deg_x[0,0:50]) == np.sum(deg_x[0,50::])*-1)
    nt.assert_true(np.sum(deg_y[0:50,0]) == np.sum(deg_y[50::,0])*-1)
    
    # try with another rescaling factor
    scale_factor = 0.5
    
    # get the horizontal and vertical coordinate matrices
    deg_x, deg_y = generate_coordinate_matrices(pixels_across,pixels_down,ppd,scale_factor)
    
    # assert
    nt.assert_true(np.sum(deg_x[0,0:50]) == np.sum(deg_x[0,50::])*-1)
    nt.assert_true(np.sum(deg_y[0:50,0]) == np.sum(deg_y[50::,0])*-1)

def test_noresample_stimulus():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,90)
    num_blank_steps = 0
    num_bar_steps = 30
    ecc = 12
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 1.0
    pixels_across = 100
    pixels_down = 100
    dtype = ctypes.c_int16
    Ns = 3
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # stimulus
    npt.assert_equal(stimulus.stim_arr.shape[0:2],stimulus.stim_arr0.shape[0:2])
    npt.assert_equal(stimulus.deg_x.shape,stimulus.deg_x0.shape)
    npt.assert_equal(stimulus.deg_y.shape,stimulus.deg_y0.shape)

def test_resample_stimulus():
    
    # set the downsampling rate
    scale_factor = 0.5
    
    # create a stimulus
    stimulus = np.random.random((100,100,20))
    
    # downsample the stimulus by 50%
    stimulus_coarse = resample_stimulus(stimulus,scale_factor)
    
    # grab the stimulus dimensions
    stim_dims = np.shape(stimulus)
    stim_coarse_dims = np.shape(stimulus_coarse)
    
    # assert
    nt.assert_true(stim_coarse_dims[0]/stim_dims[0] == scale_factor)
    nt.assert_true(stim_coarse_dims[1]/stim_dims[1] == scale_factor)
    nt.assert_true(stim_coarse_dims[2] == stim_dims[2])

def test_simulate_sinflicker_bar():
    
    # no blanks
    bar = simulate_sinflicker_bar(100,100,50,10,[0],1,10,5,1,1,60)
    y = utils.normalize(bar[1,1,:],0,1)
    t = np.linspace(0,1,60)
    yhat = utils.normalize(np.sin(2 * np.pi * t),0,1)
    nt.assert_almost_equal(np.sum(yhat-y),0)
    
    # blanks
    bar = simulate_sinflicker_bar(100,100,50,10,[0,-1],1,10,5,1,1,60)
    y = np.round(utils.normalize(bar[1,1,:],0,1),2)
    t = np.linspace(0,1,60)
    yhat = utils.normalize(np.sin(2 * np.pi * t),0,1)
    yhat = np.append(yhat,np.repeat(0.5,60))
    nt.assert_almost_equal(np.sum(yhat-np.round(y,2)),0,1)
    