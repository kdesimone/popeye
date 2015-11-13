from __future__ import division

import os

import popeye.utilities as utils
import numpy as np
import numpy.testing as npt

import nose.tools as nt

from popeye.visual_stimulus import generate_coordinate_matrices, resample_stimulus


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
