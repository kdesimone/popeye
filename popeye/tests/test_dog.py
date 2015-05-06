import os
import multiprocessing
from itertools import repeat
import ctypes

import numpy as np
import numpy.testing as npt
import nose.tools as nt
from scipy.signal import fftconvolve

import popeye.utilities as utils
from popeye import dog, og
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus, resample_stimulus

def test_dog():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_blank_steps = 20
    num_bar_steps = 40
    ecc = 12
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.20
    resample_factor = 0.25
    pixels_across = 800 * resample_factor
    pixels_down = 600 * resample_factor
    dtype = ctypes.c_int16
    Ns = 3
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 2
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, dtype)
    
    # initialize the gaussian model
    model = dog.DifferenceOfGaussiansModel(stimulus)
    
    # set the pRF params
    x = -1.4
    y = 1.5
    sigma = 1.3
    sigma_ratio = 2.2
    volume_ratio = 0.6
    hrf_delay = -0.2
    
    # create "data"
    data = dog.compute_model_ts(x, y, sigma, sigma_ratio, volume_ratio, hrf_delay,
                                stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, tr_length)
    
    # set up the grids
    x_grid = (-ecc,ecc)
    y_grid = (-ecc,ecc)
    s_grid = (1/stimulus.ppd,5)
    sr_grid = (1.0,5.0)
    vr_grid = (0.01,0.99)
    h_grid = (-5,5)
    grids = (x_grid, y_grid, s_grid, sr_grid, vr_grid, h_grid,)
    
    # set up the bounds
    x_bound = (-ecc,ecc)
    y_bound = (-ecc,ecc)
    s_bound = (1/stimulus.ppd,5)
    sr_bound = (1.0,None)
    vr_bound = (1e-8,1.0)
    h_bound = (-5,5)
    bounds = (x_bound, y_bound, s_bound, sr_bound, vr_bound, h_bound,)
    
    # fit it
    fit = dog.DifferenceOfGaussiansFit(model, data, grids, bounds, Ns, tr_length, voxel_index, verbose)
    
    # assert equivalence
    nt.assert_almost_equal(fit.x, x, 1)
    nt.assert_almost_equal(fit.y, y, 1)
    nt.assert_almost_equal(fit.sigma, sigma, 1)
    nt.assert_almost_equal(fit.sigma_ratio, sigma_ratio, 1)
    nt.assert_almost_equal(fit.volume_ratio, volume_ratio, 1)
    nt.assert_almost_equal(fit.hrf_delay, hrf_delay, 1)