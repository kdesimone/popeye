import os
from itertools import repeat
import ctypes

import numpy as np
import numpy.testing as npt
import nose.tools as nt
from scipy.signal import fftconvolve

import popeye.utilities as utils
import popeye.ogb as ogb
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus

def test_ogb_fit():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_blank_steps = 20
    num_bar_steps = 40
    ecc = 12
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.40
    resample_factor = 0.125
    pixels_across = 800 * resample_factor
    pixels_down = 600 * resample_factor
    dtype = ctypes.c_int16
    Ns = 4
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    
    # insert blanks
    thetas = list(thetas)
    thetas.insert(0,-1)
    thetas.insert(2,-1)
    thetas.insert(5,-1)
    thetas.insert(8,-1)
    thetas.insert(11,-1)
    thetas.append(-1)
    thetas = np.array(thetas)
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = ogb.GaussianModel(stimulus, utils.double_gamma_hrf)
    
    # generate a random pRF estimate
    x = -2.24
    y = 2.58
    sigma = 0.58
    beta = 1.1
    baseline = 0.5
    hrf_delay = -0.25
    
    # create data
    data = model.generate_prediction(x, y, sigma, beta, baseline, hrf_delay)
    
    # set search grid
    x_grid = (-10,10)
    y_grid = (-10,10)
    s_grid = (1/stimulus.ppd,5.25)
    b_grid = (0.1,1.0)
    bl_grid = (-1,1)
    h_grid = (-4.0,4.0)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (1/stimulus.ppd,12.0)
    b_bound = (1e-8,1e2)
    bl_bound = (-1,1)
    h_bound = (-5.0,5.0)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, b_grid, bl_grid, h_grid)
    bounds = (x_bound, y_bound, s_bound, b_bound, bl_bound, h_bound)
    
    # fit the response
    fit = ogb.GaussianFit(model, data, grids, bounds, Ns, voxel_index, auto_fit, verbose)
    
    # assert equivalence
    nt.assert_almost_equal(fit.x, x, 1)
    nt.assert_almost_equal(fit.y, y, 1)
    nt.assert_almost_equal(fit.sigma, sigma, 1)
    nt.assert_almost_equal(fit.beta, beta, 1)
    nt.assert_almost_equal(fit.hrf_delay, hrf_delay, 1)