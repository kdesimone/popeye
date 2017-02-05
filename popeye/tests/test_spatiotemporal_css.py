import os
from itertools import repeat
import ctypes

import numpy as np
import numpy.testing as npt
import nose.tools as nt

import popeye.utilities as utils
from popeye import spatiotemporal_css as strf
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus, resample_stimulus

def test_strf_fit():
    
    viewing_distance = 38
    screen_width = 25
    thetas = np.tile(np.arange(0,360,90),2)
    thetas = np.insert(thetas,0,-1)
    thetas = np.append(thetas,-1)
    num_blank_steps = 30
    num_bar_steps = 30
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.10
    pixels_down = 500
    pixels_across = 500
    dtype = ctypes.c_int16
    Ns = 3
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    projector_hz = 480
    tau = 0.0875
    mask_size = 5
    hrf = 0.25
    
    # create the sweeping bar stimulus in memory
    stim = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                 screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(stim, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    stimulus.fps = projector_hz
    flicker_vec = np.zeros_like(stim[0,0,:]).astype('uint8')
    flicker_vec[1*20:5*20] = 1
    flicker_vec[5*20:9*20] = 2
    stimulus.flicker_vec = flicker_vec
    stimulus.flicker_hz = [10,20]
    
    # initialize the gaussian model
    model = strf.SpatioTemporalModel(stimulus, utils.double_gamma_hrf)
    model.tau = tau
    model.hrf_delay = hrf
    model.mask_size = mask_size
    
    # generate a random pRF estimate
    x = -2.1
    y = 1.5
    sigma = 1.2
    n = 0.75
    weight = 0.90
    beta = 1.0
    baseline = 0
    
    # create the "data"
    data =  model.generate_prediction(x, y, sigma, n, weight, beta, baseline)
    
    # set search grid
    x_grid = (-3,2)
    y_grid = (-3,2)
    s_grid = (1/stimulus.ppd,2.75)
    n_grid = (0.1,0.90)
    w_grid = (0.1,0.90)
    
    # set search bounds
    x_bound = (-10,10)
    y_bound =  (-10,10)
    s_bound = (1/stimulus.ppd,10)
    w_bound = (1e-8,1.0)
    n_bound = (1e-8,1.0)
    b_bound = (1e-8,1e5)
    u_bound = (None, None)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, n_grid, w_grid,)
    bounds = (x_bound, y_bound, s_bound, n_bound, w_bound, b_bound, u_bound)
    
    # fit the response
    fit = strf.SpatioTemporalFit(model, data, grids, bounds, Ns=Ns)
    
    # coarse fit
    npt.assert_almost_equal((fit.x0,fit.y0,fit.sigma0,fit.weight0,fit.beta0,fit.baseline0),[-3.       ,  2.       ,  1.5570848,  0.9      ,  1.,-0.0052418])
    
    # fine fit
    npt.assert_almost_equal(fit.x, x, 1)
    npt.assert_almost_equal(fit.y, y, 1)
    npt.assert_almost_equal(fit.sigma, sigma, 1)
    npt.assert_almost_equal(fit.weight, weight, 1)
    npt.assert_almost_equal(fit.beta, beta, 1)
    npt.assert_almost_equal(fit.baseline, baseline, 1)
    
    # overloaded
    npt.assert_almost_equal(fit.overloaded_estimate, [2.5272803327894366,
                                                     2.7411676344272533,
                                                     1.2300000000000411,
                                                     0.89999999999142188,
                                                     1.0000000000029761,
                                                     -0.24999999999909184])
