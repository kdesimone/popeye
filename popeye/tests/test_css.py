import os
from itertools import repeat
import ctypes

import numpy as np
import numpy.testing as npt
import nose.tools as nt

import popeye.utilities as utils
import popeye.css_nohrf as css
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus, resample_stimulus

def test_css_fit():
    
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,90)
    num_blank_steps = 0
    num_bar_steps = 30
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.10
    pixels_down = 100
    pixels_across = 100
    dtype = ctypes.c_int16
    Ns = 5
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = css.CompressiveSpatialSummationModel(stimulus, utils.double_gamma_hrf)
    model.hrf_delay = 0.2
    
    # generate a random pRF estimate
    x = -2.24
    y = 1.58
    sigma = 1.23
    n = 0.90
    beta = 1.0
    baseline = 0.0
    
    # create the "data"
    data =  model.generate_prediction(x, y, sigma, n, beta, baseline)
    
    # set search grid
    x_grid = (-3,2)
    y_grid = (-3,2)
    s_grid = (1/stimulus.ppd,2.75)
    n_grid = (0.1,0.90)
    b_grid = (0.01,1)
    bl_grid = (-2,2)
    h_grid = (-4.0,4.0)
    
    # set search bounds
    x_bound = (-10,10)
    y_bound =  (-10,10)
    s_bound = (1/stimulus.ppd,10)
    n_bound = (1e-8,1.0)
    b_bound = (1e-8,1e5)
    bl_bound = (None,None)
    h_bound = (-5.0,5.0)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, n_grid, b_grid, bl_grid,)
    bounds = (x_bound, y_bound, s_bound, n_bound, b_bound, bl_bound,)
    
    # fit the response
    fit = css.CompressiveSpatialSummationFit(model, data, grids, bounds, Ns, voxel_index, auto_fit, 2)
    
    # assert equivalence
    nt.assert_almost_equal(fit.x, x, 1)
    nt.assert_almost_equal(fit.y, y, 1)
    nt.assert_almost_equal(fit.sigma, sigma, 1)
    nt.assert_almost_equal(fit.n, n, 1)
    nt.assert_almost_equal(fit.beta, beta, 1)
    nt.assert_almost_equal(fit.baseline, baseline, 1)
