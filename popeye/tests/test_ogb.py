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
from popeye.spinach import generate_og_receptive_field
def test_ogb_fit():
    
    # stimulus features
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
    Ns = 3
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = ogb.GaussianModel(stimulus, utils.double_gamma_hrf)
    model.tr_length = 1.0
    
    # generate a random pRF estimate
    x = -2.2
    y = 2.6
    sigma = 0.8
    beta = 85
    baseline = 0.5
    hrf_delay = -0.3
    
    # create data
    data = model.generate_prediction(x, y, sigma, beta, baseline, hrf_delay)
    
    # set search grid
    x_grid = (-5,5)
    y_grid = (-5,5)
    s_grid = (1/stimulus.ppd,5.25)
    b_grid = (1,100)
    bl_grid = (-1,1)
    h_grid = (-4.0,4.0)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (1/stimulus.ppd,12.0)
    b_bound = (1e-8,None)
    bl_bound = (-1,1)
    h_bound = (-5.0,5.0)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, b_grid, bl_grid, h_grid)
    bounds = (x_bound, y_bound, s_bound, b_bound, bl_bound, h_bound)
    
    # fit the response
    fit = ogb.GaussianFit(model, data, grids, bounds, Ns=Ns)
    
    # test ballpark
    nt.assert_almost_equal(fit.x0, 0, 1)
    nt.assert_almost_equal(fit.y0, 0, 1)
    nt.assert_almost_equal(fit.s0, 2.8, 1)
    nt.assert_almost_equal(fit.beta0, 100, 1)
    nt.assert_almost_equal(fit.baseline0, -1, 1)
    nt.assert_almost_equal(fit.hrf0, 0, 1)
    
    # test final estimate
    nt.assert_almost_equal(fit.x, x, 1)
    nt.assert_almost_equal(fit.y, y, 1)
    nt.assert_almost_equal(fit.sigma, sigma, 1)
    nt.assert_almost_equal(np.round(fit.beta), beta, 1)
    nt.assert_almost_equal(fit.hrf_delay, hrf_delay, 1)
    
    # test receptive field
    rf = generate_og_receptive_field(x, y, sigma, fit.model.stimulus.deg_x, fit.model.stimulus.deg_y)
    nt.assert_almost_equal(rf.sum(), fit.receptive_field.sum())
    
    # test HRF
    nt.assert_almost_equal(fit.hemodynamic_response.sum(), fit.model.hrf_model(fit.hrf_delay, fit.model.stimulus.tr_length).sum())
    