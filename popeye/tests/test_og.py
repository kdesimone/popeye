import os
import sharedmem
import ctypes

import numpy as np
import numpy.testing as npt
import nose.tools as nt
from scipy.signal import fftconvolve
import statsmodels.api as sm

import popeye.utilities as utils
import popeye.og as og
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus, resample_stimulus
from popeye.spinach import generate_og_receptive_field

def test_og_fit():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,90)
    num_blank_steps = 0
    num_bar_steps = 30
    ecc = 12
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.10
    pixels_across = 100
    pixels_down = 100
    dtype = ctypes.c_int16
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    model.hrf_delay = 0
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta)
    
    # set search grid
    x_grid = (-10,10)
    y_grid = (-10,10)
    s_grid = (0.25,5.25)
    b_grid = (0.1,1.0)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (0.001,12.0)
    b_bound = (1e-8,1e2)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, b_grid,)
    bounds = (x_bound, y_bound, s_bound, b_bound,)
    
    # fit the response
    fit = og.GaussianFit(model, data, grids, bounds, Ns=3)
    
    # coarse fit
    nt.assert_almost_equal(fit.x0,-10.0)
    nt.assert_almost_equal(fit.y0,0.0)
    nt.assert_almost_equal(fit.s0, 5.25)
    nt.assert_almost_equal(fit.beta0, 1.0)
    
    # assert equivalence
    nt.assert_almost_equal(fit.x, x, 2)
    nt.assert_almost_equal(fit.y, y, 2)
    nt.assert_almost_equal(fit.sigma, sigma, 2)
    nt.assert_almost_equal(fit.beta, beta, 2)
    
    # test receptive field
    rf = generate_og_receptive_field(x, y, sigma, fit.model.stimulus.deg_x, fit.model.stimulus.deg_y)
    nt.assert_almost_equal(rf.sum(), fit.receptive_field.sum()) 
    
    # test model == fit RF
    nt.assert_almost_equal(fit.model.generate_receptive_field(x,y,sigma).sum(), fit.receptive_field.sum()) 

def test_og_nuisance_fit():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,90)
    num_blank_steps = 0
    num_bar_steps = 10
    ecc = 12
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.10
    pixels_across = 100
    pixels_down = 100
    dtype = ctypes.c_int16
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.spm_hrf)
    model.hrf_delay = 0
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5
    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta)
    
    # create nuisance signal
    step = np.zeros(len(data))
    step[30:-30] = 1
    
    # add to data
    data += step
    
    # create design matrix
    nuisance = sm.add_constant(step)
    
    # recreate model with nuisance
    model = og.GaussianModel(stimulus, utils.spm_hrf, nuisance)
    model.hrf_delay = 0
    
    # set search grid
    x_grid = (-10,10)
    y_grid = (-10,10)
    s_grid = (0.25,5.25)
    b_grid = (0.1,1.0)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (0.001,12.0)
    b_bound = (1e-8,1e2)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, b_grid,)
    bounds = (x_bound, y_bound, s_bound, b_bound,)
    
    # fit the response
    fit = og.GaussianFit(model, data, grids, bounds, Ns=3)
    
    # assert equivalence
    nt.assert_almost_equal(fit.x, x, 2)
    nt.assert_almost_equal(fit.y, y, 2)
    nt.assert_almost_equal(fit.sigma, sigma, 2)
    nt.assert_almost_equal(fit.beta, beta, 2)
    