import os
from itertools import repeat
import ctypes

import numpy as np
import numpy.testing as npt
import nose.tools as nt
from scipy.signal import fftconvolve

import popeye.utilities as utils
import popeye.ogb_nohrf as ogb
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus

def test_ogb_manual_grids():
    
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
    pixels_down = 50
    pixels_across = 50
    dtype = ctypes.c_int16
    Ns = None
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
    model = ogb.GaussianModel(stimulus, utils.spm_hrf)
    model.tr_length = 1.0
    model.hrf_delay = 0.0
    
    # generate a random pRF estimate
    x = -2.2
    y = 2.6
    sigma = 0.8
    beta = 85
    baseline = 0.5
    
    # create data
    data = model.generate_prediction(x, y, sigma, beta, baseline,)
    
    # set search grid
    x_grid = slice(-5,5,3)
    y_grid = slice(-5,5,3)
    s_grid = slice(1/stimulus.ppd,5.25,5)
    b_grid = slice(1,100,2)
    bl_grid = slice(-1,1,3)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (1/stimulus.ppd,12.0)
    b_bound = (1e-8,None)
    bl_bound = (-1,1)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, b_grid, bl_grid,) 
    bounds = (x_bound, y_bound, s_bound, b_bound, bl_bound,)
    
    # fit the response
    fit = ogb.GaussianFit(model, data, grids, bounds, Ns=Ns)
    
    # assert equivalence
    nt.assert_almost_equal(fit.x, x, 1)
    nt.assert_almost_equal(fit.y, y, 1)
    nt.assert_almost_equal(fit.sigma, sigma, 1)
    nt.assert_almost_equal(np.round(fit.beta), beta, 1)
    
    npt.assert_almost_equal(fit.overloaded_estimate,[2.2730532583028005,
                                                     3.4058772731826976,
                                                     0.79999999999807414,
                                                     84.999999999997371,
                                                     0.50000000000040556])


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
    pixels_down = 50
    pixels_across = 50
    dtype = ctypes.c_int16
    Ns = 3
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
    model = ogb.GaussianModel(stimulus, utils.spm_hrf)
    model.tr_length = 1.0
    model.hrf_delay = 0.0
    
    # generate a random pRF estimate
    x = -2.2
    y = 2.6
    sigma = 0.8
    beta = 85
    baseline = 0.5
    
    # create data
    data = model.generate_prediction(x, y, sigma, beta, baseline,)
    
    # set search grid
    x_grid = (-5,5)
    y_grid = (-5,5)
    s_grid = (1/stimulus.ppd,5.25)
    b_grid = (1,100)
    bl_grid = (-1,1)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (1/stimulus.ppd,12.0)
    b_bound = (1e-8,None)
    bl_bound = (-1,1)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, b_grid, bl_grid,)
    bounds = (x_bound, y_bound, s_bound, b_bound, bl_bound,)
    
    # fit the response
    fit = ogb.GaussianFit(model, data, grids, bounds, Ns=Ns)
    
    # coarse
    npt.assert_almost_equal((fit.x0,fit.y0,fit.s0,fit.beta0,fit.baseline0),(0.0, 0.0, 2.9891696894116166, 50.5, 1.0))
    
    # assert equivalence
    nt.assert_almost_equal(fit.x, x, 1)
    nt.assert_almost_equal(fit.y, y, 1)
    nt.assert_almost_equal(fit.sigma, sigma, 1)
    nt.assert_almost_equal(fit.beta, beta, 1)
    