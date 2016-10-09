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
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5
    hrf_delay = -0.25
    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta, hrf_delay)
    
    # set search grid
    x_grid = (-10,10)
    y_grid = (-10,10)
    s_grid = (0.25,5.25)
    b_grid = (0.1,1.0)
    h_grid = (-4.0,4.0)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (0.001,12.0)
    b_bound = (1e-8,1e2)
    h_bound = (-5.0,5.0)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, b_grid, h_grid)
    bounds = (x_bound, y_bound, s_bound, b_bound, h_bound)
    
    # fit the response
    fit = og.GaussianFit(model, data, grids, bounds, Ns, voxel_index, auto_fit, verbose)
    
    # assert equivalence
    nt.assert_almost_equal(fit.x, x, 2)
    nt.assert_almost_equal(fit.y, y, 2)
    nt.assert_almost_equal(fit.sigma, sigma, 2)
    nt.assert_almost_equal(fit.beta, beta, 2)
    nt.assert_almost_equal(fit.hrf_delay, hrf_delay, 2)


def test_og_nuissance_fit():
    
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
    model = og.GaussianModel(stimulus, utils.spm_hrf)
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5
    hrf_delay = -0.25
    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta, hrf_delay)
    
    # create nuissance signal
    step = np.zeros(len(data))
    step[30:-30] = 1
    
    # add to data
    data += step
    
    # create design matrix
    nuissance = sm.add_constant(step)
    
    # recreate model with nuissance
    model = og.GaussianModel(stimulus, utils.spm_hrf, nuissance)
    
    
    # set search grid
    x_grid = (-10,10)
    y_grid = (-10,10)
    s_grid = (0.25,5.25)
    b_grid = (0.1,1.0)
    h_grid = (-4.0,4.0)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (0.001,12.0)
    b_bound = (1e-8,1e2)
    h_bound = (-5.0,5.0)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, b_grid, h_grid)
    bounds = (x_bound, y_bound, s_bound, b_bound, h_bound)
    
    # fit the response
    fit = og.GaussianFit(model, data, grids, bounds, Ns, voxel_index, auto_fit, verbose)
    
    # assert equivalence
    nt.assert_almost_equal(fit.x, x, 2)
    nt.assert_almost_equal(fit.y, y, 2)
    nt.assert_almost_equal(fit.sigma, sigma, 2)
    nt.assert_almost_equal(fit.beta, beta, 2)
    nt.assert_almost_equal(fit.hrf_delay, hrf_delay, 2)

def test_parallel_og_fit():

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
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5
    hrf_delay = -0.25
    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta, hrf_delay)
    
    # set search grid
    x_grid = (-8,8)
    y_grid = (-8,8)
    s_grid = (stimulus.ppd,5.25)
    b_grid = (0.1,1.0)
    h_grid = (-4.0,4.0)
    
    # set search bounds
    x_bound = (-ecc,ecc)
    y_bound = (-ecc,ecc)
    s_bound = (1/stimulus.ppd,ecc)
    b_bound = (1e-8,1e2)
    h_bound = (-5.0,5.0)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, b_grid, h_grid)
    bounds = (x_bound, y_bound, s_bound, b_bound, h_bound)
    
    # multiprocess 3 voxels
    data = [data,data,data]
    indices = ([1,2,3],[4,6,5],[7,8,9])
    bundle = utils.multiprocess_bundle(og.GaussianFit, model, data, grids, bounds, 3, indices, True, 1)
    
    # run analysis
    with sharedmem.MapReduce(np=sharedmem.cpu_count()-1) as pool:
        output = pool.map(utils.parallel_fit, bundle)
    
    # assert equivalence
    for fit in output:
        nt.assert_almost_equal(fit.x, x)
        nt.assert_almost_equal(fit.y, y)
        nt.assert_almost_equal(fit.sigma, sigma)
        nt.assert_almost_equal(fit.beta, beta)
        nt.assert_almost_equal(fit.hrf_delay, hrf_delay)