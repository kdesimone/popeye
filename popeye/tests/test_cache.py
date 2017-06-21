import os
import sharedmem
import ctypes
import pickle
import time

import numpy as np
import numpy.testing as npt
import nose.tools as nt

import popeye.utilities as utils
import popeye.og as og
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus

def test_cache_model():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,90)
    thetas = np.insert(thetas,0,-1)
    thetas = np.append(thetas,-1)
    num_blank_steps = 20
    num_bar_steps = 20
    ecc = 10
    tr_length = 1.5
    frames_per_tr = 1.0
    scale_factor = 0.50
    pixels_across = 100
    pixels_down = 100
    dtype = ctypes.c_int16
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, 
                                thetas, num_bar_steps, num_blank_steps, ecc, clip=0.01)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    model.hrf_delay = 0
    model.mask_size = 5
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5    
    baseline = -0.25
    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta, baseline)
    
    # set cache grids
    x_grid = utils.grid_slice(-10, 10, 10)
    y_grid = utils.grid_slice(-10, 10, 10)
    s_grid = utils.grid_slice(0.55,5.25, 10)
    grids = (x_grid, y_grid, s_grid,)
    
    # seed rng
    np.random.seed(4932)
    
    # cache the pRF model
    cache = model.cache_model(grids, ncpus=sharedmem.cpu_count()-1);
    
    # save it out
    now = int(time.clock()*1000)
    pickle.dump(cache, open('/tmp/og_cached_model_%d.pkl' %(now),'wb'))
    
    model.cached_model_path = '/tmp/og_cached_model_%d.pkl' %(now)
    
    # set search bounds
    x_bounds = (-15.0,15.0)
    y_bounds = (-15.0,15.0)
    s_bounds = (1/model.stimulus.ppd, 10.0)
    b_bounds = (1e-8,10)
    m_bounds = (-5,5)
    bounds = (x_bounds, y_bounds, s_bounds, b_bounds, m_bounds)
    
    # fitting params
    auto_fit = True
    verbose = 1
    
    # fit the model
    fit = og.GaussianFit(model, data, grids, bounds, verbose=verbose, auto_fit=False)
    
    
    

def test_cache_model():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,90)
    thetas = np.insert(thetas,0,-1)
    thetas = np.append(thetas,-1)
    num_blank_steps = 20
    num_bar_steps = 20
    ecc = 10
    tr_length = 1.5
    frames_per_tr = 1.0
    scale_factor = 0.50
    pixels_across = 100
    pixels_down = 100
    dtype = ctypes.c_int16
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, 
                                thetas, num_bar_steps, num_blank_steps, ecc, clip=0.01)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    model.hrf_delay = 0
    model.mask_size = 5
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5    
    baseline = -0.25
    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta, baseline)
    
    # set cache grids
    x_grid = utils.grid_slice(-10, 10, 10)
    y_grid = utils.grid_slice(-10, 10, 10)
    s_grid = utils.grid_slice(0.55,5.25, 10)
    grids = (x_grid, y_grid, s_grid,)
    
    # seed rng
    np.random.seed(4932)
    
    # cache the pRF model
    cache = model.cache_model(grids, ncpus=sharedmem.cpu_count()-1)
    
    # save it out
    now = int(time.clock()*1000)
    pickle.dump(cache, open('/tmp/og_cached_model_%d.pkl' %(now),'wb'),protocol=3)
    
    # re-read it in 
    cached_model = pickle.load(open('/tmp/og_cached_model_%d.pkl' %(now),'rb'))
    
    # timeseries
    orig_cached_timeseries = np.array([c[0] for c in cache])
    cached_timeseries = np.array([c[0] for c in cached_model])
    
    # parameters
    orig_cached_parameters = np.array([c[0] for c in cache])
    cached_parameters = np.array([c[0] for c in cached_model])
    
    # make sure the same
    nt.assert_true(np.sum(orig_cached_parameters - cached_parameters) == 0)
    nt.assert_true(np.sum(orig_cached_parameters - cached_parameters) == 0)
    