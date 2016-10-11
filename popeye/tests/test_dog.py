import os, ctypes

import numpy as np
import numpy.testing as npt
import nose.tools as nt
from scipy.integrate import simps

import popeye.utilities as utils
from popeye import dog
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus

def test_dog():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,90)
    num_blank_steps = 0
    num_bar_steps = 30
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.50
    pixels_down = 50
    pixels_across = 50
    dtype = ctypes.c_int16
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = dog.DifferenceOfGaussiansModel(stimulus, utils.spm_hrf)
    model.hrf_delay = 0.2
    
    # set the pRF params
    x = -1.4
    y = 1.5
    sigma = 1.0
    sigma_ratio = 2.0
    volume_ratio = 0.5
    hrf_delay = -0.2
    
    # create "data"
    data = model.generate_prediction(x, y, sigma, sigma_ratio, volume_ratio,)
    
    # set up the grids
    x_grid = slice(-10,10,3)
    y_grid = slice(-10,10,3)
    s_grid = slice(1/stimulus.ppd,5,3)
    sr_grid = slice(1.0,5.0,3)
    vr_grid = slice(0.01,0.99,3)
    grids = (x_grid, y_grid, s_grid, sr_grid, vr_grid,)
    
    # set up the bounds
    x_bound = (-ecc,ecc)
    y_bound = (-ecc,ecc)
    s_bound = (1/stimulus.ppd,5)
    sr_bound = (1.0,None)
    vr_bound = (1e-8,1.0)
    bounds = (x_bound, y_bound, s_bound, sr_bound, vr_bound,)
    
    # fit it
    fit = dog.DifferenceOfGaussiansFit(model, data, grids, bounds, voxel_index)
    
    # coarse fit
    nt.assert_almost_equal((fit.x0,fit.y0,fit.s0,fit.sr0,fit.vr0),(-1.0, 2.0, 0.72833937882323319, 1.0, 0.01))
    
    # fine fit
    nt.assert_almost_equal(fit.x, x)
    nt.assert_almost_equal(fit.y, y)
    nt.assert_almost_equal(fit.sigma, sigma)
    nt.assert_almost_equal(fit.sigma_ratio, sigma_ratio)
    nt.assert_almost_equal(fit.volume_ratio, volume_ratio)
    
    # test the RF
    rf = fit.model.receptive_field(*fit.estimate)
    est = fit.estimate.copy()
    est[2] *= 2
    rf_new = fit.model.receptive_field(*est)
    value_1 = np.sqrt(simps(simps(rf))) 
    value_2 = np.sqrt(simps(simps(rf_new)))
    nt.assert_almost_equal(value_2/value_1,sigma_ratio,1)
    
    # polar coordinates
    npt.assert_almost_equal([fit.theta,fit.rho],[np.arctan2(y,x),np.sqrt(x**2+y**2)])
    