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
    viewing_distance = 31
    screen_width = 41
    thetas = np.arange(0,360,90)
    # thetas = np.insert(thetas,0,-1)
    # thetas = np.append(thetas,-1)
    num_blank_steps = 0
    num_bar_steps = 30
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.50
    pixels_down = 100
    pixels_across = 100
    dtype = ctypes.c_int16
    auto_fit = True
    verbose = 0
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = dog.DifferenceOfGaussiansModel(stimulus, utils.spm_hrf)
    model.hrf_delay = 0
    model.mask_size = 20
    
    # set the pRF params
    x = 2.2
    y = 2.5
    sigma = 0.90
    sigma_ratio = 1.5
    volume_ratio = 0.5
    beta = 0.25
    baseline = -0.10
    
    # create "data"
    data = model.generate_prediction(x, y, sigma, sigma_ratio, volume_ratio, beta, baseline)
    
    # set up the grids
    x_grid = utils.grid_slice(-5,5,4)
    y_grid = utils.grid_slice(-5,5,4)
    s_grid = utils.grid_slice(1/stimulus.ppd0*1.10,3.5,4)
    sr_grid = utils.grid_slice(1.0,2.0,4)
    vr_grid = utils.grid_slice(0.10,0.90,4)
    grids = (x_grid, y_grid, s_grid, sr_grid, vr_grid,)
    
    # set up the bounds
    x_bound = (-ecc,ecc)
    y_bound = (-ecc,ecc)
    s_bound = (1/stimulus.ppd,5)
    sr_bound = (1.0,None)
    vr_bound = (1e-8,1.0)
    bounds = (x_bound, y_bound, s_bound, sr_bound, vr_bound,)
    
    # fit it
    fit = dog.DifferenceOfGaussiansFit(model, data, grids, bounds)
    
    # coarse fit
    ballpark = [1.666666666666667,
                1.666666666666667,
                2.8243187483428391,
                1.9999999999999998,
                0.10000000000000001]
                
    npt.assert_almost_equal((fit.x0,fit.y0,fit.s0,fit.sr0,fit.vr0), ballpark)
    # the baseline/beta should be 0/1 when regressed data vs. estimate
    (m,b) = np.polyfit(fit.scaled_ballpark_prediction, data, 1)
    npt.assert_almost_equal(m, 1.0)
    npt.assert_almost_equal(b, 0.0)

    # fine fit
    npt.assert_almost_equal(fit.x, x, 2)
    npt.assert_almost_equal(fit.y, y, 2)
    npt.assert_almost_equal(fit.sigma, sigma, 2)
    npt.assert_almost_equal(fit.sigma_ratio, sigma_ratio, 1)
    npt.assert_almost_equal(fit.volume_ratio, volume_ratio, 1)
    
    # test the RF
    rf = fit.model.receptive_field(*fit.estimate[0:-2])
    est = fit.estimate[0:-2].copy()
    rf_new = fit.model.receptive_field(*est)
    value_1 = np.sqrt(simps(simps(rf))) 
    value_2 = np.sqrt(simps(simps(rf_new)))
    nt.assert_almost_equal(value_1, value_2)
    
    # polar coordinates
    npt.assert_almost_equal([fit.theta,fit.rho],[np.arctan2(y,x),np.sqrt(x**2+y**2)], 4)
    
