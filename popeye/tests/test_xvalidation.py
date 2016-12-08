import os
import multiprocessing
from itertools import repeat
import ctypes

import numpy as np
import numpy.testing as npt
import nose.tools as nt
import scipy.signal as ss


import popeye.utilities as utils
from popeye import og
import popeye.xvalidation as xval
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus, resample_stimulus
from popeye.spinach import generate_og_timeseries

# def test_coeff_of_determination():
#     
#     # make up some data and a model
#     data = np.arange(0,100)
#     model = np.arange(0,100)
#     
#     # compute cod
#     cod = xval.coeff_of_determination(data,model)
#     
#     # assert
#     npt.assert_equal(cod, 100)
    

def test_kfold_xval_repeated_runs():

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
    
    num_runs = 4
    folds = 2
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # set up bounds for the grid search
    grids = ((-10,10),(-10,10),(0.25,5.25))
    bounds = ((-12,12),(-12,12),(1/stimulus.ppd,12),(1e-8,None),(None,None))
    
    # set the grid smaples
    Ns = 5
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    model.hrf_delay = 0
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5
    baseline = -0.25
    
    # create the args context for calling the Fit class
    fit_args = [grids, bounds, [0,0,0], Ns]
    fit_kwargs = {'auto_fit': False, 'verbose' : 0}
    
    # create a series of "runs"
    data = np.zeros((num_runs,stimulus.stim_arr.shape[-1]))
    
    for r in range(num_runs):
        
        # fill out the data list
        data[r,:] = model.generate_prediction(x, y, sigma, beta, baseline)
    
    # get predictions out for each of the folds ...
    models = (model,)
    predictions = xval.kfold_xval(models, data, og.GaussianFit, folds, fit_args, fit_kwargs)
    
    # assert the coeff of determination is 100 for each prediction
    for p in predictions:
        cod = xval.coeff_of_determination(p.data,p.prediction)
        npt.assert_almost_equal(cod,100, 4)

# def test_kfold_xval_unique_runs():
#     
#     # stimulus features
#     pixels_across = 800
#     pixels_down = 600
#     viewing_distance = 38
#     screen_width = 25
#     thetas = np.arange(0,360,45)
#     num_steps = 20
#     ecc = 10
#     tr_length = 1.0
#     frames_per_tr = 1.0
#     scale_factor = 0.05
#     dtype = ctypes.c_short
#     num_runs = 4
#     folds = 2
#     
#     # create the sweeping bar stimulus in memory
#     bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
#     
#     # create an instance of the Stimulus class
#     stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, dtype)
#     
#     # set up bounds for the grid search
#     grids = ((-10,10),(-10,10),(0.25,5.25),(0.1,1e2),(-5,5))
#     bounds = ((-12,12),(-12,12),(1/stimulus.ppd,12),(0.1,1e3),(-5,5))
#     
#     # initialize the gaussian model
#     model = og.GaussianModel(stimulus)
#     
#     # generate a random pRF estimate
#     x = -5.24
#     y = 2.58
#     sigma = 1.24
#     beta = 2.5
#     hrf_delay = -0.25
#     
#     # create the args context for calling the Fit class
#     fit_args = [grids, bounds, tr_length, [0,0,0],]
#     fit_kwargs = {'auto_fit': False, 'verbose' : False}
#     
#     # create a series of "runs"
#     data = np.zeros((stimulus.stim_arr.shape[-1],num_runs))
#     
#     for r in range(num_runs):
#         
#         # fill out the data list
#         data[:,r] = og.compute_model_ts(x, y, sigma, beta, hrf_delay,
#                                         stimulus.deg_x, stimulus.deg_y, 
#                                         stimulus.stim_arr, tr_length)
#     
#     
#     # get predictions out for each of the folds ...
#     models = np.tile(model,num_runs)
#     left_out_data, predictions = xval.kfold_xval(models, data, og.GaussianFit, folds, fit_args, fit_kwargs)
#     
#     # assert the coeff of determination is 100 for each prediction
#     for k in range(folds):
#         cod = xval.coeff_of_determination(left_out_data[k], predictions[k])
#         npt.assert_almost_equal(cod,100, 4)