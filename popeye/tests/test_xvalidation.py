import os
import multiprocessing
from itertools import repeat

import numpy as np
import numpy.testing as npt
import nose.tools as nt
import scipy.signal as ss


import popeye.utilities as utils
import popeye.gaussian as gaussian
import popeye.xvalidation as xval
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus
from popeye.spinach import MakeFastGaussPrediction

def test_coeff_of_determination():
    
    # make up some data and a model
    data = np.arange(0,100)
    model = np.arange(0,100)
    
    # compute cod
    cod = xval.coeff_of_determination(data,model)
    
    # assert
    npt.assert_equal(cod, 100)
    

def test_kfold_xval_repeated_runs():

    # stimulus features
    pixels_across = 800
    pixels_down = 600
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_steps = 20
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.05
    num_runs = 4
    folds = 2

    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)

    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, frames_per_tr)

    # set up bounds for the grid search
    search_bounds = ((-10,10),(-10,10),(0.25,5.25),(-5,5),(0.1,1e2))
    fit_bounds = ((-12,12),(-12,12),(1/(stimulus.ppd*stimulus.scale_factor),12),(-5,5),(0.1,1e2))

    # initialize the gaussian model
    model = gaussian.GaussianModel(stimulus)

    # generate a pRF estimate
    estimate = [-5.24, 2.583, 1.24, -0.25, 2.5]

    # create the args context for calling the Fit class
    fit_args = [search_bounds, fit_bounds, tr_length, [0,0,0]]
    fit_kwargs = {'auto_fit': False, 'verbose' : False}    

    # create a series of "runs"
    data = np.zeros((stimulus.stim_arr.shape[-1],num_runs))

    for r in range(num_runs):

        # make the stim time-series
        stim = MakeFastGaussPrediction(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, estimate[0], estimate[1], estimate[2])

        # create the HRF
        hrf = utils.double_gamma_hrf(estimate[3], tr_length, frames_per_tr)

        # simulate the BOLD response
        dat = ss.fftconvolve(stim,hrf)[0:len(stim)] * estimate[-1]

        # fill out the data list
        data[:,r] = dat


    # get predictions out for each of the folds ...
    models = (model,)
    left_out_data, predictions = xval.kfold_xval(models, data, gaussian.GaussianFit, folds, fit_args, fit_kwargs)

    # assert the coeff of determination is 100 for each prediction
    for k in range(folds):
        cod = xval.coeff_of_determination(left_out_data[k], predictions[k])
        npt.assert_almost_equal(cod,100, 4)

def test_kfold_xval_unique_runs():
    
    # stimulus features
    pixels_across = 800
    pixels_down = 600
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_steps = 20
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.05
    num_runs = 4
    folds = 2
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, frames_per_tr)
    
    # set up bounds for the grid search
    search_bounds = ((-10,10),(-10,10),(0.25,5.25),(-5,5),(0.1,1e2))
    fit_bounds = ((-12,12),(-12,12),(1/(stimulus.ppd*stimulus.scale_factor),12),(-5,5),(0.1,1e2))
    
    # initialize the gaussian model
    model = gaussian.GaussianModel(stimulus)
    
    # generate a pRF estimate
    estimate = [-5.24, 2.583, 1.24, -0.25, 2.5]
    
    # create the args context for calling the Fit class
    fit_args = [search_bounds, fit_bounds, tr_length, [0,0,0],]
    fit_kwargs = {'auto_fit': False, 'verbose' : False}
    
    # create a series of "runs"
    data = np.zeros((stimulus.stim_arr.shape[-1],num_runs))
    
    for r in range(num_runs):
        
        # make the stim time-series
        stim = MakeFastGaussPrediction(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, estimate[0], estimate[1], estimate[2])
        
        # create the HRF
        hrf = utils.double_gamma_hrf(estimate[3], tr_length, frames_per_tr)
        
        # simulate the BOLD response
        dat = ss.fftconvolve(stim,hrf)[0:len(stim)] * estimate[-1]
        
        # fill out the data list
        data[:,r] = dat
    
    
    # get predictions out for each of the folds ...
    models = np.tile(model,num_runs)
    left_out_data, predictions = xval.kfold_xval(models, data, gaussian.GaussianFit, folds, fit_args, fit_kwargs)
    
    # assert the coeff of determination is 100 for each prediction
    for k in range(folds):
        cod = xval.coeff_of_determination(left_out_data[k], predictions[k])
        npt.assert_almost_equal(cod,100, 4)