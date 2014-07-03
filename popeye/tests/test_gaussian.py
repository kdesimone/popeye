import os
import multiprocessing
from itertools import repeat

import numpy as np
import numpy.testing as npt
import nose.tools as nt
import scipy.signal as ss


import popeye.utilities as utils
import popeye.gaussian as gaussian
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus
from popeye.spinach import MakeFastGaussPrediction

def test_gaussian_fit():
    
    # stimulus features
    pixels_across = 500 
    pixels_down = 500
    viewing_distance = 20
    screen_width = 10
    thetas = np.arange(0,360,45)
    num_steps = 10
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.05
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, frames_per_tr)
    
    # set up bounds for the grid search
    search_bounds = ((-10,10),(-10,10),(0.25,5.25),(-5,5))
    fit_bounds = ((-12,12),(-12,12),(1/(stimulus.ppd*stimulus.scale_factor),12),(-5,5))
    
    # initialize the gaussian model
    gaussian_model = gaussian.GaussianModel(stimulus)
    
    # generate a random pRF estimate
    estimate = [1,1,1,1]
    
    # generate the modeled BOLD response`
    stim = MakeFastGaussPrediction(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, estimate[0], estimate[1], estimate[2])
    hrf = utils.double_gamma_hrf(estimate[3], tr_length, frames_per_tr)
    response = utils.zscore(ss.fftconvolve(stim,hrf)[0:len(stim)])
    
    # fit the response
    gaussian_fit = gaussian.GaussianFit(response, gaussian_model, search_bounds, fit_bounds, tr_length, [0,0,0], 0, False)
    
    # assert equivalence
    nt.assert_almost_equal(gaussian_fit.x,estimate[0])
    nt.assert_almost_equal(gaussian_fit.y,estimate[1])
    nt.assert_almost_equal(gaussian_fit.sigma,estimate[2])
    nt.assert_almost_equal(gaussian_fit.hrf_delay,estimate[3])
    

def test_parallel_gaussian_fit():

    # stimulus features
    pixels_across = 500 
    pixels_down = 500
    viewing_distance = 20
    screen_width = 10
    thetas = np.arange(0,360,45)
    num_steps = 10
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.05
    num_voxels = multiprocessing.cpu_count()-1
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, frames_per_tr)
    
    # set up bounds for the grid search
    search_bounds = [((-10,10),(-10,10),(0.25,5.25),(-5,5))]*num_voxels
    fit_bounds = [((-12,12),(-12,12),(stimulus.ppd*stimulus.scale_factor,12),(-5,5))]*num_voxels
    
    # make fake voxel indices
    indices = [(1,2,3)]*num_voxels
    
    # initialize the gaussian model
    gaussian_model = gaussian.GaussianModel(stimulus)
    
    # generate a random pRF estimate
    estimates = [(1,1,1,1)]*num_voxels
    
    # create the simulated time-series
    timeseries = []
    for estimate in estimates:
        stim = MakeFastGaussPrediction(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, estimate[0], estimate[1], estimate[2])
        hrf = utils.double_gamma_hrf(estimate[3], tr_length, frames_per_tr)
        response = utils.zscore(ss.fftconvolve(stim,hrf)[0:len(stim)])
        timeseries.append(response)
        
    # package the data structure
    dat = zip(timeseries,
              repeat(gaussian_model,num_voxels),
              search_bounds,
              fit_bounds,
              repeat(tr_length,num_voxels),
              indices,
              repeat(0.20,num_voxels),
              repeat(False,num_voxels))
              
    # run analysis
    num_cpus = multiprocessing.cpu_count()-1
    pool = multiprocessing.Pool(num_cpus)
    output = pool.map(gaussian.parallel_fit,dat)
    pool.close()
    pool.join()
    
    # assert equivalence
    for gaussian_fit,estimate in zip(output,estimates):
        nt.assert_almost_equal(gaussian_fit.x,estimate[0])
        nt.assert_almost_equal(gaussian_fit.y,estimate[1])
        nt.assert_almost_equal(gaussian_fit.sigma,estimate[2])
        nt.assert_almost_equal(gaussian_fit.hrf_delay,estimate[3])

