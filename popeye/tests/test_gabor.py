import os
import multiprocessing
from itertools import repeat

import numpy as np
import numpy.testing as npt
import nose.tools as nt

import popeye.utilities as utils
import popeye.gabor as gabor
from popeye.movie_stimulus import MovieStimulus
from popeye.visual_stimulus import simulate_bar_stimulus, VisualStimulus

def test_double_gamma_hrf():
    """
    Test voxel-wise gaussian estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # set the TR length ... this affects the HRF sampling rate ...
    tr_length = 1.0
    
    # compute the difference in area under curve for hrf_delays of -1 and 0
    diff_1 = np.abs(np.sum(gaussian.double_gamma_hrf(-1, tr_length))-np.sum(gaussian.double_gamma_hrf(0, tr_length)))
    
    # compute the difference in area under curver for hrf_delays of 0 and 1
    diff_2 = np.abs(np.sum(gaussian.double_gamma_hrf(1, tr_length))-np.sum(gaussian.double_gamma_hrf(0, tr_length)))
    
    npt.assert_almost_equal(diff_1, diff_2, 2)
    
def test_error_function():
    """
    Test voxel-wise gabor estimation error function in popeye.gabor 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # stimulus features
    pixels_across = 800 
    pixels_down = 600
    viewing_distance = 38
    screen_width = 25
    tr_length = 1.5
    
    # create the sweeping bar stimulus in memory
    movie = np.load('/Users/kevin/Desktop/battleship.npy')
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(movie, viewing_distance, screen_width, 0.10, 0, 0)
    
    # flush the movie
    movie = []
    
    # set up bounds for the grid search
    bounds = ((-10,10),(-10,10),(0.25,5.25),(0,360),(0,360),(0.25,10.25),(-5,5))
    
    # initialize the gaussian model
    gabor_model = gabor.GaborModel(stimulus)
    
    # generate a random pRF estimate
    estimate = [1,1,1,1]
    
    response = MakeFastPrediction(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, estimate[0], estimate[1], estimate[2])
    hrf = gaussian.double_gamma_hrf(estimate[3], 1)
    response = utils.zscore(np.convolve(response,hrf)[0:len(response)])
    
    # compute the error using the results of a known pRF estimation
    test_results = gaussian.error_function(estimate,
                                           response,
                                           stimulus.deg_x,
                                           stimulus.deg_y,
                                           stimulus.stim_arr,
                                           tr_length)
    
    # assert equal to 3 decimal places
    npt.assert_equal(test_results, 0.0)

def test_gabor_fit():
    
    # stimulus features
    pixels_across = 800 
    pixels_down = 600
    viewing_distance = 38
    screen_width = 25
    tr_length = 1.5
    
    movie = np.load('/Users/kevin/Desktop/battleship.npy')
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(movie, viewing_distance, screen_width, 0.10, 0, 0)
    
    # flush the movie
    movie = []
    
    # initialize the gaussian model
    gabor_model = gabor.GaborModel(stimulus)
    
    # generate a random gabor pRF estimate
    # [x0,y0,s0,hrf,theta,phi,cpd]
    estimate = np.double([1,1,0.95,0,48,67,0.75])
    
    # generate the modeled BOLD response
    response = gabor.compute_model_ts(estimate[0],estimate[1],estimate[2],estimate[3],
                                      estimate[4],estimate[5],estimate[6],
                                      stimulus.deg_x,stimulus.deg_y,
                                      stimulus.stim_arr, tr_length)
    
    
    # set the bounds for the parameter search
    bounds = ((-8,8),(-5,5),(0.25,1.25),(-1,1),(0,90),(0,90),(0.01,2))
    
    # fit the response
    gabor_fit = gabor.GaborFit(response, gabor_model, bounds, tr_length, [0,0,0], 0, True)
    
    # make the gabor for demonstration
    g_set = MakeFastGabor(stimulus.deg_x,stimulus.deg_y,estimate[0],estimate[1],estimate[2],
                      estimate[4],estimate[5],estimate[6])
    
    g_fit = MakeFastGabor(stimulus.deg_x,stimulus.deg_y,gabor_fit.x,gabor_fit.y,gabor_fit.sigma,
                          gabor_fit.theta, gabor_fit.phi, gabor_fit.cpd)
    
    # assert equivalence
    nt.assert_almost_equal(gaussian_fit.x,estimate[0])
    nt.assert_almost_equal(gaussian_fit.y,estimate[1])
    nt.assert_almost_equal(gaussian_fit.sigma,estimate[2])
    nt.assert_almost_equal(gaussian_fit.hrf_delay,estimate[3])
    

def test_parallel_fit():

    # stimulus features
    pixels_across = 800 
    pixels_down = 600
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_steps = 20
    ecc = 10
    tr_length = 1.0
    num_voxels = 50
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, 0.05, 0, 0)
    
    # set up bounds for the grid search
    bounds = [((-10,10),(-10,10),(0.25,5.25),(-5,5))]*num_voxels
    indices = [(1,2,3)]*num_voxels
    
    # initialize the gaussian model
    gaussian_model = gaussian.GaussianModel(stimulus)
    
    # generate a random pRF estimate
    estimates = [(1,1,1,1)]*num_voxels
    
    # create the simulated time-series
    timeseries = []
    for estimate in estimates:
        response = MakeFastPrediction(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, estimate[0], estimate[1], estimate[2])
        hrf = gaussian.double_gamma_hrf(estimate[3], 1)
        response = utils.zscore(np.convolve(response,hrf)[0:len(response)])
        timeseries.append(response)
        
    # package the data structure
    dat = zip(timeseries,
              repeat(gaussian_model,num_voxels),
              bounds,
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

