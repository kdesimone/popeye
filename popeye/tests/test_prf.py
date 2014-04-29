import os

import numpy as np
import numpy.testing as npt
import nose.tools as nt

import popeye.utilities as utils
import popeye.gaussian as gaussian
from popeye.base import PopulationModel, PopulationFit
from popeye.stimulus import Stimulus, simulate_bar_stimulus
from popeye.spinach import MakeFastPrediction

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
    
def test_error_function(stimulus):
    """
    Test voxel-wise gaussian estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # stimulus features
    pixels_across = 800 
    pixels_down = 600
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_steps = 30
    ecc = 10
    tr_length = 1.0
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
    
    # instantiate an instance of the Stimulus class
    stimulus = Stimulus(bar, viewing_distance, screen_width, 0.05, 0, 0)
    
    # set up bounds for the grid search
    bounds = ((-10,10),(-10,10),(0.25,5.25),(-5,5))
    
    # initialize the gaussian model
    gaussian_model = gaussian.GaussianModel(stimulus)
    
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
                                            
    # get the precomputed error
    gold_standard = estimate[0,4]
    
    # assert equal to 3 decimal places
    npt.assert_equal(test_results, 0.0)


def test_adapative_brute_force_grid_search(stimulus):
    """
    Test voxel-wise gaussian estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # stimulus features
    pixels_across = 800 
    pixels_down = 600
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_steps = 30
    ecc = 10
    tr_length = 1.0
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
    
    # instantiate an instance of the Stimulus class
    stimulus = Stimulus(bar, viewing_distance, screen_width, 0.05, 0, 0)
    
    # set up bounds for the grid search
    bounds = ((-10,10),(-10,10),(0.25,5.25),(-5,5))
    
    # initialize the gaussian model
    gaussian_model = gaussian.GaussianModel(stimulus)
    
    # generate a random pRF estimate
    estimate = [1,1,1,1]
    
    # generate the modeled BOLD response`
    response = MakeFastPrediction(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, estimate[0], estimate[1], estimate[2])
    hrf = gaussian.double_gamma_hrf(estimate[3], 1)
    response = utils.zscore(np.convolve(response,hrf)[0:len(response)])
    
    # compute the initial guess with the adaptive brute-force grid-search
    x0, y0, s0, hrf0 = gaussian.adaptive_brute_force_grid_search(bounds,
                                                                 1,
                                                                 3,
                                                                 response,
                                                                 stimulus.deg_x_coarse,
                                                                 stimulus.deg_y_coarse,
                                                                 stimulus.stim_arr_coarse,
                                                                 tr_length)
                                                             
    
    # package some of the results for comparison with known results
    test_results = np.round(np.array([x0,y0,s0,hrf0]))
    
    # assert
    npt.assert_equal(estimate, test_results)

def test_gaussian_fit():
    
    # stimulus features
    pixels_across = 800 
    pixels_down = 600
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_steps = 30
    ecc = 10
    tr_length = 1.0
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
    
    # instantiate an instance of the Stimulus class
    stimulus = Stimulus(bar, viewing_distance, screen_width, 0.05, 0, 0)
    
    # set up bounds for the grid search
    bounds = ((-10,10),(-10,10),(0.25,5.25),(-5,5))
    
    # initialize the gaussian model
    gaussian_model = gaussian.GaussianModel(stimulus)
    
    # generate a random pRF estimate
    estimate = []
    estimate.append(np.random.uniform(bounds[0][0],bounds[0][1]))
    estimate.append(np.random.uniform(bounds[1][0],bounds[1][1]))
    estimate.append(np.random.uniform(bounds[2][0],bounds[2][1]))
    estimate.append(np.random.uniform(bounds[3][0],bounds[3][1]))
    
    # generate the modeled BOLD response`
    response = MakeFastPrediction(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, estimate[0], estimate[1], estimate[2])
    hrf = gaussian.double_gamma_hrf(estimate[3], 1)
    response = utils.zscore(np.convolve(response,hrf)[0:len(response)])
    
    # fit the response
    gaussian_fit = gaussian.GaussianFit(response, gaussian_model, bounds, tr_length, [0,0,0], 0, False)
    
    # assert equivalence
    nt.assert_almost_equal(gaussian_fit.x,estimate[0])
    nt.assert_almost_equal(gaussian_fit.y,estimate[1])
    nt.assert_almost_equal(gaussian_fit.sigma,estimate[2])
    nt.assert_almost_equal(gaussian_fit.hrf_delay,estimate[3])
    
