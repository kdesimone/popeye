import multiprocessing
from itertools import repeat

import numpy as np
import numpy.testing as npt
import nose.tools as nt
from scipy.signal import fftconvolve

import popeye.utilities as utils
import popeye.dog as dog
import popeye.og as gaussian
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus
from popeye.spinach import generate_og_timeseries

def test_difference_of_gaussians_fit():
    
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
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor)
    
    # initialize the gaussian model
    model = dog.DifferenceOfGaussiansModel(stimulus)
    
    # set the pRF params
    x = -5.24
    y =  2.58
    sigma_center = 1.24
    sigma_surround = 2.89
    beta_center = 2.5
    beta_surround = 1.66
    hrf_delay = -0.26
    
    # set up bounds for search
    beta_bounds = (0.1,3)
    search_bounds = ((-10,10),(-10,10),(0.25,5.25),(0.25, 5.25),beta_bounds,beta_bounds,(-5,5),)
    fit_bounds = ((-12,12),(-12,12),(1/stimulus.ppd,12),(1/stimulus.ppd,5),beta_bounds,beta_bounds,(-5,5))
    
    # generate some data
    data = dog.compute_model_ts(x, y, sigma_center, sigma_surround, beta_center, beta_surround, hrf_delay,
                                model.stimulus.deg_x, model.stimulus.deg_y, model.stimulus.stim_arr, tr_length)
    
    # first fit the gaussian
    search_bounds = ((-10,10),(-10,10),(0.25,5.25),(0.1,1e2),(-5,5))
    fit_bounds = ((-12,12),(-12,12),(1/stimulus.ppd,12),(0.1,1e2),(-5,5))
    gaussian_fit = gaussian.GaussianFit(model, data, fit, search_bounds, fit_bounds, tr_length)
    
    # then fit the dog
    fit_bounds = ((-12,12),(-12,12),(1/stimulus.ppd,12),(1/stimulus.ppd,12),(0.1,1e2),(0.1,1e2),(-5,5))
    dog_fit = dog.DifferenceOfGaussiansFit(model, data, gaussian_fit, search_bounds, fit_bounds, tr_length)
    
    # assert equivalence
    nt.assert_almost_equal(fit.x,estimate[0])
    nt.assert_almost_equal(fit.y,estimate[1])
    nt.assert_almost_equal(fit.sigma,estimate[2])
    nt.assert_almost_equal(fit.hrf_delay,estimate[3])
    nt.assert_almost_equal(fit.beta,estimate[4])
    

def test_parallel_gaussian_fit():

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
    num_voxels = multiprocessing.cpu_count()-1
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor)
    
    # set up bounds for the grid search
    search_bounds = [((-10,10),(-10,10),(0.25,5.25),(-5,5),(0.1,1e2))]*num_voxels
    fit_bounds = [((-12,12),(-12,12),(1/stimulus.ppd,12),(-5,5),(0.1,1e2))]*num_voxels
    
    # make fake voxel indices
    indices = [(1,2,3)]*num_voxels
    
    # initialize the gaussian model
    model = gaussian.GaussianModel(stimulus)
    
    # generate a random pRF estimate
    estimate = [-5.24, 2.583, 1.24, -0.25, 2.5]
    estimates = [estimate]*num_voxels
    
    # create the simulated time-series
    timeseries = []
    for estimate in estimates:
        
        # make the stim time-series
        stim = generate_og_timeseries(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, 
                                            estimate[0], estimate[1], estimate[2], estimate[-1])
        
        # create the HRF
        hrf = utils.double_gamma_hrf(estimate[3], tr_length)
        
        # simulate the BOLD response
        response = fftconvolve(stim,hrf)[0:len(stim)]
        # append it
        timeseries.append(response)
        
    # package the data structure
    dat = zip(repeat(model,num_voxels),
              timeseries,
              search_bounds,
              fit_bounds,
              repeat(tr_length,num_voxels),
              indices,
              repeat(True,num_voxels),
              repeat(True,num_voxels))
              
    # run analysis
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    output = pool.map(gaussian.parallel_fit,dat)
    pool.close()
    pool.join()
    
    # assert equivalence
    for fit,estimate in zip(output,estimates):
        nt.assert_almost_equal(fit.x,estimate[0])
        nt.assert_almost_equal(fit.y,estimate[1])
        nt.assert_almost_equal(fit.sigma,estimate[2])
        nt.assert_almost_equal(fit.hrf_delay,estimate[3])
        nt.assert_almost_equal(fit.beta,estimate[-1])

