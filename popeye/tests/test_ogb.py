import os
import multiprocessing
from itertools import repeat
import ctypes

import numpy as np
import numpy.testing as npt
import nose.tools as nt
from scipy.signal import fftconvolve

import popeye.utilities as utils
import popeye.ogb as ogb
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus, resample_stimulus

def test_ogb_fit():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_blank_steps = 20
    num_bar_steps = 40
    ecc = 12
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.20
    resample_factor = 0.25
    pixels_across = 800 * resample_factor
    pixels_down = 600 * resample_factor
    dtype = ctypes.c_int16
    Ns = 3
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    
    # insert blanks
    thetas = list(thetas)
    thetas.insert(0,-1)
    thetas.insert(2,-1)
    thetas.insert(5,-1)
    thetas.insert(8,-1)
    thetas.insert(11,-1)
    thetas.append(-1)
    thetas = np.array(thetas)
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, dtype)
    
    # set search grid
    x_grid = (-10,10)
    y_grid = (-10,10)
    s_grid = (1/stimulus.ppd,5.25)
    h_grid = (-4.0,4.0)
    b_grid = (0.1,1.0)
    bl_grid = (-5.0,5.0)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (1/stimulus.ppd,12.0)
    h_bound = (-5.0,5.0)
    b_bound = (1e-8,1e2)
    bl_bound = (-100,100)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, h_grid, b_grid, bl_grid,)
    bounds = (x_bound, y_bound, s_bound, h_bound, b_bound, bl_bound,)
    
    # initialize the gaussian model
    model = ogb.GaussianModel(stimulus)
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 1
    hrf_delay = -0.25
    baseline = 0
    
    # create the "data"
    data = ogb.compute_model_ts(x, y, sigma, hrf_delay, beta, baseline,
                               stimulus.deg_x, stimulus.deg_y, 
                               stimulus.stim_arr, tr_length)
    
    # fit the response
    fit = ogb.GaussianFit(model, data, grids, bounds, Ns, tr_length, voxel_index, verbose)
    
    # assert equivalence
    nt.assert_almost_equal(fit.x, x, 2)
    nt.assert_almost_equal(fit.y, y, 2)
    nt.assert_almost_equal(fit.sigma, sigma, 2)
    nt.assert_almost_equal(fit.beta, beta, 2)
    nt.assert_almost_equal(fit.hrf_delay, hrf_delay, 2)
    nt.assert_almost_equal(fit.hrf_delay, hrf_delay, 2)

# def test_parallel_og_fit():
# 
#     pixels_across = 800
#     pixels_down = 600
#     viewing_distance = 38
#     screen_width = 25
#     thetas = np.arange(0,360,45)
#     num_blank_steps = 20
#     num_bar_steps = 40
#     ecc = 12
#     tr_length = 1.0
#     frames_per_tr = 1.0
#     scale_factor = 0.20
#     resample_factor = 0.25
#     dtype = ctypes.c_uint8
#     num_voxels = multiprocessing.cpu_count()-1
#     
#     # create the sweeping bar stimulus in memory
#     bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
#                                 screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
#     
#     # resample the stimulus
#     bar = resample_stimulus(bar, resample_factor)
#     
#     # create an instance of the Stimulus class
#     stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, dtype)
#     
#     # set up bounds for the grid search
#     search_bounds = [((-10,10),(-10,10),(0.25,5.25),(0.1,1e2),(-5,5),)]*num_voxels
#     fit_bounds = [((-12,12),(-12,12),(1/stimulus.ppd,12),(0.1,1e3),(-5,5),)]*num_voxels
#     
#     # make fake voxel indices
#     indices = [(1,2,3)]*num_voxels
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
#     # create the simulated time-series
#     timeseries = []
#     for voxel in range(num_voxels):
#         
#         # create "data"
#         data = og.compute_model_ts(x, y, sigma, beta, hrf_delay,
#                                    stimulus.deg_x, stimulus.deg_y, 
#                                    stimulus.stim_arr, tr_length)
#         
#         
#         # append it
#         timeseries.append(data)
#         
#     # package the data structure
#     dat = zip(repeat(model,num_voxels),
#               timeseries,
#               search_bounds,
#               fit_bounds,
#               repeat(tr_length,num_voxels),
#               indices,
#               repeat(True,num_voxels),
#               repeat(True,num_voxels))
#               
#     # run analysis
#     pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
#     output = pool.map(og.parallel_fit,dat)
#     pool.close()
#     pool.join()
#     
#     # assert equivalence
#     for fit in output:
#         nt.assert_almost_equal(fit.x, x, 2)
#         nt.assert_almost_equal(fit.y, y, 2)
#         nt.assert_almost_equal(fit.sigma, sigma, 2)
#         nt.assert_almost_equal(fit.beta, beta, 2)
#         nt.assert_almost_equal(fit.hrf_delay, hrf_delay, 2)