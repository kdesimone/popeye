import os
import multiprocessing
from itertools import repeat

import numpy as np
import numpy.testing as npt

import scipy.signal as ss

import popeye.utilities as utils
import popeye.gabor as gabor
import popeye.gaussian as gaussian
from popeye.spinach import MakeFastGaussPrediction, MakeFastGaborPrediction
from popeye.visual_stimulus import simulate_bar_stimulus, VisualStimulus

# def test_gabor_fit():
#     
#     # stimulus features
#     pixels_across = 800 
#     pixels_down = 600
#     viewing_distance = 38
#     screen_width = 25
#     tr_length = 1.0
#     clip_number = 25
#     roll_number = 0
#     frames_per_tr = 8
#     
#     movie = np.load('/Users/kevin/Desktop/battleship.npy')
#     
#     # create an instance of the Stimulus class
#     stimulus = VisualStimulus(movie, viewing_distance, screen_width, 0.10, clip_number, roll_number, frames_per_tr)
#     
#     # flush the movie
#     movie = []
#     
#     # initialize the gabor model
#     gabor_model = gabor.GaborModel(stimulus)
#     
#     # generate a random gabor pRF estimate
#     estimate = np.double([1, 1, 1, 0.2, 284, 185, 0.25])
#     
#     # generate the modeled BOLD response
#     response = gabor.compute_model_ts(estimate[0],estimate[1],estimate[2],estimate[3],
#                                       estimate[4],estimate[5],estimate[6],
#                                       stimulus.deg_x,stimulus.deg_y,
#                                       stimulus.stim_arr, tr_length, frames_per_tr, utils.zscore)
#     
#     # set the bounds for the parameter search
#     bounds = ((-10, 10),(-10, 10),(0.25,1.25),(-1,1),(0,360),(0,360),(0.01,2))
#     
#     # fit the response
#     gabor_fit = gabor.GaborFit(response, gabor_model, bounds, tr_length, [0,0,0], 0, False, None, True)
#     
#     # assert equivalence
#     nt.assert_almost_equal(gabor_fit.x,estimate[0])
#     nt.assert_almost_equal(gabor_fit.y,estimate[1])
#     nt.assert_almost_equal(gabor_fit.sigma,estimate[2])
#     nt.assert_almost_equal(gabor_fit.hrf_delay,estimate[3])
#     
# 
# def test_parallel_fit():
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
#     num_voxels = 50
#     tr_length = 1.0
#     frames_per_tr = 1.0
#     
#     # create the sweeping bar stimulus in memory
#     bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
#     
#     # create an instance of the Stimulus class
#     stimulus = VisualStimulus(bar, viewing_distance, screen_width, 0.05, 0, 0)
#     
#     # set up bounds for the grid search
#     bounds = [((-10,10),(-10,10),(0.25,5.25),(-5,5))]*num_voxels
#     indices = [(1,2,3)]*num_voxels
#     
#     # initialize the gabor model
#     gabor_model = gabor.GaborModel(stimulus)
#     
#     # generate a random pRF estimate
#     estimates = [(1,1,1,1)]*num_voxels
#     
#     # create the simulated time-series
#     timeseries = []
#     for estimate in estimates:
#         response = MakeFastGaborPrediction(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, estimate[0], estimate[1], estimate[2])
#         hrf = utils.double_gamma_hrf(estimate[3], tr_length, frames_per_tr)
#         response = utils.zscore(np.convolve(response,hrf)[0:len(response)])
#         timeseries.append(response)
#         
#     # package the data structure
#     dat = zip(timeseries,
#               repeat(gabor_model,num_voxels),
#               bounds,
#               repeat(tr_length,num_voxels),
#               indices,
#               repeat(0.20,num_voxels),
#               repeat(False,num_voxels))
#               
#     # run analysis
#     num_cpus = multiprocessing.cpu_count()-1
#     pool = multiprocessing.Pool(num_cpus)
#     output = pool.map(gabor.parallel_fit,dat)
#     pool.close()
#     pool.join()
#     
#     # assert equivalence
#     for gabor_fit,estimate in zip(output,estimates):
#         npt.assert_almost_equal(gabor_fit.x,estimate[0])
#         npt.assert_almost_equal(gabor_fit.y,estimate[1])
#         npt.assert_almost_equal(gabor_fit.sigma,estimate[2])
#         npt.assert_almost_equal(gabor_fit.hrf_delay,estimate[3])

