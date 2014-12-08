# import os
# import multiprocessing
# from itertools import repeat
# import ctypes
# 
# import numpy as np
# import numpy.testing as npt
# import nose.tools as nt
# from scipy.signal import fftconvolve
# 
# import popeye.utilities as utils
# from popeye import dog, og
# from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus, resample_stimulus
# 
# def test_dog():
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
#     scale_factor = 0.20
#     dtype = ctypes.c_short
#     
#     # create the sweeping bar stimulus in memory
#     bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, screen_width, thetas, num_steps, ecc)
#     
#     # resample the stimulus to 25% of original
#     bar = resample_stimulus(bar, 0.25)
#     
#     # create an instance of the Stimulus class
#     stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, dtype)
#     
#     # initialize the gaussian model
#     model = dog.DifferenceOfGaussiansModel(stimulus)
#     
#     # set the pRF params
#     x = -5.2
#     y =  2.5
#     sigma_center = 1.2
#     sigma_surround = 2.9
#     beta_center = 2.5
#     beta_surround = 1.6
#     hrf_delay = -0.2
#     
#     # create "data"
#     data = dog.compute_model_ts(x, y, sigma_center, sigma_surround, beta_center, beta_surround, hrf_delay,
#                                 stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, tr_length)
#     
#     # first fit the one gaussian
#     search_bounds = ((-10,10),(-10,10),(0.25,5.25),(0.1,1e2),(-5,5))
#     fit_bounds = ((-12,12),(-12,12),(1/stimulus.ppd,12),(0.1,1e3),(-5,5))
#     og_fit = og.GaussianFit(model, data, search_bounds, fit_bounds, tr_length, (1,2,3), False, False)
#     
#     # then fit the two gaussian
#     fit_bounds = ((-12,12),(-12,12),(1/stimulus.ppd,12),(1/stimulus.ppd,12),(0.1,1e2),(0.1,1e2),(-5,5),)
#     dog_fit = dog.DifferenceOfGaussiansFit(og_fit, fit_bounds, True, False)
#     
#     # assert equivalence
#     nt.assert_almost_equal(dog_fit.x, x, 2)
#     nt.assert_almost_equal(dog_fit.y, y, 2)
#     nt.assert_almost_equal(dog_fit.sigma_center, sigma_center, 2)
#     nt.assert_almost_equal(dog_fit.sigma_surround, sigma_surround, 2)
#     nt.assert_almost_equal(dog_fit.beta_center, beta_center, 2)
#     nt.assert_almost_equal(dog_fit.beta_surround, beta_surround, 2)
#     nt.assert_almost_equal(dog_fit.hrf_delay, hrf_delay, 2)
# 
