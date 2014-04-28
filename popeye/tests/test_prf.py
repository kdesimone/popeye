import os

import numpy as np
import numpy.testing as npt
import nose.tools as nt

import popeye.utilities as utils
import popeye.prf as prf
from popeye.base import PopulationModel, PopulationFit
from popeye.stimulus import Stimulus, simulate_bar_stimulus

def test_double_gamma_hrf():
    """
    Test voxel-wise prf estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # produce an HRF with 0 delay
    hrf0 = prf.double_gamma_hrf(0)
    hrf0 = np.round(np.sum(hrf0)*100,2)
    
    # produce an HRF with +1 delay
    hrf_pos = prf.double_gamma_hrf(1)
    hrf_pos = np.round(np.sum(hrf_pos)*100,2)
    
    # produce an HRF with -1 delay
    hrf_neg = prf.double_gamma_hrf(-1)
    hrf_neg = np.round(np.sum(hrf_neg)*100,2)
    
    # assert 
    nt.assert_almost_equal(hrf0, 80.02, 2)
    nt.assert_almost_equal(hrf_pos, 80.01, 2)
    nt.assert_almost_equal(hrf_neg, 80.11, 2)
    
def test_error_function(stimulus):
    """
    Test voxel-wise prf estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # set the path to data
    data_path = os.path.join(os.path.expanduser('~'), '.popeye/popeye')
    
    # load the datasets
    response = np.load('%s/sample_response.npy' %(data_path))
    estimate = np.load('%s/sample_estimate.npy' %(data_path))
    
    
    # grab a voxel's time-series and z-score it
    ts_voxel = response[0,stimulus.clip_number::]
    ts_voxel = utils.zscore(ts_voxel)
    
    # compute the error using the results of a known pRF estimation
    test_results = prf.error_function(estimate[0,0:4],
                                      ts_voxel,
                                      stimulus.deg_x,
                                      stimulus.deg_y,
                                      stimulus.stim_arr)
                                      
    # get the precomputed error
    gold_standard = estimate[0,4]
    
    # assert equal to 3 decimal places
    npt.assert_almost_equal(gold_standard, test_results)


def test_adapative_brute_force_grid_search(stimulus):
    """
    Test voxel-wise prf estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # set the path to data
    data_path = os.path.join(os.path.expanduser('~'), '.popeye/popeye')
    
    # load the datasets
    response = np.load('%s/sample_response.npy' %(data_path))
    
    # bounds for the 4-parameter brute-force grid-search
    bounds = ((-10,10),(-10,10),(0.25,5.25),(-4,4))
    
    # grab a voxel's time-series and z-score it
    ts_voxel = response[0, stimulus.clip_number::]
    ts_voxel = utils.zscore(ts_voxel)
    
    # compute the initial guess with the adaptive brute-force grid-search
    x0, y0, s0, hrf0 = pest.adaptive_brute_force_grid_search(bounds,
                                                             1,
                                                             3,
                                                             ts_voxel,
                                                             stimulus.deg_x_coarse,
                                                             stimulus.deg_y_coarse,
                                                             stimulus.stim_arr_coarse)
                                                             
    # grab the known pRF estimate for the sample data
    gold_standard = np.array([ 8.446,  2.395,  0.341, -0.777])
    
    # package some of the results for comparison with known results
    test_results = np.round(np.array([x0,y0,s0,hrf0]),3)
    
    # assert
    npt.almost_equal(np.any(gold_standard == test_results))

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
    prf_model = prf.GaussianModel(stimulus)
    
    # generate a random pRF estimate
    estimate = []
    estimate.append(np.random.uniform(bounds[0][0],bounds[0][1]))
    estimate.append(np.random.uniform(bounds[1][0],bounds[1][1]))
    estimate.append(np.random.uniform(bounds[2][0],bounds[2][1]))
    estimate.append(np.random.uniform(bounds[3][0],bounds[3][1]))
    
    # generate the modeled BOLD response`
    response = prf.MakeFastPrediction(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, estimate[0], estimate[1], estimate[2])
    hrf = prf.double_gamma_hrf(estimate[3], 1)
    response = utils.zscore(np.convolve(response,hrf)[0:len(response)])
    
    # fit the response
    prf_fit = prf.GaussianFit(prf_model, response, bounds, tr_length)
    
    # assert equivalence
    nt.assert_almost_equal(prf_fit.x,estimate[0])
    nt.assert_almost_equal(prf_fit.y,estimate[1])
    nt.assert_almost_equal(prf_fit.sigma,estimate[2])
    nt.assert_almost_equal(prf_fit.hrf_delay,estimate[3])
    
