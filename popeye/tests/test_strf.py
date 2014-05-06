from __future__ import division

import numpy as np
import numpy.testing as npt
import nose.tools as nt

from scipy.signal import chirp

import popeye.utilities as utils
import popeye.strf as strf
from popeye.auditory_stimulus import AuditoryStimulus

def test_double_gamma_hrf():
    """
    Test voxel-wise gaussian estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # set the TR length ... this affects the HRF sampling rate ...
    tr_length = 1.0
    
    # compute the difference in area under curve for hrf_delays of -1 and 0
    diff_1 = np.abs(np.sum(strf.double_gamma_hrf(-1, tr_length))-np.sum(strf.double_gamma_hrf(0, tr_length)))
    
    # compute the difference in area under curver for hrf_delays of 0 and 1
    diff_2 = np.abs(np.sum(strf.double_gamma_hrf(1, tr_length))-np.sum(strf.double_gamma_hrf(0, tr_length)))
    
    npt.assert_almost_equal(diff_1, diff_2, 2)
    
def test_error_function():
    """
    Test voxel-wise gaussian estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # stimulus features
    lo_freq = 1000 # Hz
    hi_freq = 20000 # Hz
    fs = 44100 # Hz
    duration = 100 # seconds
    tr_length = 1.0 # seconds
    window_size = 0.01 # seconds
    num_timepoints = np.floor(duration / tr_length)
    
    
    # sample the time from 0-duration by the fs
    time = np.linspace(0,duration,duration*fs)
    
    # create the sweeping bar stimulus in memory
    signal = chirp(time, lo_freq, duration, hi_freq, method='linear')
    
    # instantiate an instance of the Stimulus class
    stimulus = AuditoryStimulus(signal, tr_length)
    
    # set up bounds for the grid search
    bounds = ((lo_freq, hi_freq),(lo_freq, hi_freq),(0,duration),(-5,5))
    
    # initialize the gaussian model
    strf_model = strf.SpectrotemporalModel(stimulus)
    
    # makeup a STRF estimate
    freq_center = 2000 # center frequency
    freq_sigma = 500 # frequency dispersion
    time_sigma = 0.5 # # seconds
    hrf_delay = 0.0 # seconds
    
    response = strf.compute_model_ts(freq_center, freq_sigma, time_sigma, hrf_delay,
                                     stimulus.time_coord, stimulus.freq_coord, stimulus.spectrogram,
                                     tr_length, num_timepoints, window_size, norm_func=utils.zscore)
    
    
    test_results = strf.error_function([freq_center, freq_sigma, time_sigma, hrf_delay], response, 
                                        stimulus.time_coord, stimulus.freq_coord, stimulus.spectrogram, 
                                        tr_length, num_timepoints, window_size)
    
    
    # assert equal to 3 decimal places
    npt.assert_equal(test_results, 0.0)


def test_adapative_brute_force_grid_search():
    """
    Test voxel-wise gaussian estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # stimulus features
    lo_freq = 100 # Hz
    hi_freq = 1000 # Hz
    fs = 1000 # Hz
    duration = 100 # seconds
    tr_length = 1.0 # seconds
    window_size = 0.5 # seconds
    num_timepoints = np.floor(duration / tr_length)
    
    # sample the time from 0-duration by the fs
    time = np.linspace(0,duration,duration*fs)
    
    # create the sweeping bar stimulus in memory
    signal = chirp(time, lo_freq, duration, hi_freq, method='linear')
    
    # instantiate an instance of the Stimulus class
    stimulus = AuditoryStimulus(signal, tr_length)
    
    # set up bounds for the grid search
    bounds = ((lo_freq, hi_freq),(lo_freq, hi_freq),(0,tr_length),(-5,5))
    
    # initialize the gaussian model
    strf_model = strf.SpectrotemporalModel(stimulus)
    
    # makeup a STRF estimate
    freq_center = 550 # center frequency
    freq_sigma = 100 # frequency dispersion
    time_sigma = 0.250 # # seconds
    hrf_delay = 0.0 # seconds
    
    # generate the modeled BOLD response
    response = strf.compute_model_ts(freq_center, freq_sigma, time_sigma, hrf_delay,
                                     stimulus.time_coord, stimulus.freq_coord, stimulus.spectrogram,
                                     tr_length, num_timepoints, window_size, norm_func=utils.zscore)
    
    # compute the initial guess with the adaptive brute-force grid-search
    f0, fs0, ts0, hrf0 = strf.adaptive_brute_force_grid_search(bounds, 1, 3,
                                                               response,
                                                               stimulus.time_coord,
                                                               stimulus.freq_coord,
                                                               stimulus.spectrogram,
                                                               tr_length, num_timepoints, window_size)
                                                             
        
    # assert
    npt.assert_equal([freq_center, freq_sigma, time_sigma, hrf_delay], [f0,fs0,ts0,hrf0])

def test_strf_fit():
    
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
    gaussian_model = strf.GaussianModel(stimulus)
    
    # generate a random pRF estimate
    estimate = []
    estimate.append(np.random.uniform(bounds[0][0],bounds[0][1]))
    estimate.append(np.random.uniform(bounds[1][0],bounds[1][1]))
    estimate.append(np.random.uniform(bounds[2][0],bounds[2][1]))
    estimate.append(np.random.uniform(bounds[3][0],bounds[3][1]))
    
    # generate the modeled BOLD response`
    response = MakeFastPrediction(stimulus.deg_x, stimulus.deg_y, stimulus.stim_arr, estimate[0], estimate[1], estimate[2])
    hrf = strf.double_gamma_hrf(estimate[3], 1)
    response = utils.zscore(np.convolve(response,hrf)[0:len(response)])
    
    # fit the response
    gaussian_fit = strf.GaussianFit(response, gaussian_model, bounds, tr_length, [0,0,0], 0, False)
    
    # assert equivalence
    nt.assert_almost_equal(gaussian_fit.x,estimate[0])
    nt.assert_almost_equal(gaussian_fit.y,estimate[1])
    nt.assert_almost_equal(gaussian_fit.sigma,estimate[2])
    nt.assert_almost_equal(gaussian_fit.hrf_delay,estimate[3])
    
