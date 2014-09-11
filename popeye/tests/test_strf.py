from __future__ import division

from itertools import repeat
import multiprocessing

import numpy as np
import numpy.testing as npt
import nose.tools as nt

from scipy.signal import chirp

import popeye.utilities as utils
import popeye.strf_no_noise as strf
from popeye.auditory_stimulus import AuditoryStimulus
from popeye.spinach import MakeFastGaussian2D, MakeFastAudioPrediction

# def test_brute_force_search():
#     """
#     Test voxel-wise strf estimation function in popeye.estimation 
#     using the stimulus and BOLD time-series data that ship with the 
#     popeye installation.
#     """
#     
#     # stimulus features
#     lo_freq = 100 # Hz
#     hi_freq = 1000 # Hz
#     fs = hi_freq*2 # Hz
#     duration = 100 # seconds
#     tr_length = 1.0 # seconds
#     time_window = 0.5 # seconds
#     freq_window = 256 # this is 2x the number of freq bins we'll end up with in the spectrogram
#     
#     num_timepoints = np.floor(duration / tr_length)
#     
#     # sample the time from 0-duration by the fs
#     time = np.linspace(0,duration,duration*fs)
#     
#     # create the sweeping bar stimulus in memory
#     signal = chirp(time, lo_freq, duration, hi_freq, method='linear')
#     %
#     # instantiate an instance of the Stimulus class
#     stimulus = AuditoryStimulus(signal, tr_length, freq_window, time_window)
#     
#     # set up bounds for the grid search
#     bounds = ((lo_freq, hi_freq),(lo_freq, hi_freq),(-5,5))
#     
#     # initialize the strf model
#     strf_model = strf.SpectrotemporalModel(stimulus)
#     
#     # makeup a STRF estimate
#     freq_center = 550 # center frequency
#     freq_sigma = 100 # frequency dispersion
#     hrf_delay = 0.0 # seconds
#     
#     # generate the modeled BOLD response
#     response = strf.compute_model_ts(freq_center, freq_sigma, hrf_delay,
#                                      stimulus.time_coord, stimulus.freq_coord, stimulus.spectrogram,
#                                      tr_length, num_timepoints)
#     
#     # compute the initial guess with the adaptive brute-force grid-search
#     f0, fs0, hrf0 = strf.brute_force_search(bounds, response,
#                                             stimulus.time_coord,
#                                             stimulus.freq_coord,
#                                             stimulus.spectrogram,
#                                             tr_length, num_timepoints)
#                                                              
#         
#     # assert
#     npt.assert_equal([freq_center, freq_sigma, hrf_delay], [f0,fs0,hrf0])

def test_strf_fit():
    
    # stimulus features
    lo_freq = 200 # Hz
    hi_freq = 12000 # Hz
    fs = 44100.0 # Hz
    duration = 100 # seconds
    tr_length = 1.0 # seconds
    tr_sampling_rate = 10 # number of time-samples per TR to plot the STRF in
    time_window = 0.5 # seconds
    freq_window = 256 # this is 2x the number of freq bins we'll end up with in the spectrogram
    scale_factor = 1.0 # how much to downsample the spectrotemporal space
    num_timepoints = np.floor(duration / tr_length)
    degrees = 0.0
    
    # sample the time from 0-duration by the fs
    time = np.linspace(0,duration,duration*fs)
    
    # create a chirp stimulus
    signal = chirp(time, lo_freq, duration, hi_freq)
    
    # instantiate an instance of the Stimulus class
    stimulus = AuditoryStimulus(signal, tr_length, freq_window, time_window, sampling_rate=fs, 
                                tr_sampling_rate=tr_sampling_rate, scale_factor=scale_factor)
    
    
    # set some parameters for the mock STRF
    freq_center = 5678 # center frequency
    freq_sigma = 234 # frequency dispersion
    hrf_delay = 0.987 # seconds
    
    # initialize the strf model
    model = strf.SpectrotemporalModel(stimulus)
    
    # generate the modeled BOLD response
    data = strf.compute_model_ts(freq_center, freq_sigma, hrf_delay,
                                 model.stimulus.time_coord, 
                                 model.stimulus.freq_coord,
                                 model.stimulus.spectrogram.astype('double'),
                                 tr_length, num_timepoints, norm_func=utils.zscore)
    
    # set some searh parameters
    search_bounds = ((lo_freq, hi_freq),(lo_freq, hi_freq/2),(-5,5),)
    fit_bounds = ((lo_freq, hi_freq),(lo_freq, hi_freq/2),(-5,5),)
    
    # fit the response
    fit = strf.SpectrotemporalFit(data, model, search_bounds, fit_bounds, tr_length)
    
    # assert
    npt.assert_almost_equal(fit.estimate,[freq_center,freq_sigma,hrf_delay])

def test_parallel_fit():
    
    lo_freq = 200 # Hz
    hi_freq = 12000 # Hz
    fs = 44100.0 # Hz
    duration = 100 # seconds
    tr_length = 1.0 # seconds
    tr_sampling_rate = 10 # number of time-samples per TR to plot the STRF in
    time_window = 0.5 # seconds
    freq_window = 256 # this is 2x the number of freq bins we'll end up with in the spectrogram
    scale_factor = 1.0 # how much to downsample the spectrotemporal space
    num_timepoints = np.floor(duration / tr_length)
    degrees = 0.0
    num_voxels = multiprocessing.cpu_count()
    
    
    # sample the time from 0-duration by the fs
    time = np.linspace(0,duration,duration*fs)
    
    # create a chirp stimulus
    signal = chirp(time, lo_freq, duration, hi_freq)
    
    # instantiate an instance of the Stimulus class
    stimulus = AuditoryStimulus(signal, tr_length, freq_window, time_window, sampling_rate=fs, 
                                tr_sampling_rate=tr_sampling_rate, scale_factor=scale_factor)
    
    
    # invent a pRF estimate
    freq_center = 5678 # center frequency
    freq_sigma = 234 # frequency dispersion
    hrf_delay = 0.987 # seconds
    
    # initialize the strf model
    model = strf.SpectrotemporalModel(stimulus)
    
    # set some bounds
    search_bounds = [((lo_freq, hi_freq),(lo_freq, hi_freq/2),(-5,5),)]*num_voxels
    fit_bounds = [((lo_freq, hi_freq),(lo_freq, hi_freq/2),(-5,5),)]*num_voxels
    
    # generate the modeled BOLD response
    response = strf.compute_model_ts(freq_center, freq_sigma, hrf_delay,
                                     model.stimulus.time_coord, 
                                     model.stimulus.freq_coord,
                                     model.stimulus.spectrogram.astype('double'),
                                     tr_length, num_timepoints, norm_func=utils.zscore)
    
    
    # package the data structure
    dat = zip(repeat(response,num_voxels),
              repeat(model,num_voxels),
              search_bounds,
              fit_bounds,
              repeat(tr_length,num_voxels),
              repeat(None,num_voxels),
              repeat(True,num_voxels),
              repeat(False,num_voxels))
              
    # run analysis
    num_cpus = multiprocessing.cpu_count()-1
    pool = multiprocessing.Pool(num_cpus)
    output = pool.map(strf.parallel_fit,dat)
    pool.close()
    pool.join()
    
    # assert equivalence
    for fit in output:
        npt.assert_almost_equal(fit.estimate,[freq_center,freq_sigma,hrf_delay])
    
