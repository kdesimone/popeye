from __future__ import division

import multiprocessing

import numpy as np
import numpy.testing as npt
import nose.tools as nt

from scipy.signal import chirp

import popeye.utilities as utils
import popeye.strf as strf
from popeye.auditory_stimulus import AuditoryStimulus
from popeye.spinach import MakeFastGaussian2D, MakeFastAudioPrediction

def test_double_gamma_hrf():
    """
    Test voxel-wise strf estimation function in popeye.estimation 
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
    Test voxel-wise strf estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # stimulus features
    lo_freq = 100 # Hz
    hi_freq = 1000 # Hz
    fs = hi_freq*2 # Hz
    duration = 100 # seconds
    tr_length = 1.0 # seconds
    time_window = 0.1 # seconds
    freq_window = 256 # this is 2x the number of freq bins we'll end up with in the spectrogram
    degrees = 0.0
    
    num_timepoints = np.floor(duration / tr_length)
    
    # sample the time from 0-duration by the fs
    time = np.linspace(0,duration,duration*fs)
    
    # create the sweeping bar stimulus in memory
    signal = chirp(time, lo_freq, duration, hi_freq, method='linear')
    
    # instantiate an instance of the Stimulus class
    stimulus = AuditoryStimulus(signal, tr_length, freq_window, time_window)
    
    # set up bounds for the grid search
    bounds = ((lo_freq, hi_freq),(lo_freq, hi_freq),(0,tr_length),(-5,5))
    
    # initialize the strf model
    strf_model = strf.SpectrotemporalModel(stimulus)
    
    # makeup a STRF estimate
    freq_center = 550 # center frequency
    freq_sigma = 100 # frequency dispersion
    time_sigma = 0.250 # # seconds
    hrf_delay = 0.0 # seconds
    
    
    # generate the modeled BOLD response
    response = strf.compute_model_ts(freq_center, freq_sigma, time_sigma, hrf_delay,
                                     stimulus.time_coord, stimulus.freq_coord, stimulus.spectrogram,
                                     tr_length, num_timepoints, time_window, norm_func=utils.zscore)
    
    
    
    test_results = strf.error_function([freq_center, freq_sigma, time_sigma, hrf_delay], response, 
                                        stimulus.time_coord, stimulus.freq_coord, stimulus.spectrogram, 
                                        tr_length, num_timepoints, time_window)
    
    
    # assert equal
    npt.assert_equal(test_results, 0.0)


def test_ballpark_brute_force():
    """
    Test voxel-wise strf estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # stimulus features
    lo_freq = 100 # Hz
    hi_freq = 1000 # Hz
    fs = hi_freq*2 # Hz
    duration = 100 # seconds
    tr_length = 1.0 # seconds
    time_window = 0.5 # seconds
    freq_window = 256 # this is 2x the number of freq bins we'll end up with in the spectrogram
    
    num_timepoints = np.floor(duration / tr_length)
    
    # sample the time from 0-duration by the fs
    time = np.linspace(0,duration,duration*fs)
    
    # create the sweeping bar stimulus in memory
    signal = chirp(time, lo_freq, duration, hi_freq, method='linear')
    
    # instantiate an instance of the Stimulus class
    stimulus = AuditoryStimulus(signal, tr_length, freq_window, time_window)
    
    # set up bounds for the grid search
    bounds = ((lo_freq, hi_freq),(lo_freq, hi_freq),(-5,5))
    
    # initialize the strf model
    strf_model = strf.SpectrotemporalModel(stimulus)
    
    # makeup a STRF estimate
    freq_center = 550 # center frequency
    freq_sigma = 100 # frequency dispersion
    hrf_delay = 0.0 # seconds
    
    # generate the modeled BOLD response
    response = strf.compute_model_ts(freq_center, freq_sigma, hrf_delay,
                                     stimulus.time_coord, stimulus.freq_coord, stimulus.spectrogram,
                                     tr_length, num_timepoints, time_window, norm_func=utils.zscore)
    
    # compute the initial guess with the adaptive brute-force grid-search
    f0, fs0, ts0, hrf0 = strf.ballpark_brute_force_estimate(bounds, response,
                                                            stimulus.time_coord,
                                                            stimulus.freq_coord,
                                                            stimulus.spectrogram,
                                                            tr_length, num_timepoints, time_window)
                                                             
        
    # assert
    npt.assert_equal([freq_center, freq_sigma, time_sigma, hrf_delay], [f0,fs0,ts0,hrf0])

def test_strf_fit():
    
    # stimulus features
    lo_freq = 100 # Hz
    hi_freq = 1000 # Hz
    fs = hi_freq*2 # Hz
    duration = 100 # seconds
    tr_length = 1.0 # seconds
    tr_sampling_rate = 4 # number of time-samples per TR to plot the STRF in
    time_window = 0.5 # seconds
    freq_window = 256 # this is 2x the number of freq bins we'll end up with in the spectrogram
    scale_factor = 1.0 # how much to downsample the spectrotemporal space
    num_timepoints = np.floor(duration / tr_length)
    clip_number=10
    degrees = 0.0
    
    # sample the time from 0-duration by the fs
    time = np.linspace(1,duration,duration*fs)
    
    # create the sweeping bar stimulus in memory
    signal = chirp(time, lo_freq, duration/2, hi_freq, method='linear')
    
    # instantiate an instance of the Stimulus class
    stimulus = AuditoryStimulus(signal, tr_length, freq_window, time_window, sampling_rate=fs, 
                                tr_sampling_rate=tr_sampling_rate, scale_factor=scale_factor, clip_number=clip_number)
    
    freq_center = 550 # center frequency
    freq_sigma = 100 # frequency dispersion
    hrf_delay = 0.0 # seconds
    
    # set up bounds for the grid search
    bounds = ((lo_freq, hi_freq),(lo_freq, hi_freq),(-5,5))
    
    # initialize the strf model
    strf_model = strf.SpectrotemporalModel(stimulus)
    
    # generate the modeled BOLD response
    response = strf.compute_model_ts(freq_center, freq_sigma, hrf_delay,
                                     stimulus.time_coord, stimulus.freq_coord, stimulus.spectrogram.astype('double'),
                                     tr_length, num_timepoints, clip_number, norm_func=utils.zscore)
    
    # fit the response
    fit = strf.SpectrotemporalFit(response, strf_model, bounds, tr_length, [0,0,0], 0, True)
    f0, fs0, hrf0 = fit.strf_estimate[:]
    
    # assert
    npt.assert_almost_equal([f0, fs0, hrf0],[freq_center,freq_sigma,hrf_delay],10)
    
    
def test_parallel_fit():
    
    # stimulus features
    lo_freq = 100 # Hz
    hi_freq = 1000 # Hz
    fs = hi_freq*2 # Hz
    duration = 100 # seconds
    tr_length = 1.0 # seconds
    tr_sampling_rate = 4 # number of time-samples per TR to plot the STRF in
    time_window = 0.5 # seconds
    freq_window = 128 # this is 2x the number of freq bins we'll end up with in the spectrogram
    scale_factor = 1.0 # how much to downsample the spectrotemporal space
    num_timepoints = np.floor(duration / tr_length)
    clip_number = 10
    num_voxels = 20
    
    num_timepoints = np.floor(duration / tr_length)
    
    # sample the time from 0-duration by the fs
    time = np.linspace(1,duration,duration*fs)
    
    # create the sweeping bar stimulus in memory
    signal = chirp(time, lo_freq, duration/2, hi_freq, method='linear')
    
    # instantiate an instance of the Stimulus class
    stimulus = AuditoryStimulus(signal, tr_length, freq_window, time_window, sampling_rate=fs, 
                                tr_sampling_rate=tr_sampling_rate, clip_number=10)
    
    # set up bounds for the grid search
    indices = [(1,2,3)]*num_voxels
    bounds = [((lo_freq, hi_freq),(lo_freq, hi_freq),(-5,5))]*num_voxels
    
    # initialize the strf model
    strf_model = strf.SpectrotemporalModel(stimulus)
    
    # makeup a STRF estimate
    freq_center = 487 # center frequency
    freq_sigma = 123 # frequency dispersion
    time_center = 0.5 * stimulus.all_freqs[-1] # this won't change as we'll always convolve over each time-points time-seriesszsz
    time_sigma = 0.5 * stimulus.all_freqs[-1] # seconds
    hrf_delay = 0.0 # seconds
    degrees = 0
    
    estimates = [(freq_center,freq_sigma,hrf_delay)]*num_voxels
    
    # create the simulated time-series
    timeseries = []
    for estimate in estimates:
        response = strf.compute_model_ts(freq_center, freq_sigma, hrf_delay,
                                         stimulus.time_coord, stimulus.freq_coord, stimulus.spectrogram,
                                         tr_length, num_timepoints, clip_number, norm_func=utils.zscore)
        timeseries.append(response)
        
    # package the data structure
    dat = zip(timeseries,
              repeat(strf_model,num_voxels),
              bounds,
              repeat(tr_length,num_voxels),
              indices,
              repeat(0.20,num_voxels),
              repeat(True,num_voxels))
              
    # run analysis
    num_cpus = multiprocessing.cpu_count()-1
    pool = multiprocessing.Pool(num_cpus)
    output = pool.map(strf.parallel_fit,dat)
    pool.close()
    pool.join()
    
    # assert equivalence
    for strf_fit, estimate in zip(output, estimates):
        nt.assert_almost_equal(strf_fit.freq_center, estimate[0])
        nt.assert_almost_equal(strf_fit.freq_sigma, estimate[1])
        nt.assert_almost_equal(strf_fit.hrf_delay, estimate[2])




