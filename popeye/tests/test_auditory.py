import ctypes

import matplotlib
from matplotlib.pyplot import specgram

import numpy as np
import numpy.testing as npt
from scipy.signal import chirp

import popeye.utilities as utils
import popeye.auditory as aud
from popeye.auditory_stimulus import AuditoryStimulus

def test_auditory_fit():

    # stimulus features
    lo_freq = 200 # Hz
    hi_freq = 12000 # Hz
    Fs = hi_freq*2 # Hz
    duration = 30 # seconds
    blank = 10 # seconds
    tr_length = 1.25 # seconds
    noverlap = 0.5 # seconds
    NFFT = 1024 # this is 2x the number of freq bins we'll end up with in the spectrogram
    voxel_index = (1,2,3)
    Ns = 5
    auto_fit = True
    verbose = 0
    resample_factor = 10
    dtype = ctypes.c_double

    # sample the time from 0-duration by the fs
    time = np.linspace(0,duration,duration*Fs)

    # create a chirp stimulus
    ch = chirp(time, lo_freq, duration, hi_freq, method='quadratic')
    signal = np.concatenate((ch,ch[::-1]))

    # instantiate an instance of the Stimulus class
    stimulus = AuditoryStimulus(signal, NFFT, Fs, noverlap, resample_factor, dtype, tr_length)

    # initialize the gaussian model
    model = aud.AuditoryModel(stimulus, utils.double_gamma_hrf)

    # generate a random pRF estimate
    center_freq = 8910.0
    sigma = 567.0
    beta = 0.88
    baseline = -1.5
    hrf_delay = -0.25

    # create the "data"
    data = model.generate_prediction(center_freq, sigma, beta, baseline, hrf_delay)

    # set search grid
    c_grid = (lo_freq,hi_freq)
    s_grid = (lo_freq,hi_freq)
    b_grid = (0.1,1)
    bl_grid = (-3,3)
    h_grid = (-3.0,3.0)

    # set search bounds
    c_bound = (lo_freq,hi_freq)
    s_bound = (lo_freq,hi_freq)
    b_bound = (1e-8,None)
    bl_bound = (None,None)
    h_bound = (-4.0,4.0)

    # loop over each voxel and set up a GaussianFit object
    grids = (c_grid, s_grid, b_grid, bl_grid, h_grid,)
    bounds = (c_bound, s_bound, b_bound, bl_bound, h_bound,)

    # fit the response
    fit = aud.AuditoryFit(model, data, grids, bounds, Ns=Ns)

    # assert equivalence
    npt.assert_equal(np.round(fit.center_freq), center_freq)
    npt.assert_equal(np.round(fit.sigma), sigma)
    npt.assert_almost_equal(fit.beta, beta, 2)
    npt.assert_almost_equal(fit.baseline, baseline, 2)
    npt.assert_almost_equal(fit.hrf_delay, hrf_delay, 2)

    npt.assert_almost_equal((fit.center_freq0,fit.sigma0,fit.beta0,fit.baseline0,fit.hrf0),
                             (9050.0, 200.0, 0.32500000000000001, -1.5, 1.5))

# def test_auditory_data_upsample():
#
#     lo_freq = 200 # Hz
#     hi_freq = 12000 # Hz
#     Fs = hi_freq*2 # Hz
#     duration = 30 # seconds
#     blank = 10 # seconds
#     tr_length = 1.25 # seconds
#     noverlap = 0.5 # seconds
#     NFFT = 1024 # this is 2x the number of freq bins we'll end up with in the spectrogram
#     voxel_index = (1,2,3)
#     Ns = 5
#     auto_fit = True
#     verbose = 0
#     resample_factor = 10
#     dtype = ctypes.c_double
#
#     # sample the time from 0-duration by the fs
#     time = np.linspace(0,duration,duration*Fs)
#
#     # create a chirp stimulus
#     ch = chirp(time, lo_freq, duration, hi_freq, method='quadratic')
#     signal = np.concatenate((ch,ch[::-1]))
#
#     # instantiate an instance of the Stimulus class
#     stimulus = AuditoryStimulus(signal, NFFT, Fs, noverlap, resample_factor, dtype, tr_length)
#
#     # initialize the gaussian model
#     model = aud.AuditoryModel(stimulus, utils.double_gamma_hrf)
#
#     # generate a random pRF estimate
#     center_freq = 5910.0
#     sigma = 567.0
#     beta = 0.88
#     baseline = -1.5
#     hrf_delay = -0.25
#
#     # create the "data"
#     data = model.generate_prediction(center_freq, sigma, beta, baseline, hrf_delay)
#     data = data[np.arange(0,len(data),2)]
#
#     # set search grid
#     c_grid = (lo_freq,hi_freq)
#     s_grid = (lo_freq,hi_freq)
#     b_grid = (0.1,1)
#     bl_grid = (-3,3)
#     h_grid = (-3.0,3.0)
#
#     # set search bounds
#     c_bound = (lo_freq,hi_freq)
#     s_bound = (lo_freq,hi_freq)
#     b_bound = (1e-8,None)
#     bl_bound = (None,None)
#     h_bound = (-4.0,4.0)
#
#     # loop over each voxel and set up a GaussianFit object
#     grids = (c_grid, s_grid, b_grid, bl_grid, h_grid,)
#     bounds = (c_bound, s_bound, b_bound, bl_bound, h_bound,)
#
#     # fit the response
#     fit = aud.AuditoryFit(model, data, grids, bounds, Ns=Ns)
#
#     npt.assert_equal(len(fit.data), len(data))
