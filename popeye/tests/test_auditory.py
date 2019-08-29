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
    duration = 30 # seconds
    Fs = int(44100/2) # Hz
    lo_freq = 200.0 # Hz
    hi_freq = 10000.0 # Hz
    tr_length = 1.0 # seconds
    clip_number = 0 # TRs
    dtype = ctypes.c_double
    
    # fit settings
    auto_fit = True
    verbose = 1
    debug = False
    Ns = 10
    
    # generate auditory stimulus
    time = np.linspace(0,duration,duration*Fs)
    ch = chirp(time, lo_freq, duration, hi_freq, method='logarithmic')
    signal = np.tile(np.concatenate((ch,ch[::-1])),5)
    blank = np.zeros((30*Fs))
    signal = np.concatenate((blank,signal,blank),-1)
    
    # instantiate an instance of the Stimulus class
    stimulus = AuditoryStimulus(signal, Fs, tr_length, dtype)  ### stimulus
    
    # initialize the gaussian model
    model = aud.AuditoryModel(stimulus, utils.spm_hrf)  ### model 
    model.hrf_delay = 0
    
    # invent pRF estimate
    center_freq_hz = 987
    sigma_hz = 123
    center_freq = np.log10(center_freq_hz)
    sigma = np.log10(sigma_hz)
    beta = 2.4
    baseline = 0.59
    
    # generate data
    data = model.generate_prediction(center_freq, sigma, beta, baseline)
    
    # search grids
    c_grid = utils.grid_slice(np.log10(300), np.log10(1000), Ns)
    s_grid = utils.grid_slice(np.log10(100), np.log10(500), Ns)
    grids = (c_grid, s_grid,)
    
    # search bounds
    c_bound = (np.log10(lo_freq), np.log10(hi_freq))
    s_bound = (np.log10(50), np.log10(hi_freq))
    b_bound = (1e-8, None)
    m_bound = (None,None)
    bounds = (c_bound, s_bound, b_bound, m_bound)
    
    # fit it
    fit = aud.AuditoryFit(model, data, grids, bounds, Ns=Ns)
    
    # grid fit
    npt.assert_almost_equal(fit.center_freq0, 3)
    npt.assert_almost_equal(fit.sigma0, 2.1553266676302263)
    npt.assert_almost_equal(fit.center_freq_hz, center_freq_hz)
    # the baseline/beta should be 0/1 when regressed data vs. estimate
    (m,b) = np.polyfit(fit.scaled_ballpark_prediction, data, 1)
    npt.assert_almost_equal(m, 1.0)
    npt.assert_almost_equal(b, 0.0)

    
    # final fit
    npt.assert_almost_equal(fit.center_freq, center_freq)
    npt.assert_almost_equal(fit.sigma, sigma)
    npt.assert_almost_equal(fit.beta, beta)
    npt.assert_almost_equal(fit.baseline, baseline)
    
    # test receptive field
    rf = np.exp(-((10**fit.model.stimulus.freqs-10**fit.center_freq)**2)/(2*(10**fit.sigma)**2))
    rf /= (10**fit.sigma*np.sqrt(2*np.pi))
    npt.assert_almost_equal(np.round(rf.sum()), np.round(fit.receptive_field.sum())) 
    
    # test model == fit RF
    rf = np.exp(-((fit.model.stimulus.freqs-fit.center_freq)**2)/(2*fit.sigma**2))
    rf /= (fit.sigma*np.sqrt(2*np.pi))
    npt.assert_almost_equal(np.round(rf.sum()), np.round(fit.receptive_field_log10.sum())) 
    
    
