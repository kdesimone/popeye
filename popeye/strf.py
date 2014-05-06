#!/usr/bin/python

""" Classes and functions for fitting population encoding models """

from __future__ import division
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.optimize import brute, fmin_powell
from scipy.special import gamma
from scipy.stats import linregress

from dipy.core.onetime import auto_attr

import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import MakeFastPrediction

def double_gamma_hrf(delay,TR):
    """
    The double-gamma hemodynamic reponse function (HRF) used to convolve with
    the stimulus time-series.
    
    The user specifies only the delay of the peak and under-shoot The delay
    shifts the peak and under-shoot by a variable number of seconds.  The other
    parameters are hard-coded.  The HRF delay is modeled for each voxel
    independently.  The double-gamme HRF andhard-coded values are based on
    previous work (Glover, 1999).
    
    
    Parameters
    ----------
    delay : float
        The delay of the HRF peak and under-shoot.
    
    TR : float
        The length of the repetition time in seconds.
        
        
    Returns
    -------
    hrf : ndarray
        The hemodynamic response function to convolve with the stimulus
        time-series.
    
    
    Reference
    ----------
    Glover, G.H. (1999) Deconvolution of impulse response in event-related BOLD.
    fMRI. NeuroImage 9: 416 429.
    
    """
    
    # add delay to the peak and undershoot params (alpha 1 and 2)
    alpha_1 = 6.0/TR+delay/TR
    beta_1 = 1.0
    c = 0.2
    alpha_2 = 16.0/TR+delay/TR
    beta_2 = 1.0
    
    t = np.arange(0,33/TR,TR)
    scale = 1
    hrf = scale*( ( ( t ** (alpha_1) * beta_1 ** alpha_1 *
                      np.exp( -beta_1 * t )) /gamma( alpha_1 )) - c *
                  ( ( t ** (alpha_2 ) * beta_2 ** alpha_2 * np.exp( -beta_2 * t ))
                      /gamma( alpha_2 ) ) )
            
    return hrf

def gaussian_2D(X, Y, x0, y0, sigma_x, sigma_y, degrees, amplitude=1):
    
    theta = degrees*np.pi/180
    
    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    
    Z = amplitude*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    
    return Z


def compute_model_ts(freq_center, freq_sigma, time_sigma, hrf_delay, 
                     time_coord, freq_coord, spectrogram,
                     tr_length, num_timepoints, time_window,
                     norm_func=utils.zscore):
                     
    # create the STRF
    g = gaussian_2D(time_coord, freq_coord, 0.5, freq_center, time_sigma, freq_sigma, 0)
    
    # initialize the inter-TR stimulus model
    stim = np.zeros(num_timepoints)
    
    # loop over each TR
    for tr in np.arange(num_timepoints-1):
        
        # initialize the intra-TR stimulus model
        tr_model = np.zeros(spectrogram.shape[0])
        
        from_slice = tr*1./time_window
        to_slice = (tr+1)*1./time_window
        
        # grab the sound frame an normalize it to 1
        sound_frame = spectrogram[:,from_slice:to_slice]
        
        # don't bother if its during a silence block
        if np.sum(sound_frame)>0:
            
            # loop over each frequency and convolve
            for f in range(len(tr_model)):
                
                f_vector = sound_frame[f,:]
                g_vector = g[f,:]
                
                conv = np.convolve(f_vector,g_vector)
                tr_model[f] = np.sum(conv[1/time_window/2:1/time_window*1.5])
                
            stim[tr] = np.mean(tr_model)
            
    hrf = double_gamma_hrf(hrf_delay, tr_length)
    model = np.convolve(stim, hrf)
    model = norm_func(model[0:len(stim)])
    
    return model

def error_function(parameters, response, 
                   time_coord, freq_coord, spectrogram,
                   tr_length, num_timepoints, time_window):
    
    # unpack the tuple
    freq_center, freq_sigma, time_sigma, hrf_delay = parameters[:]
    
    # if the frequency is out of range, abort with an inf
    if np.abs(freq_center) > np.floor(np.max(freq_coord))-1:
        return np.inf
    if np.abs(freq_sigma) > np.floor(np.max(freq_coord))-1:
        return np.inf
    
    # if the sigma_time is larger than TR, abort with an inf
    if time_sigma > np.max(time_coord):
        return np.inf
        
    # if the sigma is <= 0, abort with an inf
    if freq_sigma <= 0:
        return np.inf
    if time_sigma <= 0:
        return np.inf
        
    # if the HRF delay parameter is greater than 5 seconds, abort with an inf
    if np.abs(hrf_delay) > 5:
        return np.inf
        
    # otherwise generate a prediction
    model_ts = compute_model_ts(freq_center, freq_sigma, time_sigma, hrf_delay,
                                time_coord, freq_coord, spectrogram,
                                tr_length, num_timepoints, time_window,
                                norm_func=utils.zscore)
    
    # compute the RSS
    error = np.sum((model_ts-response)**2)
    
    # catch NaN
    if np.isnan(np.sum(error)):
        return np.inf
        
    
    print(error,(freq_center, freq_sigma, time_sigma, hrf_delay))
        
    return error

def brute_force_search(bounds, response,
                       time_coord, freq_coord, spectrogram, 
                       tr_length, num_timepoints, time_window):
    
    [freq_center, freq_sigma, time_sigma, hrf_delay], err,  _, _ =\
        brute(error_function,
              args=(response, time_coord, freq_coord, spectrogram, 
                    tr_length, num_timepoints, time_window),
              ranges=bounds,
              Ns=5,
              finish=None,
              full_output=True,
              disp=None)
    
    # return the estimates
    return freq_center, freq_sigma, time_sigma, hrf_delay


def gradient_descent_search(freq_center_0, freq_sigma_0, time_sigma_0, hrf_delay_0,
                            error_function, response,
                            time_coord, freq_coord, spectrogram,
                            tr_length, num_timepoints, time_window):
                            
                            
    [freq_center, freq_sigma, time_sigma, hrf_delay], err,  _, _, _, warnflag =\
        fmin_powell(error_function,(freq_center_0, freq_sigma_0, time_sigma_0, hrf_delay_0),
                    args=(response, time_coord, freq_coord, spectrogram,
                          tr_length, num_timepoints, time_window),
                    full_output=True,
                    disp=False)
                    
    return freq_center, freq_sigma, time_sigma, hrf_delay


class SpectrotemporalModel(PopulationModel):
    
    """
    Gaussian population receptive field model.
    """
    
    def __init__(self, stim_arr):
        
        # this is a weird notation
        PopulationModel.__init__(self, stim_arr)


class SpectrotemporalFit(PopulationFit):
    
    def __init__(self, data, model, bounds, epsilon, tr_length, voxel_index, uncorrected_rval, verbose=True):
        
        self.data = utils.zscore(data)
        self.model = model
        self.bounds = bounds
        self.epsilon = epsilon
        self.tr_length = tr_length
        self.voxel_index = voxel_index
        self.uncorrected_rval = uncorrected_rval
        self.verbose = verbose
        

    @auto_attr
    def ballpark_estimate(self):
        return brute_force_search(self.bounds, self.data,
                                  self.model.stimulus.time_coord,
                                  self.model.stimulus.freq_coord,
                                  self.model.stimulus.spectrogram,
                                  self.model.stimulus.tr_length,
                                  self.model.stimulus.num_timepoints,
                                  self.model.stimulus.time_window)
    
    @auto_attr
    def strf_estimate(self):
        return gradient_descent_search(self.f0, self.fs0, self.ts0, self.hrf0,
                                       error_function, self.data, 
                                       self.model.stimulus.time_coord,
                                       self.model.stimulus.freq_coord,
                                       self.model.stimulus.spectrogram,
                                       self.model.stimulus.tr_length,
                                       self.model.stimulus.num_timepoints,
                                       self.model.stimulus.time_window)
    
    @auto_attr
    def f0(self):
        return self.ballpark_estimate[0]
    
    @auto_attr
    def fs0(self):
        return self.ballpark_estimate[1]
    
    @auto_attr
    def ts0(self):
        return self.ballpark_estimate[2]

    @auto_attr
    def hrf0(self):
        return self.ballpark_estimate[3]
                                                                         