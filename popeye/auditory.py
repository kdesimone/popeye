#!/usr/bin/python

""" Classes and functions for fitting population encoding models """

from __future__ import division
import time
import warnings
import gc
warnings.simplefilter("ignore")

import numpy as np
from scipy.optimize import brute, fmin_powell
from scipy.special import gamma
from scipy.stats import linregress
from scipy.signal import fftconvolve, decimate
from scipy.misc import imresize
from scipy.ndimage.filters import median_filter
from scipy.integrate import romb, trapz
from scipy.interpolate import interp1d
import statsmodels.api as sm
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_rf_timeseries_1D, generate_strf_timeseries

def recast_estimation_results(output, grid_parent, write=True):
    
    # load the gridParent
    dims = list(grid_parent.shape)
    dims = dims[0:3]
    dims.append(5)
    
    # initialize the statmaps
    nii_out = np.zeros(dims)
    
    # extract the pRF model estimates from the results queue output
    for fit in output:
        if fit.__dict__.has_key('OLS'):
            nii_out[fit.voxel_index] = (fit.center_freq, 
                                        fit.sigma,
                                        fit.rsquared,
                                        fit.coefficient,
                                        fit.stderr)
                                 
    # get header information from the gridParent and update for the pRF volume
    aff = grid_parent.get_affine()
    hdr = grid_parent.get_header()
    hdr.set_data_shape(dims)
    
    # recast as nifti
    nif = nibabel.Nifti1Image(nii_out,aff,header=hdr)
    
    return nif

class AuditoryModel(PopulationModel):
    
    """
    Gaussian population receptive field model.
    """
    
    def __init__(self, stimulus, hrf_model):
        
        # this is a weird notation
        PopulationModel.__init__(self, stimulus, hrf_model)
    
    def generate_prediction(self, center_freq, sigma, beta, hrf_delay):
        
        # generate stimulus time-series
        rf = np.exp(-((self.stimulus.freqs-center_freq)**2)/(2*sigma**2))
        rf /= (sigma*np.sqrt(2*np.pi))
        rf *= beta
        
        # if the rf runs off the coords
        if np.round(rf[0],3) != 0:
            return np.inf
            
        # create mask for speed
        distance = self.stimulus.freqs - center_freq
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (5*sigma)] = 1
        
        # extract the response
        response = generate_rf_timeseries_1D(self.stimulus.spectrogram, rf, mask)
        
        # resample the time-series to the BOLD 
        source_times = np.linspace(0, 1, len(response), endpoint=True)
        f = interp1d(source_times, response, kind='linear')
        num_volumes = self.stimulus.stim_arr.shape[0]/self.stimulus.Fs/self.stimulus.tr_length
        target_times = np.linspace(0, 1, num_volumes, endpoint=True)
        resampled_response = f(target_times)
        
        # generate the HRF
        hrf = utils.double_gamma_hrf(hrf_delay, self.stimulus.tr_length)
        
        # pad and convolve
        model = fftconvolve(resampled_response, hrf, 'same') / len(resampled_response)
        
        return model
    
    def generate_ballpark_prediction(self, center_freq, sigma, beta, hrf_delay):
        
        return self.generate_prediction(center_freq, sigma, beta, hrf_delay)

class AuditoryFit(PopulationFit):
    
    def __init__(self, model, data, grids, bounds, Ns,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
        
        
        PopulationFit.__init__(self, model, data, grids, bounds, Ns, voxel_index, auto_fit, verbose)
        
    @auto_attr
    def center_freq0(self):
        return self.ballpark[0]
    
    @auto_attr
    def sigma0(self):
        return self.ballpark[1]
    
    @auto_attr
    def beta0(self):
        return self.ballpark[2]
    
    @auto_attr
    def hrf0(self):
        return self.ballpark[3]
    
    @auto_attr
    def center_freq(self):
        return self.estimate[0]
        
    @auto_attr
    def sigma(self):
        return self.estimate[1]
    
    @auto_attr
    def beta(self):
        return self.estimate[2]
    
    @auto_attr
    def hrf_delay(self):
        return self.estimate[3]
    
    
    @auto_attr
    def msg(self):
        txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RSQ=%.02f  CENTER=%.02d   SIGMA=%.02f   BETA=%.08f   HRF=%.02f" 
            %(self.voxel_index[0],
              self.voxel_index[1],
              self.voxel_index[2],
              self.finish-self.start,
              self.rsquared,
              self.center_freq,
              self.sigma,
              self.beta,
              self.hrf_delay))
        return txt
