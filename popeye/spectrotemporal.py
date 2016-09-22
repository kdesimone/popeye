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
from scipy.ndimage.filters import median_filter
from scipy.integrate import romb, trapz
from scipy.interpolate import interp1d
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_rf_timeseries_1D

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

# this is actually not used, but it serves as the model for the cython function ...
def gaussian_2D(X, Y, x0, y0, sigma_x, sigma_y, degrees, amplitude=1):
    
    theta = degrees*np.pi/180
        
    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    
    Z = amplitude*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    
    return Z

def gaussian_1D(freqs, center_freq, sigma):
    
    gaussian = np.exp(-0.5 * ((freqs - center_freq)/sigma)**2)
    
    return gaussian

def compute_model_ts(center_freq, sigma,
                     spectrogram, freqs, target_times):
    
    # generate stimulus time-series
    rf = gaussian_1D(freqs, center_freq, sigma)
    # make sure sigma isn't too big
    # if np.any(np.round(rf[0],3) > 0):
    #     return np.inf
    # if np.any(np.round(rf[-1],3) > 0):
    #     return np.inf
    
    # create mask for speed
    distance = freqs - center_freq
    mask = np.zeros_like(distance, dtype='uint8')
    mask[distance < (5*sigma)] = 1
        
    # extract the response
    stim = generate_rf_timeseries_1D(spectrogram,rf,mask)
    
    # recast the stimulus into a time-series that i can 
    source_times = np.linspace(0,target_times[-1],len(stim),endpoint=True)
    f = interp1d(source_times,stim,kind='linear')
    new_stim = f(target_times)
    
    # hard-set the hrf_delay
    hrf_delay = 0
    
    # convolve it with the HRF
    hrf = utils.double_gamma_hrf(hrf_delay, 1.0, 10)
    stim_pad = np.tile(new_stim,3)
    model = fftconvolve(stim_pad, hrf,'same')[len(new_stim):len(new_stim)*2]
    
    # normalize it
    model = utils.zscore(model)
    
    return model

# this method is used to simply multiprocessing.Pool interactions
def parallel_fit(args):
    
    # unpackage the arguments
    model = args[0]
    data = args[1]
    grids = args[2]
    bounds = args[3]
    Ns = args[4]
    tr_length = args[5]
    voxel_index = args[6]
    auto_fit = args[7]
    verbose = args[8]
    
    # fit the data
    fit = SpectroTemporalFit(model,
                             data,
                             grids,
                             bounds,
                             Ns,
                             tr_length,
                             voxel_index,
                             auto_fit,
                             verbose)
    return fit


class SpectroTemporalModel(PopulationModel):
    
    """
    Gaussian population receptive field model.
    """
    
    def __init__(self, stimulus):
        
        # this is a weird notation
        PopulationModel.__init__(self, stimulus)


class SpectroTemporalFit(PopulationFit):
    
    def __init__(self, model, data, grids, bounds, Ns, tr_length,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
        
        
        PopulationFit.__init__(self, model, data, grids, bounds, Ns, tr_length, voxel_index, auto_fit, verbose)
        
        if self.auto_fit:
            
            self.start = time.clock()
            try:
                self.ballpark;
                self.estimate;
                self.OLS;
                self.finish = time.clock()
                
                if self.verbose:
                    print(self.msg)
                    
            except:
                self.finish = time.clock()
                if self.verbose:
                    print(self.errmsg)
    
    @auto_attr
    def ballpark(self):
        return utils.brute_force_search((self.model.stimulus.spectrogram,
                                         self.model.stimulus.freqs,
                                         self.model.stimulus.target_times),
                                        self.grids,
                                        self.bounds,
                                        self.Ns,
                                        self.data,
                                        utils.error_function,
                                        compute_model_ts,
                                        self.very_verbose)
                                        
    @auto_attr
    def estimate(self):
        return utils.gradient_descent_search((self.center_freq0, self.sigma0),
                                             (self.model.stimulus.spectrogram,
                                              self.model.stimulus.freqs,
                                              self.model.stimulus.target_times),
                                             self.bounds,
                                             self.data,
                                             utils.error_function,
                                             compute_model_ts,
                                             self.very_verbose)
                                             
    @auto_attr
    def center_freq0(self):
        return self.ballpark[0]
    
    @auto_attr
    def sigma0(self):
        return self.ballpark[1]
    
    # @auto_attr
    # def beta0(self):
    #     return self.ballpark[2]
    # 
    # @auto_attr
    # def baseline0(self):
    #     return self.ballpark[3]
    
    @auto_attr
    def center_freq(self):
        return self.estimate[0]
        
    @auto_attr
    def sigma(self):
        return self.estimate[1]
    
    # @auto_attr
    # def beta(self):
    #     return self.estimate[2]
    # 
    # 
    # @auto_attr
    # def baseline(self):
    #     return self.estimate[3]
    
    @auto_attr
    def prediction(self):
        return compute_model_ts(self.center_freq, self.sigma,
                                self.model.stimulus.spectrogram,
                                self.model.stimulus.freqs,
                                self.model.stimulus.target_times)
    
    @auto_attr
    def OLS(self):
        return sm.OLS(self.data,self.prediction).fit()
    
    @auto_attr
    def coefficient(self):
        return self.OLS.params[0]
    
    @auto_attr
    def rsquared(self):
        return self.OLS.rsquared
    
    @auto_attr
    def stderr(self):
        return np.sqrt(self.OLS.mse_resid)
    
    @auto_attr
    def rss(self):
        return np.sum((self.data - self.prediction)**2)
    
    @auto_attr
    def receptive_field(self):
        return gaussian_1D(self.model.stimulus.freqs, self.center_freq,)
    
    @auto_attr
    def msg(self):
        txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RSQ=%.02f  CENTER=%.02f   SIGMA=%.02f" 
            %(self.voxel_index[0],
              self.voxel_index[1],
              self.voxel_index[2],
              self.finish-self.start,
              self.rsquared,
              self.center_freq,
              self.sigma))
        return txt
    
    @auto_attr
    def errmsg(self):
        txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   There was a problem with this voxel!" 
            %(self.voxel_index[0],
              self.voxel_index[1],
              self.voxel_index[2],
              self.finish-self.start))
        return txt
        
                                    
