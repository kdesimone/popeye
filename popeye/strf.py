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

import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_strf_timeseries

def recast_estimation_results(output, grid_parent, write=True):
    """
    Recasts the output of the pRF estimation into two nifti_gz volumes.
    
    Takes `output`, a list of multiprocessing.Queue objects containing the
    output of the pRF estimation for each voxel.  The pRF estimates are
    expressed in both polar and Cartesian coordinates.  If the default value
    for the `write` parameter is set to False, then the function returns the
    arrays without writing the nifti files to disk.  Otherwise, if `write` is
    True, then the two nifti files are written to disk.
    
    Each voxel contains the following metrics: 
    
        0 x / polar angle
        1 y / eccentricity
        2 sigma
        3 HRF delay
        4 RSS error of the model fit
        5 correlation of the model fit
        
    Parameters
    ----------
    output : list
        A list of PopulationFit objects.
    grid_parent : nibabel object
        A nibabel object to use as the geometric basis for the statmap.  
        The grid_parent (x,y,z) dim and pixdim will be used.
        
    Returns
    ------ 
    cartes_filename : string
        The absolute path of the recasted pRF estimation output in Cartesian
        coordinates. 
    plar_filename : string
        The absolute path of the recasted pRF estimation output in polar
        coordinates. 
        
    """
    
    
    # load the gridParent
    dims = list(grid_parent.shape)
    dims = dims[0:3]
    dims.append(4)
    
    # initialize the statmaps
    nii_out = np.zeros(dims)
    
    # extract the pRF model estimates from the results queue output
    for fit in output:
        
        if fit.__dict__.has_key('fit_stats'):
            
            nii_out[fit.voxel_index] = (fit.center_freq, 
                                        fit.bandwidth,
                                        fit.rss,
                                        fit.fit_stats[2])
                                 
    # get header information from the gridParent and update for the pRF volume
    aff = grid_parent.get_affine()
    hdr = grid_parent.get_header()
    hdr.set_data_shape(dims)
    
    # recast as nifti
    nif = nibabel.Nifti1Image(nii_out,aff,header=hdr)
    nif.set_data_dtype('float32')
    
    return nif

# this is actually not used, but it serves as the model for the cython function ...
def gaussian_2D(X, Y, x0, y0, sigma_x, sigma_y, degrees, amplitude=1):
    
    theta = degrees*np.pi/180
        
    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    
    Z = amplitude*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    
    return Z

def gaussian_1D(freqs, center_freq, sd, integrator=trapz):
    
    gaussian = np.exp(-0.5 * ((freqs - (center_freq+1))/sd)**2) / (np.sqrt(2*np.pi) * sd)
    
    return gaussian
    
def compute_model_ts_1D(center_freq, sd,
                        times, freqs, spectrogram,
                        tr_length, convolve=True):
    
    # invoke the gaussian
    # gaussian = gaussian_1D(freqs, center_freq, sd)
    
    # now create the stimulus time-series
    # stim = np.sum(gaussian[:,np.newaxis] * spectrogram,axis=0)
    
    stim = generate_strf_timeseries(freqs, spectrogram, center_freq, sd)
    
    # recast the stimulus into a time-series that i can decimate
    old_time = np.linspace(0,np.ceil(times[-1]),len(stim))
    new_time = np.linspace(0,np.ceil(times[-1]),np.ceil(times[-1])*10)
    f = interp1d(old_time,stim)
    new_stim = f(new_time)
    
    # hard-set the hrf_delay
    hrf_delay = 0
    
    # convolve it with the HRF
    hrf = utils.double_gamma_hrf(hrf_delay, tr_length, 10)
    model = fftconvolve(new_stim, hrf)[0:len(new_stim)] 
    
    # for debugging
    if convolve:
        return utils.zscore(model)
    else:
        return utils.zscore(new_stim)

# this method is used to simply multiprocessing.Pool interactions
def parallel_fit(args):
    
    # unpackage the arguments
    model = args[0]
    data = args[1]
    search_bounds = args[2]
    fit_bounds = args[3]
    tr_length = args[4]
    voxel_index = args[5]
    auto_fit = args[6]
    verbose = args[7]
    
    # fit the data
    fit = SpectrotemporalFit(model,
                             data,
                             search_bounds,
                             fit_bounds,
                             tr_length,
                             voxel_index,
                             auto_fit,
                             verbose)
    return fit


class SpectrotemporalModel(PopulationModel):
    
    """
    Gaussian population receptive field model.
    """
    
    def __init__(self, stim_arr):
        
        # this is a weird notation
        PopulationModel.__init__(self, stim_arr)


class SpectrotemporalFit(PopulationFit):
    
    def __init__(self, model, data,
                 search_bounds, fit_bounds, tr_length,
                 voxel_index=None, auto_fit=True, verbose=True):
        
        self.model = model
        self.data = data
        self.search_bounds = search_bounds
        self.fit_bounds = fit_bounds
        self.tr_length = tr_length
        self.voxel_index = voxel_index
        self.auto_fit = auto_fit
        self.verbose = verbose
        
        # just make sure that all data is inside the mask
        if self.auto_fit:
            tic = time.clock()
            self.ballpark_estimate;
            self.estimate;
            self.fit_stats;
            self.rss;
            toc = time.clock()
                
            # print to screen if verbose
            if self.verbose and ~np.isnan(self.rss) and ~np.isinf(self.rss):
                
                # we need a voxel index for the print
                if self.voxel_index is None:
                    self.voxel_index = (0,0,0)
                
                # print
                print("VOXEL=(%.03d,%.03d,%.03d)  TIME=%.03d  RVAL=%.02f  CENTER=%.05d  SIGMA=%.05d" 
                      %(self.voxel_index[0],
                        self.voxel_index[1],
                        self.voxel_index[2],
                        int(toc-tic),
                        self.fit_stats[2],
                        self.center_freq,
                        self.bandwidth))
    
    @auto_attr
    def ballpark_estimate(self):
        return utils.brute_force_search((self.model.stimulus.times,
                                         self.model.stimulus.freqs,
                                         self.model.stimulus.spectrogram,
                                         self.model.stimulus.tr_length),
                                        self.search_bounds,
                                        self.fit_bounds,
                                        self.data,
                                        utils.error_function,
                                        compute_model_ts_1D)
                                        
    @auto_attr
    def estimate(self):
        return utils.gradient_descent_search((self.f0, self.bw0),
                                             (self.model.stimulus.times,
                                              self.model.stimulus.freqs,
                                              self.model.stimulus.spectrogram,
                                              self.model.stimulus.tr_length),
                                             self.fit_bounds,
                                             self.data,
                                             utils.error_function,
                                             compute_model_ts_1D)
                                       
    @auto_attr
    def f0(self):
        return self.ballpark_estimate[0]
    
    @auto_attr
    def bw0(self):
        return self.ballpark_estimate[1]
    
    @auto_attr
    def beta0(self):
        return self.ballpark_estimate[2]
    
    # @auto_attr
    # def hrf0(self):
    #     return self.ballpark_estimate[2]
    
    @auto_attr
    def center_freq(self):
        return self.estimate[0]
        
    @auto_attr
    def bandwidth(self):
        return self.estimate[1]
    
    @auto_attr
    def beta(self):
        return self.estimate[2]
        
    # @auto_attr
    # def hrf_delay(self):
    #     return self.estimate[2]
        
    @auto_attr
    def prediction(self):
        return compute_model_ts_1D(self.center_freq, self.bandwidth,
                                   self.model.stimulus.times,
                                   self.model.stimulus.freqs,
                                   self.model.stimulus.spectrogram,
                                   self.model.stimulus.tr_length)

    @auto_attr
    def stim_timeseries(self):
        return compute_model_ts_1D(self.center_freq, self.bandwidth,
                                  self.model.stimulus.times,
                                  self.model.stimulus.freqs,
                                  self.model.stimulus.spectrogram,
                                  self.model.stimulus.tr_length, False)
                                                                  
    @auto_attr
    def fit_stats(self):
        return linregress(self.data, self.prediction)
        
    @auto_attr
    def rss(self):
        return np.sum((self.data - self.prediction)**2)
    
    @auto_attr
    def receptive_field(self):
        return gaussian_1D(self.model.stimulus.freqs, self.center_freq, self.bandwidth)
        
                                    
