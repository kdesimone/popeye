#!/usr/bin/python

""" Classes and functions for fitting population encoding models """

from __future__ import division, print_function, absolute_import
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np

from scipy.stats import linregress
import scipy.signal as ss


from popeye.onetime import auto_attr

import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import MakeFastGaussPrediction

def compute_model_ts(x, y, sigma, hrf_delay, 
                     deg_x, deg_y, stim_arr, 
                     tr_length, frames_per_tr):
    
    # otherwise generate a prediction
    ts_stim = MakeFastGaussPrediction(deg_x,
                                      deg_y,
                                      stim_arr,
                                      x,
                                      y,
                                      sigma)
    
    # convolve it
    hrf = utils.double_gamma_hrf(hrf_delay, tr_length, frames_per_tr)
    
    # normalize it
    model = utils.zscore(ss.fftconvolve(ts_stim, hrf)[0:len(ts_stim)])
    
    # decimate it
    if frames_per_tr > 1:
        model = ss.decimate(model, int(frames_per_tr), 1)
        
    return model

# this method is used to simply multiprocessing.Pool interactions
def parallel_fit(args):
    
    # unpackage the arguments
    response = args[0]
    model = args[1]
    search_bounds = args[2]
    fit_bounds = args[3]
    tr_length = args[4]
    voxel_index = args[5]
    uncorrected_rval = args[6]
    verbose = args[7]
    
    # fit the data
    fit = GaussianFit(response,
                      model,
                      search_bounds,
                      fit_bounds,
                      tr_length,
                      voxel_index,
                      uncorrected_rval,
                      verbose)
    return fit


class GaussianModel(PopulationModel):
    
    """
    Gaussian population receptive field model.
    """
    
    def __init__(self, stimulus):
        
        # this is a weird notation
        PopulationModel.__init__(self, stimulus)
        
class GaussianFit(object):
    
    """
    Gaussian population receptive field model fitting
    """
    
    def __init__(self, data, model, search_bounds, fit_bounds, tr_length, voxel_index, uncorrected_rval, verbose=True, auto_fit=True):
            
        self.data = utils.zscore(data)
        self.model = model
        self.search_bounds = search_bounds
        self.fit_bounds = fit_bounds
        self.tr_length = tr_length
        self.voxel_index = voxel_index
        self.uncorrected_rval = uncorrected_rval
        self.verbose = verbose
        
        if auto_fit:
            tic = time.clock()
            self.ballpark_estimate;
            self.gaussian_estimate;
            toc = time.clock()
            
            # print to screen if verbose
            if self.verbose:
                print("VOXEL=(%.03d,%.03d,%.03d)  TIME=%.03d  ERROR=%.03d  RVAL=%.02f" 
                      %(self.voxel_index[0],
                        self.voxel_index[1],
                        self.voxel_index[2],
                        toc-tic,
                        self.rss,
                        self.fit_stats[2]))
                    
        
    @auto_attr
    def ballpark_estimate(self):
        return utils.brute_force_search((self.model.stimulus.deg_x_coarse,
                                         self.model.stimulus.deg_y_coarse,
                                         self.model.stimulus.stim_arr_coarse,
                                         self.tr_length,
                                         self.model.stimulus.frames_per_tr),
                                        self.search_bounds,
                                        self.fit_bounds,
                                        self.data,
                                        utils.error_function,
                                        compute_model_ts)
                                  
                                  
    @auto_attr
    def gaussian_estimate(self):
        return utils.gradient_descent_search((self.x0, self.y0, self.s0, self.hrf0),
                                             (self.model.stimulus.deg_x,
                                              self.model.stimulus.deg_y,
                                              self.model.stimulus.stim_arr,
                                              self.tr_length,
                                              self.model.stimulus.frames_per_tr),
                                             self.fit_bounds,
                                             self.data,
                                             utils.error_function,
                                             compute_model_ts)
 
                                       
    @auto_attr
    def x0(self):
        return self.ballpark_estimate[0]
        
    @auto_attr
    def y0(self):
        return self.ballpark_estimate[1]
        
    @auto_attr
    def s0(self):
        return self.ballpark_estimate[2]
        
    @auto_attr
    def hrf0(self):
        return self.ballpark_estimate[3]
        
    @auto_attr
    def x(self):
        return self.gaussian_estimate[0]
        
    @auto_attr
    def y(self):
        return self.gaussian_estimate[1]
        
    @auto_attr
    def sigma(self):
        return self.gaussian_estimate[2]
        
    @auto_attr
    def hrf_delay(self):
        return self.gaussian_estimate[3]
        
    @auto_attr
    def model_ts(self):
        return compute_model_ts(self.x, self.y, self.sigma, self.hrf_delay,
                                self.model.stimulus.deg_x,
                                self.model.stimulus.deg_y,
                                self.model.stimulus.stim_arr,
                                self.tr_length,
                                self.model.stimulus.frames_per_tr)
    
    @auto_attr
    def fit_stats(self):
        return linregress(self.data, self.model_ts)
    
    @auto_attr
    def rss(self):
        return np.sum((self.data - self.model_ts)**2)
