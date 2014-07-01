#!/usr/bin/python

""" Classes and functions for fitting population encoding models """

from __future__ import division, print_function, absolute_import
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.optimize import brute, fmin_powell
from scipy.stats import linregress
from scipy.interpolate import interp1d
import scipy.signal as ss

from popeye.onetime import auto_attr

import popeye.utilities as utils
from popeye import gaussian
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import MakeFastGaborPrediction


def compute_model_ts(x, y, sigma, hrf_delay, theta, phi, cpd,
                     deg_x, deg_y, stim_arr, tr_length, 
                     frames_per_tr, norm_func=utils.zscore):
    
    # otherwise generate a prediction
    ts_stim = MakeFastGaborPrediction(deg_x,
                                      deg_y,
                                      stim_arr,
                                      x,
                                      y,
                                      sigma,
                                      theta,
                                      phi,
                                      cpd)
    
    # convolve it
    hrf = utils.double_gamma_hrf(hrf_delay, tr_length, frames_per_tr)
    
    # normalize it
    model = norm_func(ss.fftconvolve(ts_stim, hrf)[0:len(ts_stim)])

    # decimate it
    model = ss.decimate(model, int(frames_per_tr), 1)
    
    return model

def error_function(parameters, response, deg_x, deg_y, 
                  stim_arr, tr_length, frames_per_tr, ppd):

    # unpack the tuple
    x, y, sigma, hrf_delay, theta, phi, cpd = parameters[:]
                  
    # if the theta and phi are not between 0 and 360, wrap the parameters around
    if theta < 0.:
        theta += 360
    if theta > 360:
        theta -= 360
    if phi < 0:
        phi += 360
    if phi > 360:
        phi -= 360
    
    visuotopic_limiter = 12 #np.floor(np.min((deg_x.max(),deg_y.max())))
    
    # if the x or y are off the screen, abort with an inf
    if np.abs(x) > visuotopic_limiter:
        return np.inf
    if np.abs(y) > visuotopic_limiter:
        return np.inf
        
    # if the sigma is larger than the screen width, abort with an inf
    if np.abs(sigma) > visuotopic_limiter:
        return np.inf
        
    # if the sigma is <= 0, abort with an inf
    if sigma < 1./ppd:
        return np.inf
        
    # if the HRF delay parameter is greater than 4 seconds, abort with an inf
    if np.abs(hrf_delay) > 5:
        return np.inf
    
    # if cpd is below 0, abort with inf
    if cpd < 0:
        return np.inf
    if cpd > ppd/2:
        return np.inf
    
    
    # otherwise generate a prediction
    model_ts = compute_model_ts(x, y, sigma, hrf_delay, theta, phi, cpd,
                                deg_x, deg_y, stim_arr, tr_length, frames_per_tr)
    
    # compute the RSS
    error = np.sum((model_ts-response)**2)
    
    # catch NaN
    if np.isnan(np.sum(error)):
        return np.inf
    
    txt = '%.03g,%.03g,%.03g,%.03g,%.03g,%.03g,%.03g,%.03g' %(error,x,y,sigma,hrf_delay,theta,phi,cpd)
    # print(txt)
    return error

def brute_force_search(bounds, response, error_function,
                       deg_x, deg_y, stim_arr, 
                       tr_length, frames_per_tr, ppd):

    [x0, y0, s0, hrf0, theta0, phi0, cpd0], err,  _, _ =\
        brute(error_function,
              args=(response, deg_x, deg_y, stim_arr, tr_length, frames_per_tr, ppd),
              ranges=bounds,
              Ns=4,
              finish=None,
              full_output=True,
              disp=False)

    # return the estimates
    return x0, y0, s0, hrf0, theta0, phi0, cpd0
    
def gradient_descent_search(x0, y0, s0, hrf0, theta0, phi0, cpd0,
                            error_function, response, deg_x, deg_y, 
                            stim_arr, tr_length, frames_per_tr, ppd):

    [x, y, sigma, hrf_delay, theta, phi, cpd], err,  _, _, _, warnflag =\
        fmin_powell(error_function,(x0, y0, s0, hrf0, theta0, phi0, cpd0),
                    args=(response, deg_x, deg_y, stim_arr, tr_length, frames_per_tr, ppd),
                    full_output=True,
                    disp=False)

    return x, y, sigma, hrf_delay, theta, phi, cpd

def error_function_lazy(parameters, response, deg_x, deg_y, stim_arr, 
                        tr_length, frames_per_tr, ppd,
                        x, y, sigma, hrf_delay):
    
    # unpack the tuple
    theta, phi, cpd = parameters[:]
    
    # if the theta and phi are not between 0 and 360, wrap the parameters around
    if theta < 0.:
        theta += 360
    if theta > 360:
        theta -= 360
    if phi < 0:
        phi += 360
    if phi > 360:
        phi -= 360
        
    # if cpd is below 0, abort with inf
    if cpd < 0:
        return np.inf
    if cpd > ppd/2:
        return np.inf
            
    # otherwise generate a prediction
    model_ts = compute_model_ts(x, y, sigma, hrf_delay, theta, phi, cpd,
                                deg_x, deg_y, stim_arr, tr_length, frames_per_tr)
                                
    # compute the RSS
    error = np.sum((model_ts-response)**2)
    
    # catch NaN
    if np.isnan(np.sum(error)):
        return np.inf
        
    txt = '%.03g,%.03g,%.03g,%.03g,%.03g,%.03g,%.03g,%.03g' %(error,x,y,sigma,hrf_delay,theta,phi,cpd)
    # print(txt)
    return error
    
def brute_force_search_lazy(x0, y0, s0, hrf0,
                            bounds, response, error_function,
                            deg_x, deg_y, stim_arr, 
                            tr_length, frames_per_tr, ppd):
                            
    [theta0, phi0, cpd0], err,  _, _ =\
        brute(error_function,
              args=(response, deg_x, deg_y, stim_arr, tr_length, frames_per_tr, ppd, x0, y0, s0, hrf0),
              ranges=bounds,
              Ns=4,
              finish=None,
              full_output=True,
              disp=False)
              
    # return the estimates
    return theta0, phi0, cpd0

def gradient_descent_search_lazy(theta0, phi0, cpd0, 
                                 x0, y0, s0, hrf0, 
                                 error_function, response, 
                                 deg_x, deg_y, stim_arr, 
                                 tr_length, frames_per_tr, ppd):
    
    [theta, phi, cpd], err,  _, _, _, warnflag =\
        fmin_powell(error_function_lazy,(theta0, phi0, cpd0),
                    args=(response, deg_x, deg_y, stim_arr, tr_length, frames_per_tr, ppd, x0, y0, s0, hrf0),
                    full_output=True,
                    disp=False)
    
    return theta, phi, cpd

def make_gabor(X,Y, ppd, theta, phi, trim, x0, y0, s0, cpd):
    theta_rad = theta * pi/180
    phase_rad = phase * pi/180
    
    XYt =  (X * cos(theta_rad)) + (Y * sin(theta_rad))
    XYf = XYt * cpd * pi/180
    
    grating = sin(XYf + phase_rad)
    gauss = np.exp(-((X-x0)**2+(Y-y0)**2)/(2*s0**2))
    gauss[gauss<trim] = 0
    gabor = grating*gauss
    
    return gabor

# this method is used to simply multiprocessing.Pool interactions
def parallel_fit(args):
    
    # unpackage the arguments
    response = args[0]
    model = args[1]
    bounds = args[2]
    tr_length = args[3]
    voxel_index = args[4]
    uncorrected_rval = args[5]
    verbose = args[6]
    lazy = args[7]
    auto_fit = args[8]
    
    # fit the data
    fit = GaborFit(response,
                   model,
                   bounds,
                   tr_length,
                   voxel_index,
                   uncorrected_rval,
                   verbose,
                   lazy,
                   auto_fit)
    return fit


class GaborModel(PopulationModel):
    
    """
    gabor population receptive field model.
    """
    
    def __init__(self, stimulus):
        
        # this is a weird notation
        PopulationModel.__init__(self, stimulus)
        
class GaborFit(object):
    
    """
    gabor population receptive field model fitting
    """
    
    def __init__(self, data, model, bounds, tr_length, voxel_index, uncorrected_rval, 
                 verbose=True, lazy=None, auto_fit=True):
            
        self.data = utils.zscore(data)
        self.model = model
        self.tr_length = tr_length
        self.voxel_index = voxel_index
        self.uncorrected_rval = uncorrected_rval
        self.verbose = verbose
        self.has_gaussian = False
        
        if auto_fit:
            
            # if we want to solve the gaussian prior to the gabor
            if lazy == 'both':
                self.bounds = bounds[0:4]
                tic = time.clock()
                
                # compute the gaussian first
                self.gaussian_model = gaussian.GaussianModel(self.model.stimulus)
                self.gaussian_fit = gaussian.GaussianFit(self.data, self.gaussian_model, self.bounds,
                                                         self.tr_length, self.voxel_index, 
                                                         self.uncorrected_rval, False, True)
                self.has_gaussian = True
                self.bounds = bounds[4::]
                
                # finish off the rest of the gabor parameters
                self.ballpark_estimate_lazy;
                self.gabor_estimate_lazy;
                toc = time.clock()
            
            # if we want to estimate the coarse paramters with just a gaussian model
            elif lazy == 'brute':
                self.bounds = bounds[0:4]
                tic = time.clock()
                
                # compute the gaussian first
                self.gaussian_model = gaussian.GaussianModel(self.model.stimulus)
                self.gaussian_fit = gaussian.GaussianFit(self.data, self.gaussian_model, self.bounds,
                                                         self.tr_length, self.voxel_index, 
                                                         self.uncorrected_rval, False, True)
                self.has_gaussian = True
                self.bounds = bounds[4::]
                
                # finish off the rest of the gabor parameters
                self.ballpark_estimate_lazy;
                self.bounds = bounds
                self.gabor_estimate;
                toc = time.clock()
            
            # if we want to estimate 7 at once
            else:
                self.bounds = bounds
                tic = time.clock()
                self.ballpark_estimate;
                self.gabor_estimate;
                toc = time.clock()
                self.has_gaussian = False
            
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
        return brute_force_search(self.bounds, self.data, error_function,
                                  self.model.stimulus.deg_x_coarse,
                                  self.model.stimulus.deg_y_coarse,
                                  self.model.stimulus.stim_arr_coarse,
                                  self.tr_length,
                                  self.model.stimulus.frames_per_tr,
                                  self.model.stimulus.ppd * self.model.stimulus.scale_factor)
    @auto_attr
    def ballpark_estimate_lazy(self):
      return brute_force_search_lazy(self.x0, self.y0, self.s0, self.hrf0,
                                     self.bounds, self.data, error_function_lazy,
                                     self.model.stimulus.deg_x_coarse,
                                     self.model.stimulus.deg_y_coarse,
                                     self.model.stimulus.stim_arr_coarse,
                                     self.tr_length,
                                     self.model.stimulus.frames_per_tr,
                                     self.model.stimulus.ppd * self.model.stimulus.scale_factor)
                                     
    @auto_attr
    def gabor_estimate(self):
        return gradient_descent_search(self.x0, self.y0, self.s0, self.hrf0, 
                                       self.theta0, self.phi0, self.cpd0,
                                       error_function, self.data, 
                                       self.model.stimulus.deg_x,
                                       self.model.stimulus.deg_y,
                                       self.model.stimulus.stim_arr,
                                       self.tr_length,
                                       self.model.stimulus.frames_per_tr,
                                       self.model.stimulus.ppd)
                                       
    @auto_attr
    def gabor_estimate_lazy(self):
       return gradient_descent_search_lazy(self.theta0, self.phi0, self.cpd0,
                                           self.x0, self.y0, self.s0, self.hrf0,
                                           error_function_lazy, self.data,
                                           self.model.stimulus.deg_x,
                                           self.model.stimulus.deg_y,
                                           self.model.stimulus.stim_arr,
                                           self.tr_length,
                                           self.model.stimulus.frames_per_tr,
                                           self.model.stimulus.ppd)
                                           
    @auto_attr
    def x0(self):
        if self.has_gaussian:
            return self.gaussian_fit.ballpark_estimate[0]
        else:
            return self.ballpark_estimate[0]
        
    @auto_attr
    def y0(self):
        if self.has_gaussian:
            return self.gaussian_fit.ballpark_estimate[1]
        else:
            return self.ballpark_estimate[1]
    
    @auto_attr
    def s0(self):
        if self.has_gaussian:
            return self.gaussian_fit.ballpark_estimate[2]
        else:
            return self.ballpark_estimate[2]
        
    @auto_attr
    def hrf0(self):
        if self.has_gaussian:
            return self.gaussian_fit.ballpark_estimate[3]
        else:
            return self.ballpark_estimate[3]
    
    @auto_attr
    def theta0(self):
        if self.has_gaussian:
            return self.ballpark_estimate_lazy[0]
        else:
            return self.ballpark_estimate[4]
    
    @auto_attr
    def phi0(self):
        if self.has_gaussian:
            return self.ballpark_estimate_lazy[1]
        else:
            return self.ballpark_estimate[5]
            
    @auto_attr
    def cpd0(self):
        if self.has_gaussian:
            return self.ballpark_estimate_lazy[2]
        else:
            return self.ballpark_estimate[6]
            
    @auto_attr
    def x(self):
        if self.has_gaussian:
            return self.gaussian_fit.gaussian_estimate[0]
        else:
            return self.gabor_estimate[0]
                
    @auto_attr
    def y(self):
        if self.has_gaussian:
            return self.gaussian_fit.gaussian_estimate[1]
        else:
            return self.gabor_estimate[1]
            
    @auto_attr
    def sigma(self):
        if self.has_gaussian:
            return self.gaussian_fit.gaussian_estimate[2]
        else:
            return self.gabor_estimate[2]
            
    @auto_attr
    def hrf_delay(self):
        if self.has_gaussian:
            return self.gaussian_fit.gaussian_estimate[3]
        else:
            return self.gabor_estimate[3]
    
    @auto_attr
    def theta(self):
        if self.has_gaussian:
            return self.gabor_estimate_lazy[0]
        else:
            return self.gabor_estimate[4]
        
    @auto_attr
    def phi(self):
        if self.has_gaussian:
            return self.gabor_estimate_lazy[1]
        else:
            return self.gabor_estimate[5]
            
    @auto_attr
    def cpd(self):
        if self.has_gaussian:
            return self.gabor_estimate_lazy[2]
        else:
            return self.gabor_estimate[6]
                
    @auto_attr
    def model_ts(self):
        return compute_model_ts(self.x, self.y, self.sigma, self.hrf_delay, 
                                self.theta, self.phi, self.cpd,
                                self.model.stimulus.deg_x,
                                self.model.stimulus.deg_y,
                                self.model.stimulus.stim_arr,
                                self.tr_length, self.model.stimulus.frames_per_tr)
    
    @auto_attr
    def fit_stats(self):
        return linregress(self.data, self.model_ts)
    
    @auto_attr
    def rss(self):
        return np.sum((self.data - self.model_ts)**2)
