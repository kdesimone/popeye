#!/usr/bin/python

""" Classes and functions for fitting SpatioTemporal population receptive field models """

from __future__ import division, print_function, absolute_import
import time
import gc
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.stats import linregress
from scipy.signal import fftconvolve
from scipy.integrate import trapz, simps
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries

class SpatioTemporalModel(PopulationModel):
    
    """
    A Gaussian population receptive field model class
    
    """
    
    def __init__(self, stimulus, hrf_model):
        
        """
        A spatiotemporal population receptive field model.
        
        Paramaters
        ----------
        
        stimulus : `VisualStimulus` class object
            A class instantiation of the `VisualStimulus` class
            containing a representation of the visual stimulus.
        
        
        """
        
        PopulationModel.__init__(self, stimulus, hrf_model)
    
    
    def generate_ballpark_prediction(self, x, y, sigma, beta, baseline, hrf_delay, mbeta, pbeta):
        
        # generate the RF
        spatial_rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x_coarse, self.stimulus.deg_y_coarse)
        spatial_rf /= (2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x_coarse[0,0:2])**2
        
        # if the spatial RF is running off the screen ...
        x_rf = np.sum(spatial_rf,axis=1)
        y_rf = np.sum(spatial_rf,axis=0)
        if ( np.round(x_rf[0],3) != 0 or np.round(x_rf[-1],3) != 0 or 
             np.round(y_rf[0],3) != 0 or np.round(x_rf[-1],3) != 0 ):
            return np.inf
        
        # create mask for speed
        distance = (self.stimulus.deg_x_coarse - x)**2 + (self.stimulus.deg_y_coarse - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (3*sigma)**2] = 1
        
        # set the coordinate
        t = np.linspace(0, 1, self.stimulus.fps_coarse)
        tsigma = 0.01
        
        # set the center of the responses
        center = t[len(t)/2]
        
        # create sustain response for 1 TR
        p = np.exp(-((t-center)**2)/(2*tsigma**2))
        p = p * 1/(np.sqrt(2*np.pi)*tsigma)
        
        # create transient response for 1 TR
        m = np.insert(np.diff(p),0,0)
        m = m/(simps(np.abs(m),t))
        
        if ( np.round(p[0],3) != 0 or np.round(p[-1],3) != 0 or 
             np.round(m[0],3) != 0 or np.round(m[-1],3) != 0 ):
            return np.inf
        
        # extract the timeseries
        s_resp = generate_rf_timeseries(self.stimulus.stim_arr_coarse, spatial_rf, mask)
        
        # convolve with transient and sustained RF
        p_resp = np.abs(fftconvolve(s_resp,p)[0:len(s_resp)] / len(s_resp))
        m_resp = np.abs(fftconvolve(s_resp,m)[0:len(s_resp)] / len(s_resp))
        
        # take the mean of each TR
        s_ts = np.array([np.mean(s_resp[tp:tp+self.stimulus.fps_coarse*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.fps_coarse*self.stimulus.tr_length)])
        p_ts = np.array([np.mean(p_resp[tp:tp+self.stimulus.fps_coarse*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.fps_coarse*self.stimulus.tr_length)])
        m_ts = np.array([np.mean(m_resp[tp:tp+self.stimulus.fps_coarse*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.fps_coarse*self.stimulus.tr_length)])
        
        # convolve with hrf 
        hrf = self.hrf_model(hrf_delay, self.stimulus.tr_length)
        sustained_model = fftconvolve(p_ts, hrf)[0:len(p_ts)]
        transient_model = fftconvolve(m_ts, hrf)[0:len(m_ts)]
        
        # M+P mixture
        model = sustained_model * mbeta + transient_model * pbeta
        
        # scale
        model *= beta
         
        # offset
        model += baseline
        
        return model
    
    # for the final solution, we use spatiotemporal
    def generate_prediction(self, x, y, sigma, beta, baseline, hrf_delay, mbeta, pbeta):
        
        # generate the RF
        spatial_rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        spatial_rf /= (2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x[0,0:2])**2
        
        # if the spatial RF is running off the screen ...
        x_rf = np.sum(spatial_rf,axis=1)
        y_rf = np.sum(spatial_rf,axis=0)
        if ( np.round(x_rf[0],3) != 0 or np.round(x_rf[-1],3) != 0 or 
             np.round(y_rf[0],3) != 0 or np.round(x_rf[-1],3) != 0 ):
            return np.inf
        
        # create mask for speed
        distance = (self.stimulus.deg_x - x)**2 + (self.stimulus.deg_y - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (3*sigma)**2] = 1
        
        # set the coordinate
        t = np.linspace(0, 1, self.stimulus.fps)
        tsigma = 0.01
        
        # set the center of the responses
        center = t[len(t)/2]
        
        # create sustain response for 1 TR
        p = np.exp(-((t-center)**2)/(2*tsigma**2))
        p = p * 1/(np.sqrt(2*np.pi)*tsigma)
        
        # create transient response for 1 TR
        m = np.insert(np.diff(p),0,0)
        m = m/(simps(np.abs(m),t))
        
        if ( np.round(p[0],3) != 0 or np.round(p[-1],3) != 0 or 
             np.round(m[0],3) != 0 or np.round(m[-1],3) != 0 ):
            return np.inf
        
        # extract the timeseries
        s_resp = generate_rf_timeseries(self.stimulus.stim_arr, spatial_rf, mask)
        
        # convolve with transient and sustained RF
        p_resp = np.abs(fftconvolve(s_resp,p)[0:len(s_resp)] / len(s_resp))
        m_resp = np.abs(fftconvolve(s_resp,m)[0:len(s_resp)] / len(s_resp))
        
        # take the mean of each TR
        s_ts = np.array([np.mean(s_resp[tp:tp+self.stimulus.fps*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.fps*self.stimulus.tr_length)])
        p_ts = np.array([np.mean(p_resp[tp:tp+self.stimulus.fps*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.fps*self.stimulus.tr_length)])
        m_ts = np.array([np.mean(m_resp[tp:tp+self.stimulus.fps*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.fps*self.stimulus.tr_length)])
        
        # convolve with hrf 
        hrf = self.hrf_model(hrf_delay, self.stimulus.tr_length)
        sustained_model = fftconvolve(p_ts, hrf)[0:len(p_ts)]
        transient_model = fftconvolve(m_ts, hrf)[0:len(m_ts)]
        
        # M+P mixture
        model = sustained_model * mbeta + transient_model * pbeta
        
        # scale
        model *= beta
         
        # offset
        model += baseline
        
        return model
    
class SpatioTemporalFit(PopulationFit):
    
    """
    A spatiotemporal population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds, Ns,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
        
        PopulationFit.__init__(self, model, data, grids, bounds, Ns, 
                               voxel_index, auto_fit, verbose)
    
    @auto_attr
    def x0(self):
        return self.ballpark[0]
        
    @auto_attr
    def y0(self):
        return self.ballpark[1]
    
    @auto_attr
    def sigma0(self):
        return self.ballpark[2]
        
    @auto_attr
    def mbeta0(self):
        return self.ballpark[3]
    
    @auto_attr
    def pbeta0(self):
        return self.ballpark[4]
    
    @auto_attr
    def baseline0(self):
        return self.ballpark[5]
        
    @auto_attr
    def hrf0(self):
        return self.ballpark[6]
        
    @auto_attr
    def x(self):
        return self.estimate[0]
    
    @auto_attr
    def y(self):
        return self.estimate[1]
    
    @auto_attr
    def sigma(self):
        return self.estimate[2]
    
    @auto_attr
    def mbeta(self):
        return self.estimate[3]
    
    @auto_attr
    def pbeta(self):
        return self.estimate[4]
    
    def baseline(self):
        return self.estimate[5]
        
    @auto_attr
    def hrf_delay(self):
        return self.estimate[6]
    
    @auto_attr
    def prediction(self):
        return self.model.generate_prediction(*self.estimate)
        
    @auto_attr
    def rho(self):
        return np.sqrt(self.x**2+self.y**2)
    
    @auto_attr
    def theta(self):
        return np.mod(np.arctan2(self.y,self.x),2*np.pi)
    
    @auto_attr
    def receptive_field(self):
        return generate_og_receptive_field(self.x, self.y, self.sigma,
                                           self.model.stimulus.deg_x,
                                           self.model.stimulus.deg_y)