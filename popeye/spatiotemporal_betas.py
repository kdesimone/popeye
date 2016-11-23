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
from popeye.spinach import generate_og_receptive_field, generate_mp_timeseries, generate_rf_timeseries

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
    
    def generate_mp_timeseries_long(self, spatial_ts):
        
        m_ts = np.zeros(self.stimulus.flicker_vec.shape[0])
        m_ts[self.stimulus.flicker_vec==1] = self.m_amp[0] * spatial_ts[self.stimulus.flicker_vec==1]
        m_ts[self.stimulus.flicker_vec==2] = self.m_amp[1] * spatial_ts[self.stimulus.flicker_vec==2]
        
        p_ts = np.zeros(self.stimulus.flicker_vec.shape[0])
        p_ts[self.stimulus.flicker_vec==1] = self.p_amp[0] * spatial_ts[self.stimulus.flicker_vec==1]
        p_ts[self.stimulus.flicker_vec==2] = self.p_amp[1] * spatial_ts[self.stimulus.flicker_vec==2]
        
        return m_ts, p_ts
        
    
    # for the final solution, we use spatiotemporal
    def generate_ballpark_prediction(self, x, y, sigma, mbeta, pbeta):
        
        # mask for speed
        mask = self.distance_mask_coarse(x, y, sigma)
        
        # generate the RF
        spatial_rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x0, self.stimulus.deg_y0)
        spatial_rf /= ((2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x0[0,0:2])**2)
        
        # spatial_response
        spatial_ts = generate_rf_timeseries(self.stimulus.stim_arr0, spatial_rf, mask)
        
        # temporal response
        m_ts, p_ts = generate_mp_timeseries(spatial_ts, self.m_amp, self.p_amp, self.stimulus.flicker_vec)
        
        # convolve with HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # M
        m_model = fftconvolve(m_ts, hrf)[0:len(m_ts)]
        
        # P
        p_model = fftconvolve(p_ts, hrf)[0:len(p_ts)]
        
        # convert units
        m_model = (m_model - np.mean(m_model))/np.mean(m_model)
        p_model = (p_model - np.mean(p_model))/np.mean(p_model)
        
        # mix
        model = m_model * mbeta + p_model * pbeta
        
        return model
    
    # for the final solution, we use spatiotemporal
    def generate_prediction(self, x, y, sigma, mbeta, pbeta):
        
        # mask for speed
        mask = self.distance_mask(x, y, sigma)
        
        # generate the RF
        spatial_rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        spatial_rf /= ((2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x[0,0:2])**2)
        
        # spatial response
        spatial_ts = generate_rf_timeseries(self.stimulus.stim_arr, spatial_rf, mask)
        
        # temporal response
        m_ts, p_ts = generate_mp_timeseries(spatial_ts, self.m_amp, self.p_amp, self.stimulus.flicker_vec)
        
        # convolve with HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # M
        m_model = fftconvolve(m_ts, hrf)[0:len(m_ts)]
        
        # P
        p_model = fftconvolve(p_ts, hrf)[0:len(p_ts)]
        
        # convert units
        m_model = (m_model - np.mean(m_model))/np.mean(m_model)
        p_model = (p_model - np.mean(p_model))/np.mean(p_model)
        
        # mix
        model = m_model * mbeta + p_model * pbeta
        
        return model
    
    @auto_attr
    def p(self):
        p = np.exp(-((self.t-self.center)**2)/(2*self.tau**2))
        p = p * 1/(np.sqrt(2*np.pi)*self.tau)
        return p
    
    @auto_attr
    def m(self):
        m = np.insert(np.diff(self.p),0,0)
        m = m/(simps(np.abs(m),self.t))
        return m
    
    def p_rf(self, tau):
        p = np.exp(-((self.t-self.center)**2)/(2*tau**2))
        p = p * 1/(np.sqrt(2*np.pi)*tau)
        return p
    
    def m_rf(self, tau):
        p = self.p_rf(tau)
        m = np.insert(np.diff(p),0,0)
        m = m/(simps(np.abs(m),self.t))
        return m
    
    @auto_attr
    def t(self):
        return np.linspace(0, self.stimulus.tr_length, self.stimulus.fps * self.stimulus.tr_length)
    
    @auto_attr
    def center(self):
        return self.t[len(self.t)/2]
    
    @auto_attr
    def flickers(self):
        return np.sin(2 * np.pi * np.single(self.stimulus.flicker_hz) * self.t[:,np.newaxis])
    
    @auto_attr
    def m_resp(self):
        m_resp = fftconvolve(self.flickers,self.m[:,np.newaxis])
        m_resp = utils.normalize(m_resp,-1,1)
        return m_resp
        
    def generate_m_resp(self, tau):
        m_rf = self.m_rf(tau)
        m_resp = fftconvolve(self.flickers,m_rf[:,np.newaxis])
        m_resp = utils.normalize(m_resp,-1,1)
        return m_resp
        
    @auto_attr
    def m_amp(self):
        return np.sum(np.abs(self.m_resp),0)
        
    @auto_attr
    def p_resp(self):
        p_resp = fftconvolve(self.flickers,self.p[:,np.newaxis])
        p_resp = utils.normalize(p_resp,-1,1)
        return p_resp
        
    def generate_p_resp(self, tau):
        p_rf = self.p_rf(tau)
        p_resp = fftconvolve(self.flickers,p_rf[:,np.newaxis])
        p_resp = utils.normalize(p_resp,-1,1)
        return p_resp
        
    @auto_attr
    def p_amp(self):
        return np.sum(np.abs(self.p_resp),0)
        
class SpatioTemporalFit(PopulationFit):
    
    """
    A spatiotemporal population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds, Ns,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
        
        PopulationFit.__init__(self, model, data, grids, bounds, Ns, 
                               voxel_index, auto_fit, verbose)
    
    @auto_attr
    def overloaded_estimate(self):
        return [self.theta, self.rho, self.sigma, self.mbeta, self.pbeta, self.weight]
    
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
    
    @auto_attr
    def prediction(self):
        return self.model.generate_prediction(*self.estimate)
        
    @auto_attr
    def rho(self):
        return np.sqrt(self.x**2+self.y**2)
    
    @auto_attr
    def weight(self):
        return (self.pbeta - self.mbeta) / (self.pbeta + self.mbeta)
    
    @auto_attr
    def theta(self):
        return np.mod(np.arctan2(self.y,self.x),2*np.pi)
    
    @auto_attr
    def receptive_field(self):
        return generate_og_receptive_field(self.x, self.y, self.sigma,
                                           self.model.stimulus.deg_x,
                                           self.model.stimulus.deg_y)
