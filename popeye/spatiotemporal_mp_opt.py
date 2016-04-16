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
from popeye.spinach import generate_og_receptive_field, generate_strf_timeseries, generate_rf_timeseries

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
    
    
    # for the final solution, we use spatiotemporal
    def generate_ballpark_prediction(self, x, y, sigma, m_beta, p_beta, baseline, hrf_delay):
        
        # generate the RF
        spatial_rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x_coarse, self.stimulus.deg_y_coarse)
        spatial_rf /= ((2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x_coarse[0,0:2])**2)
        
        # create mask for speed
        distance = (self.stimulus.deg_x_coarse - x)**2 + (self.stimulus.deg_y_coarse - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (1*sigma)**2] = 1
        
        # spatial response
        rf_ts = generate_rf_timeseries(self.stimulus.stim_arr_coarse, spatial_rf, mask)
        
        # temporal response
        m_ts, p_ts = generate_strf_timeseries(rf_ts, self.m_resp, self.p_resp, self.stimulus.flicker_vec)
        
        # normalize each timeseries
        m_ts = utils.normalize(m_ts,0,1)
        p_ts = utils.normalize(p_ts,0.1)
        
        # mix them
        mp_ts = (m_ts * m_beta + p_ts * p_beta) / 2
        
        # convolve with HRF
        hrf = self.hrf_model(hrf_delay, self.stimulus.tr_length)
        model = fftconvolve(mp_ts, hrf)[0:len(mp_ts)]
        
        # offset
        model += baseline
        
        return model
    
    # for the final solution, we use spatiotemporal
    def generate_prediction(self, x, y, sigma, m_beta, p_beta, baseline, hrf_delay):
        
        # generate the RF
        spatial_rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        spatial_rf /= ((2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x[0,0:2])**2)
        
        # create mask for speed
        distance = (self.stimulus.deg_x - x)**2 + (self.stimulus.deg_y - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (1*sigma)**2] = 1
        
        # spatial response
        rf_ts = generate_rf_timeseries(self.stimulus.stim_arr, spatial_rf, mask)
        
        # temporal response
        m_ts, p_ts = generate_strf_timeseries(rf_ts, self.m_resp, self.p_resp, self.stimulus.flicker_vec)
        
        # normalize each timeseries
        m_ts = utils.normalize(m_ts,0,1)
        p_ts = utils.normalize(p_ts,0.1)
        
        # mix them
        mp_ts = (m_ts * m_beta + p_ts * p_beta) / 2
        
        # convolve with HRF
        hrf = self.hrf_model(hrf_delay, self.stimulus.tr_length)
        model = fftconvolve(mp_ts, hrf)[0:len(mp_ts)]
        
        # offset
        model += baseline
        
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
    
    @auto_attr
    def t(self):
        return np.linspace(0, self.stimulus.tr_length, self.stimulus.fps * self.stimulus.tr_length)
    
    @auto_attr
    def center(self):
        return self.t[len(self.t)/2]
    
    @auto_attr
    def flickers(self):
        ts = self.t.shape[-1]
        fs = int(np.max(self.stimulus.flicker_vec))
        flickers = np.zeros((ts,fs))
        for f in xrange(1,fs+1):
            flickers[:,f-1] = np.sin(2 * np.pi * self.stimulus.flicker_hz[f-1] * self.t)
        return flickers
    
    @auto_attr
    def p_resp(self):
        ts = self.t.shape[-1]
        fs = int(np.max(self.stimulus.flicker_vec))
        p_resp = np.zeros((ts,fs))
        for f in xrange(1,fs+1):
            p_resp[:,f-1] = np.abs(fftconvolve(self.flickers[:,f-1],self.p))[0:len(self.flickers[:,f-1])] / len(self.flickers[:,f-1])
        return p_resp
    
    @auto_attr
    def m_resp(self):
        ts = self.t.shape[-1]
        fs = int(np.max(self.stimulus.flicker_vec))
        m_resp = np.zeros((ts,fs))
        for f in xrange(1,fs+1):
            m_resp[:,f-1] = np.abs(fftconvolve(self.flickers[:,f-1],self.m))[0:len(self.flickers[:,f-1])] / len(self.flickers[:,f-1])
        return m_resp
        
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
        return [self.theta,self.rho,self.sigma,self.weight,self.mbeta,self.pbeta,self.baseline,self.hrf_delay + 5.0]
    
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
        
    @auto_attr
    def baseline(self):
        return self.estimate[5]
        
    @auto_attr
    def hrf_delay(self):
        return self.estimate[6]
    
    @auto_attr
    def prediction(self):
        return self.model.generate_prediction(*self.estimate)
    
    @auto_attr
    def weight(self):
        return (self.pbeta - self.mbeta) / (self.mbeta + self.pbeta)
        
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
    @auto_attr
    def ideal_magno_resp(self):
        response = self.model.generate_prediction(self.x, self.y, self.sigma, self.mbeta, 0, self.baseline, self.hrf_delay)
        return utils.normalize(response, self.prediction.min(), self.prediction.max())
    
    @auto_attr
    def ideal_parvo_resp(self):
        response = self.model.generate_prediction(self.x, self.y, self.sigma, 0, self.pbeta, self.baseline, self.hrf_delay)
        return utils.normalize(response, self.prediction.min(), self.prediction.max())

        