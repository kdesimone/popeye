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
from scipy.optimize import fmin
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_2dcos_receptive_field, generate_mp_timeseries, generate_rf_timeseries

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
    
    
    def generate_ballpark_prediction(self, x, y, sigma, weight):
        
        r"""
        Predict signal for the Gaussian Model using the downsampled stimulus.
        The rate of stimulus downsampling is defined in `model.stimulus.scale_factor`.
        
        Parameters
        __________
        x : float
            Horizontal location of the 2D Cosine RF.
        
        y: float 
            Vertical location of the 2D Cosine RF.
        
        sigma: float
            Dipsersion of the 2D Cosine RF.
        
        weight: float
            Mixture of the magnocellar and parvocellular temporal response to
            a flickering visual stimulus. The `weight` ranges between 0 and 1, 
            with 0 being a totally magnocellular response and ` being a totally
            parvocellular response.
        
        """
        # generate the RF
        spatial_rf = generate_2dcos_receptive_field(x, y, sigma, self.power, self.stimulus.deg_x0, self.stimulus.deg_y0)
        
        # normalize by volume
        spatial_rf /= (trapz(trapz(spatial_rf)) * 1/np.diff(self.stimulus.deg_x0[0,0:2])**2)
        
        # make mask
        mask = np.uint8(spatial_rf>0)
        
        # spatial_response
        spatial_ts = generate_rf_timeseries(self.stimulus.stim_arr0, spatial_rf, mask)
        
        # temporal response
        m_ts, p_ts = generate_mp_timeseries(spatial_ts, self.m_amp, self.p_amp, self.stimulus.flicker_vec)
        
        # mix them
        mp_ts = (1-weight) * m_ts + weight * p_ts
        
        # convolve with HRF
        model = fftconvolve(mp_ts, self.hrf())[0:len(mp_ts)]
        
        # units
        model = (model - np.mean(model)) / np.mean(model)
        
        # regress out mean and linear
        p = linregress(model, self.data)
        
        # scale
        model *= p[0]
        
        # offset
        model += p[1]
        
        return model
    
    def generate_prediction(self, x, y, sigma, weight, beta, baseline,  unscaled=False):
        
        r"""
        Predict signal for the Gaussian Model using the full resolution stimulus.
        
        Parameters
        __________
        x : float
            Horizontal location of the 2D Cosine RF.
        
        y: float 
            Vertical location of the 2D Cosine RF.
        
        sigma: float
            Dipsersion of the 2D Cosine RF.
        
        weight: float
            Mixture of the magnocellar and parvocellular temporal response to
            a flickering visual stimulus. The `weight` ranges between 0 and 1, 
            with 0 being a totally magnocellular response and ` being a totally
            parvocellular response.
            
        beta : float
            Amplitude scaling factor to account for units.
            
        baseline: float
            Amplitude intercept to account for baseline.
        
        """
        
        # generate the RF
        spatial_rf = generate_2dcos_receptive_field(x, y, sigma, self.power, self.stimulus.deg_x, self.stimulus.deg_y)
        
        # normalize by volume
        spatial_rf /= (trapz(trapz(spatial_rf)) * 1/np.diff(self.stimulus.deg_x[0,0:2])**2)
        
        # make mask
        mask = np.uint8(spatial_rf>0)
        
        # spatial response
        spatial_ts = generate_rf_timeseries(self.stimulus.stim_arr, spatial_rf, mask)
        
        # temporal response
        m_ts, p_ts = generate_mp_timeseries(spatial_ts, self.m_amp, self.p_amp, self.stimulus.flicker_vec)
        
        # mix them
        mp_ts = (1-weight) * m_ts + weight * p_ts 
        
        # convolve with HRF
        model = fftconvolve(mp_ts, self.hrf())[0:len(mp_ts)]
        
        # units
        model = (model - np.mean(model)) / np.mean(model)
        
        if unscaled:
            return model
        else:
            
            # scale it by beta
            model *= beta
            
            # offset
            model += baseline
            
            return model
    
    @auto_attr
    def p(self):
        
        r"""Returns the parvocellular sustained receptive field. The temporal dispersion
        parameter `tau` must be hard-set in the `SpatioTemporalModel.tau` by the user."""
        
        p = np.exp(-((self.t-self.center)**2)/(2*self.tau**2))
        p = p * 1/(np.sqrt(2*np.pi)*self.tau)
        return p
    
    @auto_attr
    def m(self):
        
        r"""Returns the magnocellar transient receptive field. The temporal dispersion
        parameter `tau` must be hard-set in the `SpatioTemporalModel.tau` by the user."""
        
        m = np.insert(np.diff(self.p),0,0)
        m = m/(simps(np.abs(m),self.t))
        return m
    
    def p_rf(self, tau):
        
        r""" Returns the sustained parvocellular receptive field. The sustained receptive field
        is a one dimensional gaussian in time.
        
        Parameters
        ----------
        
        tau : float
            The temporal dispersion of the sustained temporal receptive field.
            
        Returns
        -------
        
        p : ndarray
            The sustained parvocellular temporal receptive field normalized by its integral.
            
        """
        
        p = np.exp(-((self.t-self.center)**2)/(2*tau**2))
        p = p * 1/(np.sqrt(2*np.pi)*tau)
        return p
    
    def m_rf(self, tau):
        
        r""" Returns the transient magnocellular receptive field. The transient rectpive field
        is the temporal derivative of the sustained reponse.
        
        Parameters
        ----------
        
        tau : float
            The temporal dispersion of the sustained temporal receptive field.
            
        Returns
        -------
        
        m : ndarray
            The transient magnocellular temporal receptive field normalized by its integral.
            
        """
        
        p = self.p_rf(tau)
        m = np.insert(np.diff(p),0,0)
        m = m/(simps(np.abs(m),self.t))
        
        return m
    
    @auto_attr
    def t(self):
        
        r""" Returns the time coordinate."""
        
        return np.linspace(0, self.stimulus.tr_length, self.stimulus.fps * self.stimulus.tr_length)
    
    @auto_attr
    def center(self):
        
        r""" Returns the center of the time coordinate for constructing the temporal RFs."""
        
        return self.stimulus.tr_length/2
    
    @auto_attr
    def flickers(self):
        
        r""" Returns the temporal profiles of the flickering stimuli."""
        
        return np.sin(2 * np.pi * np.single(self.stimulus.flicker_hz) * self.t[:,np.newaxis])
    
    @auto_attr
    def m_resp(self):
        
        r""" Returns the transient magnocellular response to flicker stimulus."""
        
        m_resp = fftconvolve(self.flickers,self.m[:,np.newaxis])
        m_resp= utils.normalize(m_resp,-1,1)
        return m_resp
        
    def generate_m_resp(self, tau):
        
        r""" Returns the transient magnocellular response to flicker stimulus.
        
        Parameters
        ----------
        
        tau : float
            The temporal dispersion of the sustained temporal receptive field.
            
        Returns
        -------
        
        m_resp : ndarray
            The temporal response of the flickering visual stimulus convolved
            with the transient magnocellular temporal receptive field.
            
        """
        
        m_rf = self.m_rf(tau)
        m_resp = fftconvolve(self.flickers,m_rf[:,np.newaxis])
        m_resp = utils.normalize(m_resp,-1,1)
        return m_resp
    
    @auto_attr
    def m_amp(self):
        
        r""" Returns the amplitude of the transient mangmocellular response to flicker stimulus."""
        
        m_amp = np.sum(np.abs(self.m_resp),0)
        m_amp /= m_amp.max()
        return m_amp
    
    @auto_attr
    def p_resp(self):
        
        r""" Returns the sustained parvocellular response to flicker stimulus."""
        
        p_resp = fftconvolve(self.flickers,self.p[:,np.newaxis])
        p_resp = utils.normalize(p_resp,-1,1)
        return p_resp
    
    def generate_p_resp(self, tau):
        
        r""" Returns the sustained parvocellular response to flicker stimulus.
        
        Parameters
        ----------
        
        tau : float
            The temporal dispersion of the sustained temporal receptive field.
            
        Returns
        -------
        
        p_resp : ndarray
            The temporal response of the flickering visual stimulus convolved
            with the sustained parvocellular temporal receptive field.
            
        """
        
        p_rf = self.p_rf(tau)
        p_resp = fftconvolve(self.flickers,p_rf[:,np.newaxis])
        p_resp = utils.normalize(p_resp,-1,1)
        return p_resp
    
    @auto_attr    
    def p_amp(self):
        
        r""" Returns the amplitude of the transient mangmocellular response to flicker stimulus."""
        
        p_amp = np.sum(np.abs(self.p_resp),0)
        p_amp /= np.max(p_amp)
        return p_amp
    
    def generate_receptive_field(self, x, y, sigma):
        
        r"""
        Generate a 2D Cosine receptive field in stimulus-referred coordinates.
        
        Parameters
        __________
        x : float
            Horizontal location of the 2D Cosine RF.
            
        y: float 
            Vertical location of the 2D Cosine RF.
            
        sigma: float
            Dipsersion of the 2D Cosine RF.
        
        """
        
        # generate the RF
        rf = generate_2dcos_receptive_field(x, y, sigma, self.power, self.stimulus.deg_x, self.stimulus.deg_y)
        rf /= (trapz(trapz(rf)) * 1/np.diff(self.stimulus.deg_x[0,0:2])**2)
        
        return rf
    
class SpatioTemporalFit(PopulationFit):
    
    """
    A spatiotemporal population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds, 
                 voxel_index=(1,2,3), Ns=None, auto_fit=True, verbose=0):
        
        PopulationFit.__init__(self, model, data, grids, bounds, 
                               voxel_index, Ns, auto_fit, verbose)
    
    @auto_attr
    def overloaded_estimate(self):
        return [self.theta, self.rho, self.sigma, self.weight, self.beta, self.baseline]
    
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
    def weight0(self):
        return self.ballpark[3]
    
    @auto_attr
    def beta0(self):
        return self.ballpark[4]
    
    @auto_attr
    def baseline0(self):
        return self.ballpark[5]
    
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
    def weight(self):
        return self.estimate[3]
    
    @auto_attr
    def beta(self):
        return self.estimate[4]
    
    @auto_attr
    def baseline(self):
        return self.estimate[5]
        
    @auto_attr
    def rho(self):
        
        r""" Returns the eccentricity of the fitted pRF. """
        
        return np.sqrt(self.x**2+self.y**2)
        
    @auto_attr
    def theta(self):
        
        r""" Returns the polar angle of the fitted pRF. """
        
        return np.mod(np.arctan2(self.y,self.x),2*np.pi)
        
    @auto_attr
    def receptive_field(self):
        
        r""" Returns the fitted 2D Cosine pRF. """
        
        return self.model.generate_receptive_field(self.x, self.y, self.sigma)
