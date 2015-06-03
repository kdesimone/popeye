#!/usr/bin/python

""" Classes and functions for fitting Gaussian population encoding models """

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

class GaussianModel(PopulationModel):
    
    """
    A Gaussian population receptive field model class
    
    """
    
    def __init__(self, stimulus, hrf_model):
        
        """
        A Gaussian population receptive field model [1]_.
        
        Paramaters
        ----------
        
        stimulus : `VisualStimulus` class object
            A class instantiation of the `VisualStimulus` class
            containing a representation of the visual stimulus.
            
        hrf_model : 
        
        
        References
        ----------
        
        .. [1] Dumoulin SO, Wandell BA. (2008) Population receptive field 
        estimates in human visual cortex. NeuroImage 39:647-660
        
        """
        
        PopulationModel.__init__(self, stimulus, hrf_model)
    
    # main method for deriving model time-series
    def generate_ballpark_prediction(self, x, y, sigma, beta, hrf_delay):
        
        # create mask for speed
        distance = (self.stimulus.deg_x_coarse - x)**2 + (self.stimulus.deg_y_coarse - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (5*sigma)**2] = 1
        
        # generate the RF
        rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x_coarse, self.stimulus.deg_y_coarse)
        rf /= 2 * np.pi * sigma**2
        rf *= beta
        
        # extract the stimulus time-series
        response = generate_rf_timeseries(self.stimulus.stim_arr_coarse, rf, mask)
        
        # generate HRF
        hrf = self.hrf_model(hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf, 'same') / len(response)
        
        # scale it by beta
        model *= beta
        
        return model
        
    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, beta, hrf_delay):
        
        # create mask of central 5 sigmas for speed
        distance = (self.stimulus.deg_x - x)**2 + (self.stimulus.deg_y - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (5*sigma)**2] = 1
        
        # generate the RF
        rf = generate_og_receptive_field(x, y, sigma, 
                                         self.stimulus.deg_x,
                                         self.stimulus.deg_y)
        
        # normalize by the integral
        rf /= (2 * np.pi * sigma**2)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries(self.stimulus.stim_arr, rf, mask)
        
        # convolve with the HRF
        hrf = self.hrf_model(hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf, 'same') / len(response)
        
        # scale it by beta
        model *= beta
        
        return model 
        
class GaussianFit(PopulationFit):
    
    """
    A Gaussian population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds, Ns,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
        
        
        """
        A Gaussian population receptive field model [1]_.

        Paramaters
        ----------
        
        data : ndarray
            An array containing the measured BOLD signal.
        
        model : `GaussianModel` class instance containing the representation
            of the visual stimulus.
        
        search_bounds : tuple
            A tuple indicating the search space for the brute-force grid-search.
            The tuple contains pairs of upper and lower bounds for exploring a
            given dimension.  For example `fit_bounds=((-10,10),(0,5),)` will
            search the first dimension from -10 to 10 and the second from 0 to 5.
            These values cannot be None. 
            
            For more information, see `scipy.optimize.brute`.
        
        fit_bounds : tuple
            A tuple containing the upper and lower bounds for each parameter
            in `parameters`.  If a parameter is not bounded, simply use
            `None`.  For example, `fit_bounds=((0,None),(-10,10),)` would 
            bound the first parameter to be any positive number while the
            second parameter would be bounded between -10 and 10.
        
        tr_length : float
            The length of the repetition time in seconds.
        
        voxel_index : tuple
            A tuple containing the index of the voxel being modeled. The 
            fitting procedure does not require a voxel index, but 
            collating the results across many voxels will does require voxel
            indices. With voxel indices, the brain volume can be reconstructed 
            using the newly computed model estimates.
        
        auto-fit : bool
            A flag for automatically running the fitting procedures once the 
            `GaussianFit` object is instantiated.
        
        verbose : bool
            A flag for printing some summary information about the model estiamte
            after the fitting procedures have completed.
        
        References
        ----------
        
        .. [1] Dumoulin SO, Wandell BA. (2008) Population receptive field 
        estimates in human visual cortex. NeuroImage 39:647-660

        """
        
        PopulationFit.__init__(self, model, data, grids, bounds, Ns, 
                               voxel_index, auto_fit, verbose)
                               
    @auto_attr
    def x0(self):
        return self.ballpark[0]
        
    @auto_attr
    def y0(self):
        return self.ballpark[1]
        
    @auto_attr
    def s0(self):
        return self.ballpark[2]
    
    @auto_attr
    def beta0(self):
        return self.ballpark[3]
    
    @auto_attr
    def hrf0(self):
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
    def beta(self):
        return self.estimate[3]
    
    @auto_attr
    def hrf_delay(self):
        return self.estimate[4]
    
    @auto_attr
    def rho(self):
        return np.sqrt(self.x**2+self.y**2)
    
    @auto_attr
    def theta(self):
        return np.mod(np.arctan2(self.y,self.x),2*np.pi)
    
    @auto_attr
    def receptive_field(self):
        return generate_og_receptive_field(self.x, self.y, self.sigma,
                                           self.stimulus.deg_x,
                                           self.stimulus.deg_y)
    
    @auto_attr
    def msg(self):
        txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RSQ=%.02f  THETA=%.02f   RHO=%.02d   SIGMA=%.02f   BETA=%.08f   HRF=%.02f" 
            %(self.voxel_index[0],
              self.voxel_index[1],
              self.voxel_index[2],
              self.finish-self.start,
              self.rsquared,
              self.theta,
              self.rho,
              self.sigma,
              self.beta,
              self.hrf_delay))
        return txt
