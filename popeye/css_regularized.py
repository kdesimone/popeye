#!/usr/bin/python

""" Classes and functions for estimating Compressive Spatial Summation pRF model """

from __future__ import division
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.signal import fftconvolve
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries_nomask

class CompressiveSpatialSummationModel(PopulationModel):
    
    r"""
    A Compressive Spatial Summation population receptive field model class
    
    """
    
    def __init__(self, stimulus, hrf_model, nuisance=None):
        
        r"""
        A Compressive Spatial Summation population receptive field model [1]_.
        
        Paramaters
        ----------
        
        stimulus : `VisualStimulus` class object
            A class instantiation of the `VisualStimulus` class
            containing a representation of the visual stimulus.
        
        hrf_model : callable
            A function that generates an HRF model given an HRF delay.
            For more information, see `popeye.utilties.double_gamma_hrf_hrf`
        
        References
        ----------
        
        .. [1] Kay KN, Winawer J, Mezer A, Wandell BA (2014) Compressive spatial
        summation in human visual cortex. Journal of Neurophysiology 110:481-494.
        
        """
        
        PopulationModel.__init__(self, stimulus, hrf_model, nuisance)
        
    
    # main method for deriving model time-series
    def generate_ballpark_prediction(self, x, y, sigma, beta):
        
        # generate the RF
        rf = generate_og_receptive_field(x, y, sigma,self.stimulus.deg_x0, self.stimulus.deg_y0)
        
        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x0[0,0:2])**2)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr0, rf)
        
        # compression
        response **= self.n()
        
        # convolve with the HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf)[0:len(response)]
        
        # convert units
        model = (model - np.mean(model)) / np.mean(model)
        
        # scale it by beta
        model *= beta
        
        return model
        
    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, beta):
        
        # generate the RF
        rf = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        
        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1/np.diff(self.stimulus.deg_x[0,0:2])**2)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr, rf)
        
        # compression
        response **= self.n()
        
        # convolve with the HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf)[0:len(response)]
        
        # convert units
        model = (model - np.mean(model)) / np.mean(model)
        
        # scale it by beta
        model *= beta
        
        return model
    
    # rename
    def n(self):
        return self.parameter
        
class CompressiveSpatialSummationFit(PopulationFit):
    
    """
    A Compressive Spatial Summation population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds,
                 voxel_index=(1,2,3), Ns=None, auto_fit=True, verbose=0):
        
        
        r"""
        A class containing tools for fitting the CSS pRF model.
        
        The `CompressiveSpatialSummationFit` class houses all the fitting tool that 
        are associated with estimating a pRF model. The `GaussianFit` takes a 
        `CompressiveSpatialSummationModel` instance  `model` and a time-series `data`. 
        In addition, extent and sampling-rate of a  brute-force grid-search is set 
        with `grids` and `Ns`.  Use `bounds` to set limits on the search space for 
        each parameter.  
        
        Paramaters
        ----------
        
                
        model : `CompressiveSpatialSummationModel` class instance
            An object representing the CSS model. 
        
        data : ndarray
            An array containing the measured BOLD signal of a single voxel.
        
        grids : tuple
            A tuple indicating the search space for the brute-force grid-search.
            The tuple contains pairs of upper and lower bounds for exploring a
            given dimension.  For example `grids=((-10,10),(0,5),)` will
            search the first dimension from -10 to 10 and the second from 0 to 5.
            These values cannot be `None`. 
            
            For more information, see `scipy.optimize.brute`.
        
        bounds : tuple
            A tuple containing the upper and lower bounds for each parameter
            in `parameters`.  If a parameter is not bounded, simply use
            `None`.  For example, `fit_bounds=((0,None),(-10,10),)` would 
            bound the first parameter to be any positive number while the
            second parameter would be bounded between -10 and 10.
        
        Ns : int
            Number of samples per stimulus dimension to sample during the ballpark search.
            
            For more information, see `scipy.optimize.brute`.
        
        voxel_index : tuple
            A tuple containing the index of the voxel being modeled. The 
            fitting procedure does not require a voxel index, but 
            collating the results across many voxels will does require voxel
            indices. With voxel indices, the brain volume can be reconstructed 
            using the newly computed model estimates.
        
        auto_fit : bool
            A flag for automatically running the fitting procedures once the 
            `GaussianFit` object is instantiated.
        
        verbose : int
            0 = silent
            1 = print the final solution of an error-minimization
            2 = print each error-minimization step
        
        """
        
        PopulationFit.__init__(self, model, data, grids, bounds, 
                               voxel_index, Ns, auto_fit, verbose)
    
    @auto_attr
    def overloaded_estimate(self):
        return [self.theta, self.rho, self.sigma_size, self.n, self.beta, self.n]
    
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
    def n(self):
        return self.model.n()
        
    @auto_attr
    def rho(self):
        return np.sqrt(self.x**2+self.y**2)
    
    @auto_attr
    def theta(self):
        return np.mod(np.arctan2(self.y,self.x),2*np.pi)
    
    @auto_attr
    def sigma_size(self):
        return self.sigma / np.sqrt(self.n)