#!/usr/bin/python

""" Classes and functions for fitting Gaussian population encoding models """

from __future__ import division, print_function, absolute_import
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np
np.set_printoptions(suppress=True)
from scipy.stats import linregress
from scipy.signal import fftconvolve
from scipy.integrate import trapz

import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries_nomask

class DifferenceOfGaussiansModel(PopulationModel):
    
    r"""
    A Difference of Gaussian population receptive field model class
    
    """
    
    def __init__(self, stimulus, hrf_model):
        
        r"""
        A Difference of Gaussian population receptive field model [1]_.
        
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
        
        .. [1] Zuiderbaan W, Harvey BM, Dumoulin SO (2012) Modeling 
        center-surroundconfigurations in population receptive fields 
        using fMRI. Journal of Vision 12(3):10,1-15.
        
        """
        PopulationModel.__init__(self, stimulus, hrf_model)
        
        
    def generate_ballpark_prediction(self, x, y, sigma, sigma_ratio, volume_ratio):
        
        # extract the center response
        rf_center = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x0, self.stimulus.deg_y0)
        
        # extract surround response
        rf_surround = generate_og_receptive_field(x, y, sigma*sigma_ratio, 
                                                  self.stimulus.deg_x0, self.stimulus.deg_x0) * 1/sigma_ratio**2
        
        # difference
        rf = rf_center - np.sqrt(volume_ratio)*rf_surround
        
        # extract the response
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr0, rf)
        
        # generate the hrf
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # convolve it
        model = fftconvolve(response, hrf)[0:len(response)]
        
        return model
        
    def generate_prediction(self, x, y, sigma, sigma_ratio, volume_ratio):
        
        # extract the center response
        rf_center = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        
        # extract surround response
        rf_surround = generate_og_receptive_field(x, y, sigma*sigma_ratio, 
                                                  self.stimulus.deg_x, self.stimulus.deg_y) * 1/sigma_ratio**2
        
        # difference
        rf = rf_center - np.sqrt(volume_ratio)*rf_surround
        
        # extract the response
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr, rf)
        
        # generate the hrf
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # convolve it
        model = fftconvolve(response, hrf)[0:len(response)]
        
        return model
    
    # DoG receptive field
    def receptive_field(self, x, y, sigma, sigma_ratio, volume_ratio):
            rf_center = generate_og_receptive_field(x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
            rf_surround = generate_og_receptive_field(x, y, sigma*sigma_ratio, self.stimulus.deg_x, self.stimulus.deg_y) * 1.0/sigma_ratio**2                                    
            rf = rf_center - np.sqrt(volume_ratio)*rf_surround
            return rf
    
class DifferenceOfGaussiansFit(PopulationFit):
    
    r"""
    A Difference of Gaussians population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds, Ns,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
        
        r"""
        A class containing tools for fitting the Difference of Gaussians pRF model.
        
        The `DifferenceOfGaussiansFit` class houses all the fitting tool that 
        are associated with estimating a pRF model. The `GaussianFit` takes a 
        `DifferenceOfGaussiansModel` instance  `model` and a time-series `data`. 
        In addition, extent and sampling-rate of a  brute-force grid-search is set 
        with `grids` and `Ns`.  Use `bounds` to set limits on the search space for 
        each parameter.  
        
        Paramaters
        ----------
        
                
        model : `DifferenceOfGaussiansModel` class instance
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
    def sr0(self):
        return self.ballpark[3]
    
    @auto_attr
    def vr0(self):
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
    def sigma_ratio(self):
        return self.estimate[3]
    
    @auto_attr
    def volume_ratio(self):
        return self.estimate[4]
    
    @auto_attr
    def rho(self):
        return np.sqrt(self.x**2+self.y**2)
    
    @auto_attr
    def theta(self):
        return np.mod(np.arctan2(self.y,self.x),2*np.pi)
