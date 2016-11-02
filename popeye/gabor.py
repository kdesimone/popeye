#!/usr/bin/python

""" Classes and functions for fitting population encoding models """

from __future__ import division
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.signal import fftconvolve
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_gabor_receptive_field, generate_rf_timeseries

class GaborModel(PopulationModel):
    
    r"""
    Gabor population receptive field model.
    
    """
    
    def __init__(self, stimulus, hrf_model):
        
        # this is a weird notation
        PopulationModel.__init__(self, stimulus, hrf_model)
        
        
    def generate_ballpark_prediction(self, x, y, sigma, theta, phi, cpd):
        
        # make sure theta and phi are 0-2*pi
        theta = np.mod(theta, 2*np.pi)
        phi = np.mod(theta, 2*np.pi)
        
        # create mask for speed
        distance = (self.stimulus.deg_x0 - x)**2 + (self.stimulus.deg_y0 - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (5*sigma)**2] = 1
        
        # generate the RF
        rf = generate_gabor_receptive_field(x, y, sigma, theta, phi, cpd,
                                            self.stimulus.deg_x0,
                                            self.stimulus.deg_y0)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries(self.stimulus.stim_arr0, rf, mask)
        
        # convolve with the HRF
        hrf = self.hrf_model(0, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf, 'same')
        
        return model
    
    
    def generate_prediction(self, x, y, sigma, theta, phi, cpd):
        
        # make sure theta and phi are 0-2*pi
        theta = np.mod(theta, 2*np.pi)
        phi = np.mod(theta, 2*np.pi)
        
        # create mask for speed
        distance = (self.stimulus.deg_x - x)**2 + (self.stimulus.deg_y - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (5*sigma)**2] = 1
        
        # generate the RF
        rf = generate_gabor_receptive_field(x, y, sigma, theta, phi, cpd,
                                            self.stimulus.deg_x, self.stimulus.deg_y)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries(self.stimulus.stim_arr, rf, mask)
        
        # convolve with the HRF
        hrf = self.hrf_model(0, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf, 'same')
        
        return model
            
class GaborFit(PopulationFit):
    
    r"""
    A class containing tools for fitting the 1D Gaussian pRF model.
    
    """
    
    def __init__(self, model, data, grids, bounds, Ns,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
                 
        r"""
        A class containing tools for fitting the Difference of Gaussians pRF model.
        
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
    def hrf0(self):
        return self.ballpark[3]
    
    @auto_attr
    def theta0(self):
        return self.ballpark[4]
    
    @auto_attr
    def phi0(self):
        return self.ballpark[5]
            
    @auto_attr
    def cpd0(self):
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
    def hrf_delay(self):
        return self.estimate[3]
    
    @auto_attr
    def theta(self):
        return self.estimate[4]
        
    @auto_attr
    def phi(self):
        return self.estimate[5]
            
    @auto_attr
    def cpd(self):
        return self.estimate[6]
        
    @auto_attr
    def receptive_field(self):
        return generate_gabor_receptive_field(*self.estimate)
    
