#!/usr/bin/python

""" Classes and functions for fitting population encoding models """

from __future__ import division
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.stats import linregress
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_rf_timeseries_1D

class AuditoryModel(PopulationModel):
    
    def __init__(self, stimulus, hrf_model):
        
        r"""A 1D Gaussian population receptive field model [1]_.
        
        Paramaters
        ----------
        
        stimulus : `AuditoryStimulus` class object
            A class instantiation of the `AuditoryStimulus` class
            containing a representation of the auditory stimulus.
            
        hrf_model : callable
            A function that generates an HRF model given an HRF delay.
            For more information, see `popeye.utilties.double_gamma_hrf_hrf`
        
        
        References
        ----------
        
        .. [1] Thomas JM, Huber E, Stecker GC, Boynton GM, Saenz M, Fine I. (2015) 
        Population receptive field estimates in human auditory cortex. 
        NeuroImage 105, 428-439.
        
        """
        
        # invoke the base class
        PopulationModel.__init__(self, stimulus, hrf_model)
    
    
    def generate_prediction(self, center_freq, sigma, beta, baseline):
        
        r"""
        Generate a prediction for the 1D Gaussian model.
        
        This function generates a prediction of the 1D Gaussian model, 
        given a stimulus and the stimulus-referred model parameters.  This
        function operates on the native stimulus.
        
        Paramaters
        ----------
        
        center_freq : float
            The center frequency of the 1D Gaussian, units are in Hz.
            
        sigma : float
            The dispersion of the 1D Gaussian, units are in Hz.
        
        beta : float
            The scaling factor to account for arbitrary units of the BOLD signal.
        
        hrf_delay : float
            The delay of the HRF, units are in seconds.
        
        """ 
        
        # receptive field
        rf = np.exp(-((10**self.stimulus.freqs-10**center_freq)**2)/(2*(10**sigma)**2))
        rf /= (10**sigma*np.sqrt(2*np.pi))
        
        # # create mask for speed
        # distance = self.stimulus.freqs - center_freq
        # mask = np.zeros_like(distance, dtype='uint8')
        # mask[distance < (self.mask_size*sigma)] = 1
        mask = np.ones_like(rf).astype('uint8')
        
        # extract the response
        response = generate_rf_timeseries_1D(self.stimulus.spectrogram, rf, mask)
        
        # convolve it with the stimulus
        model = fftconvolve(response, self.hrf())[0:len(response)]
        
        # units
        model = (model - np.mean(model)) / np.mean(model)
        
        # offset
        model += baseline
        
        # scale it
        model *= beta
        
        return model
    
    def generate_ballpark_prediction(self, center_freq, sigma):
        
        r"""
        Generate a prediction for the 1D Gaussian model.
        
        This function generates a prediction of the 1D Gaussian model, 
        given a stimulus and the stimulus-referred model parameters.  This
        function operates on the native stimulus.  Usually, the function
        `generate_ballpark_prediction` would operate on the downsampled
        stimulus.
        
        Paramaters
        ----------
        
        center_freq : float
            The center frequency of the 1D Gaussian, units are in Hz.
            
        sigma : float
            The dispersion of the 1D Gaussian, units are in Hz.
        
        beta : float
            The scaling factor to account for arbitrary units of the BOLD signal.
        
        hrf_delay : float
            The delay of the HRF, units are in seconds.
        
        """
        
        # receptive field
        rf = np.exp(-((10**self.stimulus.freqs-10**center_freq)**2)/(2*(10**sigma)**2))
        rf /= (10**sigma*np.sqrt(2*np.pi))
        
        # # create mask for speed
        # distance = self.stimulus.freqs - center_freq
        # mask = np.zeros_like(distance, dtype='uint8')
        # mask[distance < (self.mask_size*sigma)] = 1
        mask = np.ones_like(rf).astype('uint8')
        
        # extract the response
        response = generate_rf_timeseries_1D(self.stimulus.spectrogram, rf, mask)
        
        # convolve it with the stimulus
        model = fftconvolve(response, self.hrf())[0:len(response)]
        
        # units
        model = (model - np.mean(model)) / np.mean(model)
        
        # regress to find beta and baseline
        p = linregress(model, self.data)
        
        # offset
        model += p[1]
        
        # scale it
        model *= np.abs(p[0])
        
        return model
        
        
class AuditoryFit(PopulationFit):
    
    def __init__(self, model, data, grids, bounds,
                 voxel_index=(1,2,3), Ns=None, auto_fit=True, verbose=0):
        
        r"""
        A class containing tools for fitting the 1D Gaussian pRF model.
        
        The `AuditoryFit` class houses all the fitting tool that are associated with 
        estimatinga pRF model.  The `PopulationFit` takes a `AuditoryModel` instance 
        `model` and a time-series `data`.  In addition, extent and sampling-rate of a 
        brute-force grid-search is set with `grids` and `Ns`.  Use `bounds` to set 
        limits on the search space for each parameter.  
        
        Paramaters
        ----------
        
                
        model : `AuditoryModel` class instance
            An object representing the 1D Gaussian model.
        
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
        
        # invoke the base class
        PopulationFit.__init__(self, model, data, grids, bounds, 
                               voxel_index, Ns, auto_fit, verbose)
                               
    # overload ballpark
    # with the linregress slope
    # as a sub-stitute for beta
    @auto_attr
    def overloaded_ballpark(self):
        return np.append(self.ballpark, (self.beta0, self.baseline0))
    
    @auto_attr
    def overloaded_estimate(self):
        return [10**self.center_freq, 10**self.sigma, self.beta, self.baseline]
    
    @auto_attr
    def center_freq0(self):
        return self.ballpark[0]
        
    @auto_attr
    def sigma0(self):
        return self.ballpark[1]
        
    @auto_attr
    def beta0(self):
        return np.abs(self.slope)

    @auto_attr
    def baseline0(self):
        return self.intercept
    
    @auto_attr
    def center_freq(self):
        return self.estimate[0]
        
    @auto_attr
    def sigma(self):
        return self.estimate[1]
        
    @auto_attr
    def beta(self):
        return self.estimate[2]
    
    @auto_attr
    def baseline(self):
        return self.estimate[3]
    
    
    @auto_attr
    def center_freq_hz(self):
        return 10**self.center_freq
    
    @auto_attr
    def receptive_field(self):
        
        # generate stimulus time-series
        rf = np.exp(-((10**self.model.stimulus.freqs-10**self.center_freq)**2)/(2*(10**self.sigma)**2))
        rf /= (10**self.sigma*np.sqrt(2*np.pi))
        return rf
    
    @auto_attr
    def receptive_field_log10(self):
            
        # generate stimulus time-series
        rf = np.exp(-((self.model.stimulus.freqs-self.center_freq)**2)/(2*self.sigma**2))
        rf /= (self.sigma*np.sqrt(2*np.pi))
        return rf
       