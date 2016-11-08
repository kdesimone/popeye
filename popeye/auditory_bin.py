#!/usr/bin/python

""" Classes and functions for fitting population encoding models """

from __future__ import division
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_rf_timeseries_1D, binner

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
    
    def generate_prediction(self, center_freq, sigma, beta, baseline, hrf_delay):
        
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
        
        # generate stimulus time-series
        rf = self.receptive_field(center_freq, sigma)
        
        # if the tuning curve is running off the edges of the frequency space
        if self.compare_rf_volumes(rf):
            return np.inf
        
        # create mask for speed
        distance = self.stimulus.freqs - center_freq
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (5*sigma)] = 1
        
        # extract the response
        response = generate_rf_timeseries_1D(self.stimulus.spectrogram, rf, mask)
        
        # bin the response to the resolution of the BOLD data
        binned_response = self.binned_response(response)
        
        # roll it forward to correct for the binning
        rolled_response = np.roll(binned_response, 0)
        
        # generate the HRF
        hrf = utils.double_gamma_hrf(hrf_delay, self.stimulus.tr_length)
        
        # pad and convolve
        model = fftconvolve(rolled_response, hrf)[0:len(rolled_response)]
        
        # scale by beta
        model *= beta
        
        # add the offset
        model += baseline
        
        return model
    
    def generate_ballpark_prediction(self, center_freq, sigma, beta, baseline, hrf_delay):
        
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
        
        return self.generate_prediction(center_freq, sigma, beta, baseline, hrf_delay)
        
    # seconds
    @auto_attr
    def length_in_seconds(self):
        return np.ceil(self.stimulus.times[-1])
    
    # volumes
    @auto_attr
    def length_in_volumes(self):
        return self.length_in_seconds / self.stimulus.tr_length
    
    # convenience function
    def receptive_field(self, center_freq, sigma):
        return np.exp(-((self.stimulus.freqs-center_freq)**2)/(2*sigma**2)) * 1/(sigma * np.sqrt(2*np.pi))
    
    # Reference RF.  We always compare the volume of the candidate RF with this one.
    # If candidate RF volume is more than 50% smaller in volume, we return inf above ...
    @auto_attr
    def reference_rf(self):
        return self.receptive_field(self.stimulus.freqs[-1]/4,self.stimulus.freqs[-1]/20)
    
    # convenience function
    def compare_rf_volumes(self, rf):
        if (1-np.sum(rf)/np.sum(self.reference_rf))*100 > 50:
            return True
        else:
            return False
    
    # bin stimulus to BOLD
    def binned_response(self, response):
        
        # initialize empty array
        b = np.zeros(self.length_in_volumes)
        
        # bin it
        counter = 0
        for i in np.arange(0,self.length_in_seconds,self.stimulus.tr_length):
            idx = np.nonzero((self.stimulus.times < i+1) & (self.stimulus.times > i))[0]
            b[counter] = np.sum(response[idx[0]:idx[-1]])
            counter += 1
        
        return b
        
class AuditoryFit(PopulationFit):
    
    def __init__(self, model, data, grids, bounds, Ns,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
        
        PopulationFit.__init__(self, model, data, grids, bounds, Ns, 
                               voxel_index, auto_fit, verbose)
        
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
            
    @auto_attr
    def center_freq0(self):
        return self.ballpark[0]
    
    @auto_attr
    def sigma0(self):
        return self.ballpark[1]
    
    @auto_attr
    def beta0(self):
        return self.ballpark[2]
    
    @auto_attr
    def baseline0(self):
        return self.ballpark[3]
    
    @auto_attr
    def hrf0(self):
        return self.ballpark[4]
    
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
    def hrf_delay(self):
        return self.estimate[4]
        