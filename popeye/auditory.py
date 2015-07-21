#!/usr/bin/python

""" Classes and functions for fitting population encoding models """

from __future__ import division
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import statsmodels.api as sm
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_rf_timeseries_1D

class AuditoryModel(PopulationModel):
    
    """
    1D Gaussian population receptive field model.
    """
    
    def __init__(self, stimulus, hrf_model):
        
        """
        A 1D Gaussian population receptive field model [1]_.
        
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
        NeuroImage 105:428-439.
        
        """
        
        # invoke the base class
        PopulationModel.__init__(self, stimulus, hrf_model)
    
    def generate_prediction(self, center_freq, sigma, beta, hrf_delay):
        
        """
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
        rf = np.exp(-((self.stimulus.freqs-center_freq)**2)/(2*sigma**2))
        rf /= (sigma*np.sqrt(2*np.pi))
        rf *= beta
        
        # if the rf runs off the coords
        if np.round(rf[0],3) != 0:
            return np.inf
            
        # create mask for speed
        distance = self.stimulus.freqs - center_freq
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (5*sigma)] = 1
        
        # extract the response
        response = generate_rf_timeseries_1D(self.stimulus.spectrogram, rf, mask)
        
        # resample the time-series to the BOLD 
        source_times = np.linspace(0, 1, len(response), endpoint=True)
        f = interp1d(source_times, response, kind='linear')
        num_volumes = self.stimulus.stim_arr.shape[0]/self.stimulus.Fs/self.stimulus.tr_length
        target_times = np.linspace(0, 1, num_volumes, endpoint=True)
        resampled_response = f(target_times)
        
        # generate the HRF
        hrf = utils.double_gamma_hrf(hrf_delay, self.stimulus.tr_length)
        
        # pad and convolve
        model = fftconvolve(resampled_response, hrf, 'same')
        
        return model
    
    def generate_ballpark_prediction(self, center_freq, sigma, beta, hrf_delay):
        
        """
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
        
        return self.generate_prediction(center_freq, sigma, beta, hrf_delay)

class AuditoryFit(PopulationFit):
    
    def __init__(self, model, data, grids, bounds, Ns,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
        
        """
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
        PopulationFit.__init__(self, model, data, grids, bounds, Ns, voxel_index, auto_fit, verbose)
        
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
    def hrf0(self):
        return self.ballpark[3]
    
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
    def hrf_delay(self):
        return self.estimate[3]
    
    
    @auto_attr
    def msg(self):
        txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RSQ=%.02f  CENTER=%.02d   SIGMA=%.02f   BETA=%.08f   HRF=%.02f" 
            %(self.voxel_index[0],
              self.voxel_index[1],
              self.voxel_index[2],
              self.finish-self.start,
              self.rsquared,
              self.center_freq,
              self.sigma,
              self.beta,
              self.hrf_delay))
        return txt
