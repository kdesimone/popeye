#!/usr/bin/python

""" Classes and functions for fitting Gaussian population encoding models """

from __future__ import division, print_function, absolute_import
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np

from scipy.stats import linregress
from scipy.signal import fft_convolve

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import MakeFastGaussPrediction, MakeFastRF

def compute_model_ts(x, y, sigma, hrf_delay, beta,
                     deg_x, deg_y, stim_arr, tr_length):
    
    
    """
    The objective function for GaussianFi class.
    
    Parameters
    ----------
    x : float
        The model estimate along the horizontal dimensions of the display.

    y : float
        The model estimate along the vertical dimensions of the display.

    sigma : float
        The model estimate of the dispersion across the the display.
    
    hrf_delay : float
        The model estimate of the relative delay of the HRF.  The canonical
        HRF is assumed to be 5 s post-stimulus [1]_.
    
    beta : float
        The model estimate of the amplitude of the BOLD signal.
    
    tr_length : float
        The length of the repetition time in seconds.
    
    
    Returns
    -------
    
    model : ndarray
    The model prediction time-series.
    
    
    References
    ----------
    
    .. [1] Glover, GH. (1999). Deconvolution of impulse response in 
    event-related BOLD fMRI. NeuroImage 9: 416–429.
    
    """
    
    # otherwise generate a prediction
    ts_stim = MakeFastGaussPrediction(deg_x,
                                      deg_y,
                                      stim_arr,
                                      x,
                                      y,
                                      sigma)
    
    # create the HRF
    hrf = utils.double_gamma_hrf(hrf_delay, tr_length, frames_per_tr)
    
    # convolve it with the stimulus
    model = fftconvolve(ts_stim, hrf)[0:len(ts_stim)]
    
    # scale it
    model *= beta
        
    return model

def parallel_fit(args):
    
    """
    This is a convenience function for parallelizing the fitting
    procedure.  Each call is handed a tuple or list containing
    all the necessary inputs for instantiaing a `GaussianFit`
    class object and estimating the model parameters.
    
    
    Paramaters
    ----------
    args : list/tuple
        A list or tuple containing all the necessary inputs for fitting
        the Gaussian pRF model.
    
    Returns
    -------
    
    fit : `GaussianFit` class object
        A fit object that contains all the inputs and outputs of the 
        Gaussian pRF model estimation for a single voxel.
    
    """
    
    
    # unpackage the arguments
    data = args[0]
    model = args[1]
    search_bounds = args[2]
    fit_bounds = args[3]
    tr_length = args[4]
    voxel_index = args[5]
    auto_fit = args[6]
    verbose = args[7]
    
    
    # fit the data
    fit = GaussianFit(response,
                      model,
                      search_bounds,
                      fit_bounds,
                      tr_length,
                      voxel_index,
                      auto_fit,
                      verbose)
    return fit


class GaussianModel(PopulationModel):
    
    """
    A Gaussian population receptive field model class
    
    """
    
    def __init__(self, stimulus):
        
        """
        A Gaussian population receptive field model [1]_.
        
        Paramaters
        ----------
        
        stimulus : `VisualStimulus` class object
            A class instantiation of the `VisualStimulus` class
            containing a representation of the visual stimulus.
        
        
        References
        ----------
        
        .. [1] Dumoulin SO, Wandell BA. (2008) Population receptive field 
        estimates in human visual cortex. NeuroImage 39:647-660
        
        """
        
        # this is a weird notation
        PopulationModel.__init__(self, stimulus)
        
class GaussianFit(object):
    
    """
    A Gaussian population receptive field fit class
    
    """
    
    def __init__(self, data, model, search_bounds, fit_bounds, tr_length,
                 voxel_index=None, auto_fit=True, verbose=True):
        
        
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
        
        self.data = data
        self.model = model
        self.search_bounds = search_bounds
        self.fit_bounds = fit_bounds
        self.tr_length = tr_length
        self.voxel_index = voxel_index
        self.auto_fit = auto_fit
        self.verbose = verbose
        
        if self.auto_fit:
            tic = time.clock()
            self.ballpark_estimate;
            self.estimate;
            toc = time.clock()
            
            # print to screen if verbose
            if self.verbose:
                
                # if no voxel index is specified we need something for the print
                if self.voxel_index is None:
                    self.voxel_index = (0,0,0)
                
                # print
                print("VOXEL=(%.03d,%.03d,%.03d)  TIME=%.03d  ERROR=%.03d  RVAL=%.02f" 
                      %(self.voxel_index[0],
                        self.voxel_index[1],
                        self.voxel_index[2],
                        toc-tic,
                        self.rss,
                        self.fit_stats[2]))
                    
        
    @auto_attr
    def ballpark_estimate(self):
        return utils.brute_force_search((self.model.stimulus.deg_x_coarse,
                                         self.model.stimulus.deg_y_coarse,
                                         self.model.stimulus.stim_arr_coarse,
                                         self.tr_length,
                                         self.model.stimulus.frames_per_tr),
                                        self.search_bounds,
                                        self.fit_bounds,
                                        self.data,
                                        utils.error_function,
                                        compute_model_ts)
                                  
                                  
    @auto_attr
    def estimate(self):
        return utils.gradient_descent_search((self.x0, self.y0, self.s0, self.hrf0, self.beta0),
                                             (self.model.stimulus.deg_x,
                                              self.model.stimulus.deg_y,
                                              self.model.stimulus.stim_arr,
                                              self.tr_length,
                                              self.model.stimulus.frames_per_tr),
                                             self.fit_bounds,
                                             self.data,
                                             utils.error_function,
                                             compute_model_ts)
 
                                       
    @auto_attr
    def x0(self):
        return self.ballpark_estimate[0]
        
    @auto_attr
    def y0(self):
        return self.ballpark_estimate[1]
        
    @auto_attr
    def s0(self):
        return self.ballpark_estimate[2]
        
    @auto_attr
    def hrf0(self):
        return self.ballpark_estimate[3]
    
    @auto_attr
    def beta0(self):
        return self.ballpark_estimate[4]
        
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
    def beta(self):
        return self.estimate[4]
        
    @auto_attr
    def prediction(self):
        return compute_model_ts(self.x, self.y, self.sigma, self.hrf_delay, self.beta,
                                self.model.stimulus.deg_x,
                                self.model.stimulus.deg_y,
                                self.model.stimulus.stim_arr,
                                self.tr_length,
                                self.model.stimulus.frames_per_tr)
    
    @auto_attr
    def fit_stats(self):
        return linregress(self.data, self.prediction)
    
    @auto_attr
    def rss(self):
        return np.sum((self.data - self.prediction)**2)
    
    @auto_attr
    def receptive_field(self):
        return MakeFastRF(self.model.stimulus.deg_x,
                          self.model.stimulus.deg_y,
                          self.x, self.y, self.sigma)
