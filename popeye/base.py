"""

Base-classes for poulation encoding models and fits.


"""
import time, ctypes
import statsmodels.api as sm 
import numpy as np
from scipy.stats import linregress
from popeye.onetime import auto_attr
import popeye.utilities as utils
import numpy as np

def set_verbose(verbose):
    
    r"""A convenience function for setting the verbosity of a popeye model+fit.
    
    Paramaters
    ----------
    
    verbose : int
        0 = silent
        1 = print the final solution of an error-minimization
        2 = print each error-minimization step
        
    """
    
    # set verbose
    if verbose == 0: # pragma: no cover
        verbose = False
        very_verbose = False
    if verbose == 1: # pragma: no cover
        verbose = True
        very_verbose = False
    if verbose == 2: # pragma: no cover
        verbose = True
        very_verbose = True
    
    return verbose, very_verbose

class PopulationModel(object):
    
    r""" Base class for all pRF models."""
    
    def __init__(self, stimulus, hrf_model, nuisance=None):
        
        r"""Base class for all pRF models.
        
        Paramaters
        ----------
        
        stimulus : `StimulusModel` class instance
            An instance of the `StimulusModel` class containing the `stim_arr` and 
            various stimulus parameters, and can represent various stimulus modalities, 
            including visual, auditory, etc.
        
        hrf_model : callable
            A function that generates an HRF model given an HRF delay.
            For more information, see `popeye.utilties.double_gamma_hrf_hrf
            and `popeye.utilities.spm_hrf`
            
        nuisance : ndarray
            A nuisance regressor for removing effects of non-interest.
            You can regress out any nuisance effects from you data prior to fitting
            the model of interest. The nuisance model is a statsmodels.OLS compatible
            design matrix, and the user is expected to have already added any constants.
        
        """
        
        self.stimulus = stimulus
        self.hrf_model = hrf_model
        self.nuisance = nuisance
    
    def generate_ballpark_prediction(self): # pragma: no cover
        raise NotImplementedError("Each pRF model must implement its own prediction generation!") 
    
    def generate_prediction(self): # pragma: no cover
        raise NotImplementedError("Each pRF model must implement its own prediction generation!")
    
    def distance_mask_coarse(self, x, y, sigma):
        
        if hasattr(self, 'mask_size'): # pragma: no cover
            distance = (self.stimulus.deg_x0 - x)**2 + (self.stimulus.deg_y0 - y)**2
            mask = np.zeros_like(distance, dtype='uint8')
            mask[distance < self.mask_size*sigma**2] = 1
        else: # pragma: no cover
            mask = np.ones_like(self.stimulus.deg_x0, dtype='uint8')
            
        return mask
        
    def distance_mask(self, x, y, sigma):
        
        if hasattr(self, 'mask_size'): # pragma: no cover
            distance = (self.stimulus.deg_x - x)**2 + (self.stimulus.deg_y - y)**2
            mask = np.zeros_like(distance, dtype='uint8')
            mask[distance < self.mask_size*sigma**2] = 1
        else: # pragma: no cover
            mask = np.ones_like(self.stimulus.deg_x, dtype='uint8')
            
        return mask
    
    def hrf(self):
        if hasattr(self, 'hrf_delay'): # pragma: no cover
            return self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        else: # pragma: no cover
            raise NotImplementedError("You must set the HRF delay to generate the HRF")
    
class PopulationFit(object):
    
    
    r""" Base class for all pRF model fits."""
    
    
    def __init__(self, model, data, grids, bounds, 
                 voxel_index=(1,2,3), Ns=None, auto_fit=True, verbose=False):
        
        r"""A class containing tools for fitting pRF models.
        
        The `PopulationFit` class houses all the fitting tool that are associated with 
        estimatinga pRF model.  The `PopulationFit` takes a `PoulationModel` instance 
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
        
        # absorb vars
        self.grids = grids
        self.bounds = bounds
        self.Ns = Ns
        self.voxel_index = voxel_index
        self.auto_fit = auto_fit
        self.model = model
        self.data = data
        self.verbose, self.very_verbose = set_verbose(verbose)
        
        # regress out any nuisance
        if self.model.nuisance is not None:
            self.model.nuisance_model = sm.OLS(self.data,self.model.nuisance)
            self.model.results = self.model.nuisance_model.fit()
            self.original_data = self.data
            self.data = self.model.results.resid
        
        # push data down to model
        # there is probably a better of exposing the data
        # to the PopulationModel. the idea is that the we want
        # compute beta and baseline via linear regression rather
        # than estimate through descent. Thus, model needs to see
        # the data. Since the data is in shared memory, no overhead
        self.model.data = data
        
        # automatic fitting
        if self.auto_fit:
            
            # start
            self.start = time.time()
            
            # init
            self.brute_force
            self.ballpark
            
            # final
            self.gradient_descent
            self.estimate
            self.overloaded_estimate
            
            # finish
            self.finish = time.time()
            
            # performance
            self.rss
            self.rsquared
            
            # flush if not testing
            if not hasattr(self.model, 'store_search_space'): # pragma: no cover
                self.gradient_descent = [None]*6
                self.brute_force =[None,]*4
            
            # print
            if self.verbose: # pragma: no cover
                print(self.msg)
                                                                              
    # the brute search
    @auto_attr
    def brute_force(self):
        return utils.brute_force_search(self.data,
                                        utils.error_function,
                                        self.model.generate_ballpark_prediction,
                                        self.grids,
                                        self.bounds,
                                        self.Ns,
                                        self.very_verbose)
     
     
    @auto_attr
    def ballpark(self):
        return self.brute_force[0]
    
    # the gradient search
    @auto_attr
    def gradient_descent(self):
        
        # this is in case we want to compute baseline and beta
        # parameters via linear regression rather than estimation
        # this should only be used for a grid-search, not fmin
        if self.overloaded_ballpark is not None:
            return utils.gradient_descent_search(self.data,
                                                 utils.error_function,
                                                 self.model.generate_prediction,
                                                 self.overloaded_ballpark,
                                                 self.bounds,
                                                 self.very_verbose)
        else:
            return utils.gradient_descent_search(self.data,
                                                 utils.error_function,
                                                 self.model.generate_prediction,
                                                 self.ballpark,
                                                 self.bounds,
                                                 self.very_verbose)
    
    @auto_attr
    def overloaded_estimate(self):
        return None
    
    @auto_attr
    def overloaded_ballpark(self):
        return None
    
    @auto_attr
    def estimate(self):
        return self.gradient_descent[0]
    
    @auto_attr
    def ballpark_prediction(self):
        return self.model.generate_ballpark_prediction(*self.ballpark)
    
    @auto_attr
    def slope(self):
        return linregress(self.ballpark_prediction, self.data)[0]
    
    @auto_attr
    def intercept(self):
        return linregress(self.ballpark_prediction, self.data)[1]
    
    @auto_attr
    def prediction(self):
        return self.model.generate_prediction(*self.estimate)
    
    @auto_attr
    def rsquared(self):
        return np.corrcoef(self.data,self.prediction)[1][0]**2
    
    @auto_attr
    def rss(self):
        return np.sum((self.data - self.prediction)**2)
    
    @auto_attr
    def msg(self):
        if self.auto_fit is True and self.overloaded_estimate is not None: # pragma: no cover
            txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RSQ=%.02f  EST=%s"
                %(self.voxel_index[0],
                  self.voxel_index[1],
                  self.voxel_index[2],
                  self.finish-self.start,
                  self.rsquared,
                  np.round(self.overloaded_estimate,4)))
        elif self.auto_fit is True: # pragma: no cover
            txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RSQ=%.02f  EST=%s"
                %(self.voxel_index[0],
                  self.voxel_index[1],
                  self.voxel_index[2],
                  self.finish-self.start,
                  self.rsquared,
                  np.round(self.estimate,4)))
        else: # pragma: no cover
            txt = ("VOXEL=(%.03d,%.03d,%.03d)   RSQ=%.02f  EST=%s" 
                %(self.voxel_index[0],
                  self.voxel_index[1],
                  self.voxel_index[2],
                  self.rsquared,
                  np.round(self.estimate,4)))
        return txt
    
class StimulusModel(object):

    def __init__(self, stim_arr, dtype=ctypes.c_int16, tr_length=1.0):
        
        r"""A base class for all encoding stimuli.
        
        This class houses the basic and common features of the encoding stimulus, which
        along with a `PopulationModel` constitutes what is commonly referred to as the 
        pRF model.
        
        Paramaters
        ----------
        
        stim_arr : ndarray
            An array containing the stimulus.  The dimensionality of the data is arbitrary
            but must be consistent with the pRF model, as specified in `PopulationModel` and 
            `PopluationFit` class instances.
        
        dtype : string
            Sets the data type the stimulus array is cast into.
        
        dtype : string
            Sets the data type the stimulus array is cast into.
        
        tr_length : float
            The repetition time (TR) in seconds.
            
        """
        
        self.dtype = dtype
        self.stim_arr = utils.generate_shared_array(stim_arr, self.dtype)
        self.tr_length = tr_length
