"""

Base-classes for poulation encoding models and fits.


"""
import time
import ctypes
import numpy as np
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
    if verbose == 0:
        verbose = False
        very_verbose = False
    if verbose == 1:
        verbose = True
        very_verbose = False
    if verbose == 2:
        verbose = True
        very_verbose = True
    
    return verbose, very_verbose

class PopulationModel(object):
    
    r""" Base class for all pRF models."""
    
    def __init__(self, stimulus, hrf_model):
        
        r"""Base class for all pRF models.
        
        Paramaters
        ----------
        
        stimulus : `StimulusModel` class instance
            An instance of the `StimulusModel` class containing the `stim_arr` and 
            various stimulus parameters, and can represent various stimulus modalities, 
            including visual, auditory, etc.
        
        hrf_model : callable
            A function that generates an HRF model given an HRF delay.
            For more information, see `popeye.utilties.double_gamma_hrf_hrf`
        
        """
        
        self.stimulus = stimulus
        self.hrf_model = hrf_model
    
    def generate_ballpark_prediction(self):
        raise NotImplementedError("Each pRF model must implement its own prediction generation!")
    
    def generate_prediction(self):
        raise NotImplementedError("Each pRF model must implement its own prediction generation!")
    
class PopulationFit(object):
    
    
    r""" Base class for all pRF model fits."""
    
    
    def __init__(self, model, data, grids, bounds, Ns, voxel_index, auto_fit, verbose):
        
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
            if self.very_verbose == False:
                self.gradient_descent = [None]*6
                self.brute_force =[None,]*4
            
            # print
            if self.verbose:
                print(self.msg)
                                                                              
    # the brute search
    @auto_attr
    def brute_force(self):
        return utils.brute_force_search(self.grids,
                                        self.bounds,
                                        self.Ns,
                                        self.data,
                                        utils.error_function,
                                        self.model.generate_ballpark_prediction,
                                        self.very_verbose)
     
     
    @auto_attr
    def ballpark(self):
        return self.brute_force[0]

    @auto_attr
    def fval(self):
        return self.brute_force[1]
    
    @auto_attr
    def grid(self):
        return self.brute_force[2]
    
    @auto_attr
    def Jout(self):
        return self.brute_force[3]
    
    # the gradient search
    @auto_attr
    def gradient_descent(self):
        return utils.gradient_descent_search(self.ballpark,
                                             self.bounds,
                                             self.data,
                                             utils.error_function,
                                             self.model.generate_prediction,
                                             self.very_verbose)
    
    @auto_attr
    def overloaded_estimate(self):
        return None
        
    @auto_attr
    def estimate(self):
        return self.gradient_descent[0]
    
    @auto_attr
    def fopt(self):
        return self.gradient_descent[1]
    
    @auto_attr
    def direc(self):
        return self.gradient_descent[2]
    
    @auto_attr
    def iter(self):
        return self.gradient_descent[3]
    
    @auto_attr
    def funcalls(self):
        return self.gradient_descent[4]
    
    @auto_attr
    def allvecs(self):
        return self.gradient_descent[5]
    
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
        if self.auto_fit is True and self.overloaded_estimate is not None:
            txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RSQ=%.02f  EST=%s"
                %(self.voxel_index[0],
                  self.voxel_index[1],
                  self.voxel_index[2],
                  self.finish-self.start,
                  self.rsquared,
                  np.round(self.overloaded_estimate,4)))
        elif self.auto_fit is True:
            txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RSQ=%.02f  EST=%s"
                %(self.voxel_index[0],
                  self.voxel_index[1],
                  self.voxel_index[2],
                  self.finish-self.start,
                  self.rsquared,
                  np.round(self.estimate,4)))
        else:
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
