"""

Base-classes for poulation encoding models and fits.


"""
import time
import ctypes
import numpy as np
from popeye.onetime import auto_attr
import popeye.utilities as utils
import statsmodels.api as sm

def set_verbose(verbose):
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
    
    """ Abstract class which holds the PopulationModel. """
    
    def __init__(self, stimulus, hrf_model):
        self.stimulus = stimulus
        self.hrf_model = hrf_model
    
    def generate_ballpark_prediction(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def generate_prediction(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
class PopulationFit(object):
    """ Abstract class which holds the PopulationFit
    """
    
    def __init__(self, model, data, grids, bounds, Ns, voxel_index, auto_fit, verbose):
        
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
            
            try:
                
                # start
                self.start = time.clock()
                
                # the business
                self.ballpark
                self.estimate
                self.OLS
                self.rss
                self.rsquared
                
                # finish
                self.finish = time.clock()
                
                # print
                if self.verbose:
                    print(self.msg)
            
            # failsafe
            except:
                pass
            

    
    # the brute search
    @auto_attr
    def ballpark(self):
        return utils.brute_force_search(self.grids,
                                        self.bounds,
                                        self.Ns,
                                        self.data,
                                        utils.error_function,
                                        self.model.generate_ballpark_prediction,
                                        self.very_verbose)
     
    # the gradient search                                  
    @auto_attr
    def estimate(self):
        return utils.gradient_descent_search(self.ballpark,
                                             self.bounds,
                                             self.data,
                                             utils.error_function,
                                             self.model.generate_prediction,
                                             self.very_verbose)
    
    
    @auto_attr
    def prediction(self):
        return self.model.generate_prediction(*self.estimate)
    
    @auto_attr
    def OLS(self):
        return sm.OLS(self.data,self.prediction).fit()
    
    @auto_attr
    def coefficient(self):
        return self.OLS.params[0]
    
    @auto_attr
    def rsquared(self):
        return self.OLS.rsquared
    
    @auto_attr
    def stderr(self):
        return np.sqrt(self.OLS.mse_resid)
    
    @auto_attr
    def rss(self):
        return np.sum((self.data - self.prediction)**2)
    
class StimulusModel(object):
    """ Abstract class which holds the StimulusModel
    """

    def __init__(self, stim_arr, dtype=ctypes.c_int16, tr_length=1.0):
        
        self.dtype = dtype
        self.stim_arr = utils.generate_shared_array(stim_arr, self.dtype)
        self.tr_length = tr_length
