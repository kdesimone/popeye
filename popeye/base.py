"""

Base-classes for poulation encoding models and fits.


"""
import time
import ctypes
import numpy as np
from popeye.onetime import auto_attr
import popeye.utilities as utils

class PopulationModel(object):
    """ Abstract class which holds the PopulationModel
    """
    
    def __init__(self, stimulus):
        self.stimulus = stimulus

class PopulationFit(object):
    """ Abstract class which holds the PopulationFit
    """
    
    def __init__(self, model, data, grids, bounds, Ns, tr_length, voxel_index, auto_fit, verbose):
        
        # absorb vars
        self.grids = grids
        self.bounds = bounds
        self.Ns = Ns
        self.tr_length = tr_length
        self.voxel_index = voxel_index
        self.auto_fit = auto_fit
        self.model = model
        self.data = data
        
        # set verbose
        if verbose == 0:
            self.verbose = False
            self.very_verbose = False
        if verbose == 1:
            self.verbose = True
            self.very_verbose = False
        if verbose == 2:
            self.verbose = True
            self.very_verbose = True
        
        # automatic fitting
        if self.auto_fit:
            
            # the business
            self.start = time.clock()
            self.ballpark;
            self.estimate;
            self.OLS;
            self.rss;
            self.finish = time.clock()
            
            # print
            if self.verbose:
                print(self.msg)
    
    # the brute search
    @auto_attr
    def ballpark(self):
        return utils.brute_force_search(self.grids,
                                        self.bounds,
                                        self.Ns,
                                        self.data,
                                        utils.error_function,
                                        self.generate_prediction,
                                        self.very_verbose)
     
    # the gradient search                                  
    @auto_attr
    def estimate(self):
        return utils.gradient_descent_search(self.ballpark,
                                             self.bounds,
                                             self.data,
                                             utils.error_function,
                                             self.generate_prediction,
                                             self.very_verbose)
 
    

class StimulusModel(object):
    """ Abstract class which holds the StimulusModel
    """

    def __init__(self, stim_arr, dtype):
        
        self.dtype = dtype
        self.stim_arr = utils.generate_shared_array(stim_arr, self.dtype)

