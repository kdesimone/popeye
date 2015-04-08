"""

Base-classes for poulation encoding models and fits.


"""

import numpy as np
import ctypes
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
        
        self.model = model
        self.data = data

class StimulusModel(object):
    """ Abstract class which holds the StimulusModel
    """

    def __init__(self, stim_arr, dtype):
        
        self.dtype = dtype
        self.stim_arr = utils.generate_shared_array(stim_arr, self.dtype)

