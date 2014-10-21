"""

Base-classes for poulation encoding models and fits.


"""

import numpy as np
import ctypes
import sharedmem

class PopulationModel(object):
    """ Abstract class which holds the PopulationModel
    """
    
    def __init__(self, stimulus):
        self.stimulus = stimulus

class PopulationFit(object):
    """ Abstract class which holds the PopulationFit
    """
    
    def __init__(self, model, data):
        self.data = data
        self.model = model

class StimulusModel(object):
    """ Abstract class which holds the StimulusModel
    """

    def __init__(self, stim_arr):
        
        self.stim_arr  = sharedmem.empty(stim_arr.shape, dtype='int16')
        self.stim_arr[:] = stim_arr[:]

