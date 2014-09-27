"""

Base-classes for poulation encoding models and fits.


"""

import numpy as np


class PopulationModel(object):
    """ Abstract class which holds the stimulus model
    """
    
    def __init__(self, stimulus):
        self.stimulus = stimulus

class PopulationFit(object):
    """ Abstract class which holds the fit result of PopulationModel
    """
    
    def __init__(self, model, data):
        self.data = data
        self.model = model

class StimulusModel(object):
    """ Abstract class which holds the stimulus model
    """

    def __init__(self, stim_arr):
        pass
        
        # self.stim_arr = stim_arr
        
        # share the arrays via memmap to reduce size
        self.stim_arr = np.memmap('%s%s_%d.npy' %('/tmp/','stim_arr',self.__hash__()),dtype = np.double, mode = 'w+',shape = np.shape(stim_arr))
        self.stim_arr[:] = stim_arr[:]
        stim_arr = []