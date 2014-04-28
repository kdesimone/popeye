"""

Base-classes for poulation encoding models and fits.


"""


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
