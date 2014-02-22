"""

Base-classes for poulation encoding models and fits.


"""


class PopulationModel(object):
    """ Abstract class for signal reconstruction models
    """
    
    def __init__(self, stimulus):
        """

        """
        self.stimulus = stimulus

    def fit(self, data, mask=None,**kwargs):
        
        return PopulationFit(self, data, mask)

class PopulationFit(object):
    """ Abstract class which holds the fit result of PopulationModel
    """
    
    def __init__(self, model, data):
        self.data = data
        self.model = model
