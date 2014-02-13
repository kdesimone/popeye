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
        return PopulationModel(self, data)

class PopulationFit(object):
    """ Abstract class which holds the fit result of PopulationModel

    For example that could be holding FA or GFA etc.
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data
