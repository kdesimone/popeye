#!/usr/bin/python

""" Classes and functions for fitting Gaussian population encoding models """

from __future__ import division
import time
import gc
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.signal import fftconvolve
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_ogb_receptive_field, generate_og_receptive_field, generate_rf_timeseries

def recast_estimation_results(output, grid_parent, polar=False):
    """
    Recasts the output of the prf estimation into two nifti_gz volumes.
    
    Takes `output`, a list of multiprocessing.Queue objects containing the
    output of the prf estimation for each voxel.  The prf estimates are
    expressed in both polar and Cartesian coordinates.  If the default value
    for the `write` parameter is set to False, then the function returns the
    arrays without writing the nifti files to disk.  Otherwise, if `write` is
    True, then the two nifti files are written to disk.
    
    Each voxel contains the following metrics: 
    
        0 x / polar angle
        1 y / eccentricity
        2 sigma
        3 HRF delay
        4 RSS error of the model fit
        5 correlation of the model fit
        
    Parameters
    ----------
    output : list
        A list of PopulationFit objects.
    grid_parent : nibabel object
        A nibabel object to use as the geometric basis for the statmap.  
        The grid_parent (x,y,z) dim and pixdim will be used.
        
    Returns
    ------ 
    cartes_filename : string
        The absolute path of the recasted prf estimation output in Cartesian
        coordinates. 
    plar_filename : string
        The absolute path of the recasted prf estimation output in polar
        coordinates. 
        
    """
    
    
    # load the gridParent
    dims = list(grid_parent.shape)
    dims = dims[0:3]
    dims.append(9)
    
    # initialize the statmaps
    estimates = np.zeros(dims)
    
    # extract the prf model estimates from the results queue output
    for fit in output:
        
        if polar:
            estimates[fit.voxel_index] = (fit.theta,
                                          fit.rho,
                                          fit.sigma,
                                          fit.beta,
                                          fit.hrf_delay,
                                          fit.baseline,
                                          fit.rsquared,
                                          fit.coefficient,
                                          fit.stderr)
        else:
            estimates[fit.voxel_index] = (fit.x, 
                                          fit.y,
                                          fit.sigma,
                                          fit.beta,
                                          fit.hrf_delay,
                                          fit.baseline,
                                          fit.rsquared,
                                          fit.coefficient,
                                          fit.stderr)
                                          
                                 
    # get header information from the gridParent and update for the prf volume
    aff = grid_parent.get_affine()
    hdr = grid_parent.get_header()
    hdr.set_data_shape(dims)
    
    # recast as nifti
    nifti_estimates = nibabel.Nifti1Image(estimates,aff,header=hdr)
    
    return nifti_estimates
    
def parallel_fit(args):
    
    """
    This is a convenience function for parallelizing the fitting
    procedure.  Each call is handed a tuple or list containing
    all the necessary inputs for instantiaing a `GaussianFit`
    class object and estimating the model parameters.
    
    
    Paramaters
    ----------
    args : list/tuple
        A list or tuple containing all the necessary inputs for fitting
        the Gaussian pRF model.
    
    Returns
    -------
    
    fit : `GaussianFit` class object
        A fit object that contains all the inputs and outputs of the 
        Gaussian pRF model estimation for a single voxel.
    
    """
    
    
    # unpackage the arguments
    model = args[0]
    data = args[1]
    grids = args[2]
    bounds = args[3]
    Ns = args[4]
    voxel_index = args[5]
    auto_fit = args[6]
    verbose = args[7]
    
    # fit the data
    fit = GaussianFit(model,
                      data,
                      grids,
                      bounds,
                      Ns,
                      voxel_index,
                      auto_fit,
                      verbose)
    return fit


class GaussianModel(PopulationModel):
    
    """
    A Gaussian population receptive field model class
    
    """
    
    def __init__(self, stimulus, hrf_model):
        
        """
        A Gaussian population receptive field model [1]_.
        
        Paramaters
        ----------
        
        stimulus : `VisualStimulus` class object
            A class instantiation of the `VisualStimulus` class
            containing a representation of the visual stimulus.
        
        
        References
        ----------
        
        .. [1] Dumoulin SO, Wandell BA. (2008) Population receptive field 
        estimates in human visual cortex. NeuroImage 39:647-660
        
        """
        
        PopulationModel.__init__(self, stimulus, hrf_model)
    
    
    # main method for deriving model time-series
    def generate_ballpark_prediction(self, x, y, sigma, beta, baseline, hrf_delay):
        
        # create mask for speed
        distance = (self.stimulus.deg_x_coarse - x)**2 + (self.stimulus.deg_y_coarse - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (5*sigma)**2] = 1
        
        # generate the RF
        rf = generate_og_receptive_field(x, y, sigma,
                                         self.stimulus.deg_x_coarse,
                                         self.stimulus.deg_y_coarse)
        
        # normalize by the integral
        rf /= (2 * np.pi * sigma**2)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries(self.stimulus.stim_arr_coarse, rf, mask)
        
        # convolve with the HRF
        hrf = self.hrf_model(hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf, 'same')
        
        # scale it by beta
        model *= beta
        
        # add the baseline
        model += baseline
        
        return model
        
    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, beta, baseline, hrf_delay):
        
        # create mask for speed
        distance = (self.stimulus.deg_x - x)**2 + (self.stimulus.deg_y - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (5*sigma)**2] = 1
        
        # generate the RF
        rf = generate_og_receptive_field(x, y, sigma,
                                         self.stimulus.deg_x,
                                         self.stimulus.deg_y)
        
        # normalize by the integral
        rf /= (2 * np.pi * sigma**2)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries(self.stimulus.stim_arr, rf, mask)
        
        # convolve with the HRF
        hrf = self.hrf_model(hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf, 'same')
        
        # scale it by beta
        model *= beta
        
        # add the baseline
        model += baseline
        
        return model
    
class GaussianFit(PopulationFit):
    
    """
    A Gaussian population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds, Ns,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
        
        
        """
        A Gaussian population receptive field model [1]_.

        Paramaters
        ----------
        
        data : ndarray
            An array containing the measured BOLD signal.
        
        model : `GaussianModel` class instance containing the representation
            of the visual stimulus.
        
        search_bounds : tuple
            A tuple indicating the search space for the brute-force grid-search.
            The tuple contains pairs of upper and lower bounds for exploring a
            given dimension.  For example `fit_bounds=((-10,10),(0,5),)` will
            search the first dimension from -10 to 10 and the second from 0 to 5.
            These values cannot be None. 
            
            For more information, see `scipy.optimize.brute`.
        
        fit_bounds : tuple
            A tuple containing the upper and lower bounds for each parameter
            in `parameters`.  If a parameter is not bounded, simply use
            `None`.  For example, `fit_bounds=((0,None),(-10,10),)` would 
            bound the first parameter to be any positive number while the
            second parameter would be bounded between -10 and 10.
        
        voxel_index : tuple
            A tuple containing the index of the voxel being modeled. The 
            fitting procedure does not require a voxel index, but 
            collating the results across many voxels will does require voxel
            indices. With voxel indices, the brain volume can be reconstructed 
            using the newly computed model estimates.
        
        auto-fit : bool
            A flag for automatically running the fitting procedures once the 
            `GaussianFit` object is instantiated.
        
        verbose : bool
            A flag for printing some summary information about the model estiamte
            after the fitting procedures have completed.
        
        References
        ----------
        
        .. [1] Dumoulin SO, Wandell BA. (2008) Population receptive field 
        estimates in human visual cortex. NeuroImage 39:647-660

        """
        
        PopulationFit.__init__(self, model, data, grids, bounds, Ns, 
                               voxel_index, auto_fit, verbose)
    
    @auto_attr
    def x0(self):
        return self.ballpark[0]
        
    @auto_attr
    def y0(self):
        return self.ballpark[1]
        
    @auto_attr
    def s0(self):
        return self.ballpark[2]
    
    @auto_attr
    def beta0(self):
        return self.ballpark[3]
    
    @auto_attr
    def baseline0(self):
        return self.ballpark[4]
    
    @auto_attr
    def hrf0(self):
        return self.ballpark[5]
        
    @auto_attr
    def x(self):
        return self.estimate[0]
        
    @auto_attr
    def y(self):
        return self.estimate[1]
        
    @auto_attr
    def sigma(self):
        return self.estimate[2]
    
    @auto_attr
    def beta(self):
        return self.estimate[3]
    
    @auto_attr
    def baseline(self):
        return self.estimate[4]
    
    @auto_attr
    def hrf_delay(self):
        return self.estimate[5]
    
    @auto_attr
    def rho(self):
        return np.sqrt(self.x**2+self.y**2)
    
    @auto_attr
    def theta(self):
        return np.mod(np.arctan2(self.y,self.x),2*np.pi)
    
    @auto_attr
    def prediction(self):
        return self.model.generate_prediction(self.x, self.y, self.sigma, self.beta, self.baseline, self.hrf_delay)
       
    @auto_attr
    def receptive_field(self):
        return generate_og_receptive_field(self.x, self.y, self.sigma, self.beta, self.baseline,
                                           self.model.stimulus.deg_x,
                                           self.model.stimulus.deg_y) 
    @auto_attr
    def msg(self):
        txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RSQUARED=%.02f   STDERR=%.02f   THETA=%.02f   RHO=%.02d   SIGMA=%.02f   BETA=%.08f   BASELINE=%.03f" 
                %(self.voxel_index[0],
                  self.voxel_index[1],
                  self.voxel_index[2],
                  self.finish-self.start,
                  self.rsquared,
                  self.stderr,
                  self.theta,
                  self.rho,
                  self.sigma,
                  self.beta,
                  self.baseline))
        
        return txt