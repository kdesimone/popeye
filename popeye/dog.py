#!/usr/bin/python

""" Classes and functions for fitting Gaussian population encoding models """

from __future__ import division, print_function, absolute_import
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np
np.set_printoptions(suppress=True)
from scipy.stats import linregress
from scipy.signal import fftconvolve
from scipy.integrate import trapz

import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_dog_timeseries, generate_og_timeseries, generate_og_receptive_field, generate_rf_timeseries

def recast_estimation_results(output, grid_parent):
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
    dims.append(8)
    
    # initialize the statmaps
    polar = np.zeros(dims)
    cartes = np.zeros(dims)
    
    # extract the prf model estimates from the results queue output
    for fit in output:
        
        if fit.__dict__.has_key('fit_stats'):
        
            cartes[fit.voxel_index] = (fit.x, 
                                      fit.y,
                                      fit.sigma_center,
                                      fit.sigma_surround,
                                      fit.beta_center,
                                      fit.beta_surround,
                                      fit.hrf_delay,
                                      fit.fit_stats[2])
                                 
            polar[fit.voxel_index] = (fit.theta,
                                     fit.rho,
                                     fit.sigma_center,
                                     fit.sigma_surround,
                                     fit.beta_center,
                                     fit.beta_surround,
                                     fit.hrf_delay,
                                     fit.fit_stats[2])
                                 
    # get header information from the gridParent and update for the prf volume
    aff = grid_parent.get_affine()
    hdr = grid_parent.get_header()
    hdr.set_data_shape(dims)
    
    # recast as nifti
    nif_polar = nibabel.Nifti1Image(polar,aff,header=hdr)
    nif_polar.set_data_dtype('float32')
   
    nif_cartes = nibabel.Nifti1Image(cartes,aff,header=hdr)
    nif_cartes.set_data_dtype('float32')
    
    return nif_cartes, nif_polar

def compute_model_ts(x, y, sigma, sigma_ratio, volume_ratio, hrf_delay,
                     deg_x, deg_y, stim_arr, tr_length):
    
    
    """
    The objective function for GaussianFi class.
    
    Parameters
    ----------
    x : float
        The model estimate along the horizontal dimensions of the display.
        
    y : float
        The model estimate along the vertical dimensions of the display.
        
    sigma_center : float
        The model estimate of the dispersion of the excitatory center.
    
    sigma_surround : float
        The model estimate of the dispersion of the inhibitory surround.
    
    beta_center : float
        The amplitude of the excitatory center.
    
    beta_surround : float
        The amplitude of the inhibitory surround.
    
    hrf_delay : float
        The model estimate of the relative delay of the HRF.  The canonical
        HRF is assumed to be 5 s post-stimulus [1]_.
    
    tr_length : float
        The length of the repetition time in seconds.
    
    
    Returns
    -------
    
    model : ndarray
    The model prediction time-series.
    
    
    References
    ----------
    
    .. [1] Glover, GH. (1999). Deconvolution of impulse response in 
    event-related BOLD fMRI. NeuroImage 9: 416-429.
    
    """
    
    # extract the response
    rf_center = generate_og_receptive_field(deg_x, deg_y, x, y, sigma)
    rf_surround = generate_og_receptive_field(deg_x, deg_y, x, y, sigma*sigma_ratio) * 1/sigma_ratio**2
    rf = rf_center - np.sqrt(volume_ratio)*rf_surround
    response = generate_rf_timeseries(deg_x, deg_y, stim_arr, rf, x, y, sigma*sigma_ratio)
    
    # generate the hrf
    hrf = utils.double_gamma_hrf(hrf_delay, tr_length)
    
    # convolve it
    model = fftconvolve(response,hrf)[0:len(response)]
    
    return model

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
    tr_length = args[4]
    voxel_index = args[5]
    auto_fit = args[6]
    verbose = args[7]
    
    # fit the data
    fit = DifferenceOfGaussiansFit(model,
                                   data,
                                   grids,
                                   bounds,
                                   tr_length,
                                   voxel_index,
                                   auto_fit,
                                   verbose)
    return fit


class DifferenceOfGaussiansModel(PopulationModel):
    
    """
    A Gaussian population receptive field model class
    
    """
    
    def __init__(self, stimulus):
        
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
        
        PopulationModel.__init__(self, stimulus)
        
class DifferenceOfGaussiansFit(PopulationFit):
    
    """
    A Gaussian population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds, tr_length,
                 voxel_index=(1,2,3), auto_fit=True, verbose=True):
        
        """
        A Gaussian population receptive field model [1]_.
        
        Paramaters
        ----------
        
        og_fit : `GaussianFit` class instance
            A `GaussianFit` object.  This object does not have to be fitted already,
            as the fit will be performed inside the `DifferenceOfGaussiansFit` in any 
            case.  The `GaussianFit` is used as a seed for the `DifferenceOfGaussiansFit`
            search.
        
        bounds : tuple
            A tuple containing the upper and lower bounds for each parameter
            in `parameters`.  If a parameter is not bounded, simply use
            `None`.  For example, `bounds=((0,None),(-10,10),)` would 
            bound the first parameter to be any positive number while the
            second parameter would be bounded between -10 and 10.
            
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
        
        
        PopulationFit.__init__(self, model, data)
        
        self.grids = grids
        self.bounds = bounds
        self.tr_length = tr_length
        self.voxel_index = voxel_index
        self.auto_fit = auto_fit
        self.verbose = verbose
        
        if self.auto_fit:
            
            # run the OG
            tic = time.clock()
            self.ballpark;
            self.estimate;
            self.fit_stats;
            toc = time.clock()
                
            msg = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RVAL=%.02f  THETA=%.02f   RHO=%.02d   SIGMA_1=%.02f   SIGMA_2=%.02f   VOLUME_2=%.02f" 
                    %(self.voxel_index[0],
                      self.voxel_index[1],
                      self.voxel_index[2],
                      toc-tic,
                      self.fit_stats[2],
                      self.theta,
                      self.rho,
                      self.sigma,
                      self.sigma*self.sigma_ratio,
                      self.volume_ratio))
            
            if self.verbose:
                print(msg)
    
    @auto_attr
    def ballpark(self):
        return utils.brute_force_search((self.model.stimulus.deg_x_coarse,
                                         self.model.stimulus.deg_y_coarse,
                                         self.model.stimulus.stim_arr_coarse,
                                         self.tr_length),
                                        self.grids,
                                        self.bounds,
                                        self.data,
                                        utils.error_function,
                                        compute_model_ts)
    
    @auto_attr
    def estimate(self):
        return utils.gradient_descent_search((self.x0, self.y0, self.s0, self.sr0, self.vr0, self.h0),
                                              (self.model.stimulus.deg_x,
                                               self.model.stimulus.deg_y,
                                               self.model.stimulus.stim_arr,
                                               self.tr_length),
                                             self.bounds,
                                             self.data,
                                             utils.error_function,
                                             compute_model_ts)
                                             
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
    def sr0(self):
        return self.ballpark[3]
    
    @auto_attr
    def vr0(self):
        return self.ballpark[4]
                
    @auto_attr
    def h0(self):
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
    def sigma_ratio(self):
        return self.estimate[3]
    
    @auto_attr
    def volume_ratio(self):
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
        return compute_model_ts(self.x, self.y, self.sigma, 
                                self.sigma_ratio, self.volume_ratio, self.hrf_delay,
                                self.model.stimulus.deg_x,
                                self.model.stimulus.deg_y,
                                self.model.stimulus.stim_arr,
                                self.tr_length)
    @auto_attr
    def fit_stats(self):
        return linregress(self.data, self.prediction)
    
    @auto_attr
    def rss(self):
        return np.sum((self.data - self.prediction)**2)
    
    @auto_attr
    def hemodynamic_response(self):
        return utils.double_gamma_hrf(self.hrf_delay, self.tr_length)
    
    @auto_attr
    def receptive_field(self):
            rf_center = generate_og_receptive_field(self.model.stimulus.deg_x, 
                                                    self.model.stimulus.deg_y, 
                                                    self.x, self.y, self.sigma)
                                                    
            rf_surround = generate_og_receptive_field(self.model.stimulus.deg_x, 
                                                      self.model.stimulus.deg_y, 
                                                      self.x, self.y, self.sigma*self.sigma_ratio) * 1.0/self.sigma_ratio**2
            
            rf = rf_center - np.sqrt(self.volume_ratio)*rf_surround
            
            return rf
