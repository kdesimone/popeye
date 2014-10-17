#!/usr/bin/python

""" Classes and functions for fitting Gaussian population encoding models """

from __future__ import division, print_function, absolute_import
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.stats import linregress
from scipy.signal import fftconvolve
import nibabel

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_og_timeseries, generate_og_receptive_field


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
    dims.append(7)
    
    # initialize the statmaps
    polar = np.zeros(dims)
    cartes = np.zeros(dims)
    
    # extract the prf model estimates from the results queue output
    for fit in output:
        
        if fit.__dict__.has_key('rss'):
        
            cartes[fit.voxel_index] = (fit.x, 
                                      fit.y,
                                      fit.sigma,
                                      fit.hrf_delay,
                                      fit.beta,
                                      fit.rss,
                                      fit.fit_stats[2])
                                 
            polar[fit.voxel_index] = (np.mod(np.arctan2(fit.x,fit.y),2*np.pi),
                                     np.sqrt(fit.x**2+fit.y**2),
                                     fit.sigma,
                                     fit.hrf_delay,
                                     fit.beta,
                                     fit.rss,
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

def compute_model_ts(x, y, sigma_center, sigma_surround, 
                     beta_center, beta_surround, hrf_delay,
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
    
    # limiting cases for a center-surround receptive field
    if sigma_center > sigma_surround:
        return np.inf
    if beta_center < beta_surround:
        return np.inf
    
    # time-series for the center and surround
    stim_center = generate_og_timeseries(deg_x, deg_y, stim_arr, x, y, sigma_center, 1)
    stim_surround = generate_og_timeseries(deg_x, deg_y, stim_arr, x, y, sigma_surround, 1)

    # # combine to create the DoG time-series
    # stim_dog = beta_center * stim_center + beta_surround * stim_surround
    
    # # get the stim time-series for center and surround
    # stim_center, stim_surround = generate_dog_timeseries(deg_x, deg_y, stim_arr, x, y,
    #                                                      sigma_center, sigma_surround,
    #                                                      beta_center, beta_surround)
    
    # scale them
    scaled_center = beta_center * stim_center
    scaled_surround = -beta_surround * stim_surround
    
    # combine them
    stim_dog = scaled_center + scaled_surround
    
    # generate the hrf
    hrf = utils.double_gamma_hrf(hrf_delay, tr_length)
    
    # convolve it with the stimulus timeseries
    model = fftconvolve(stim_dog, hrf)[0:len(stim_dog)]
    
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
    search_bounds = args[2]
    fit_bounds = args[3]
    tr_length = args[4]
    voxel_index = args[5]
    auto_fit = args[6]
    verbose = args[7]
    
    
    # fit the data
    fit = GaussianFit(model,
                      data,
                      search_bounds,
                      fit_bounds,
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
    
    def __init__(self, model, data, og_fit, search_bounds, fit_bounds, tr_length,
                 voxel_index=(1,2,3), auto_fit=True, verbose=True, uncorrected_rval=0):
        
        
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
        
        tr_length : float
            The length of the repetition time in seconds.
        
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
        
        PopulationFit.__init__(self, model, data)
        
        self.og_fit = og_fit
        self.search_bounds = search_bounds
        self.fit_bounds = fit_bounds
        self.tr_length = tr_length
        self.voxel_index = voxel_index
        self.auto_fit = auto_fit
        self.verbose = verbose
        self.uncorrected_rval = uncorrected_rval
        
        if self.auto_fit:
            
            tic = time.clock()
            self.estimate;
            self.fit_stats;
            self.rss;
            self.receptive_field;
            toc = time.clock()
            
            msg = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RVAL=%.02f" 
                    %(self.voxel_index[0],
                      self.voxel_index[1],
                      self.voxel_index[2],
                      toc-tic,
                      self.fit_stats[2]))
            
            if self.verbose:
                print(msg)
                
    @auto_attr
    def ballpark_estimate(self):
        return utils.brute_force_search((self.model.stimulus.deg_x_coarse,
                                         self.model.stimulus.deg_y_coarse,
                                         self.model.stimulus.stim_arr_coarse,
                                         self.tr_length),
                                        self.search_bounds,
                                        self.fit_bounds,
                                        self.data,
                                        utils.error_function,
                                        compute_model_ts)
                                        
    @auto_attr
    def estimate(self):
        return utils.gradient_descent_search((self.x0, self.y0, self.sigma_center0, self.sigma_surround0,
                                              self.beta_center0, self.beta_surround0, self.hrf0),
                                             (self.model.stimulus.deg_x,
                                              self.model.stimulus.deg_y,
                                              self.model.stimulus.stim_arr,
                                              self.tr_length),
                                             self.fit_bounds,
                                             self.data,
                                             utils.error_function,
                                             compute_model_ts)
                                             
    @auto_attr
    def x0(self):
        return self.og_fit.estimate[0]
        
    @auto_attr
    def y0(self):
        return self.og_fit.estimate[1]
    
    @auto_attr
    def sigma_center0(self):
        return self.og_fit.estimate[2]
    
    @auto_attr
    def sigma_surround0(self):
        return self.og_fit.estimate[2]
    
    @auto_attr
    def beta_center0(self):
        return self.og_fit.estimate[3]
    
    @auto_attr
    def beta_surround0(self):
        return self.og_fit.estimate[3]
    
    @auto_attr
    def hrf0(self):
        return self.og_fit.estimate[4]
        
    @auto_attr
    def x(self):
        return self.estimate[0]
        
    @auto_attr
    def y(self):
        return self.estimate[1]
        
    @auto_attr
    def sigma_center(self):
        return self.estimate[2]
        
    @auto_attr
    def sigma_surround(self):
        return self.estimate[3]
        
    @auto_attr
    def beta_center(self):
        return self.estimate[4]
    
    @auto_attr
    def beta_surround(self):
        return self.estimate[5]
            
    @auto_attr
    def hrf_delay(self):
        return self.estimate[6]
        
    @auto_attr
    def coarse_prediction(self):
        return compute_model_ts(self.x0, self.y0, self.sigma_center0, self.sigma_surround0,
                                self.beta_center0, self.beta_surround0, self.hrf0, 
                                self.model.stimulus.deg_x_coarse,
                                self.model.stimulus.deg_y_coarse,
                                self.model.stimulus.stim_arr_coarse,
                                self.tr_length)
    
    @auto_attr
    def prediction(self):
        return compute_model_ts(self.x, self.y, self.sigma_center, self.sigma_surround,
                                self.beta_center, self.beta_surround, self.hrf_delay,
                                self.model.stimulus.deg_x,
                                self.model.stimulus.deg_y,
                                self.model.stimulus.stim_arr,
                                self.tr_length)
    
    @auto_attr
    def coarse_fit_stats(self):
        return linregress(self.data, self.coarse_prediction)
    
    @auto_attr
    def fit_stats(self):
        return linregress(self.data, self.prediction)
    
    @auto_attr
    def rss(self):
        return np.sum((self.data - self.prediction)**2)
    
    @auto_attr
    def receptive_field(self):
        rf_center = generate_og_receptive_field(self.model.stimulus.deg_x,
                                                      self.model.stimulus.deg_y,
                                                      self.x, self.y, self.sigma_center, 1)
        
        rf_surround = generate_og_receptive_field(self.model.stimulus.deg_x,
                                                      self.model.stimulus.deg_y,
                                                      self.x, self.y, self.sigma_surround, 1)
        
        return rf_center*self.beta_center - rf_surround*self.beta_surround
        
    @auto_attr
    def hemodynamic_response(self):
        return utils.double_gamma_hrf(self.hrf_delay, self.tr_length)
