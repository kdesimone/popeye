#!/usr/bin/python

""" Classes and functions for fitting Gaussian population encoding models """

from __future__ import division, print_function, absolute_import
import time
import gc
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.stats import linregress
from scipy.signal import fftconvolve
import nibabel
import statsmodels.api as sm

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_og_timeseries, generate_og_receptive_field, generate_og_receptive_field, generate_rf_timeseries

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
        
        if polar is True:
            estimates[fit.voxel_index] = (fit.theta,
                                          fit.rho,
                                          fit.sigma,
                                          fit.n,
                                          fit.beta,
                                          fit.hrf_delay,
                                          fit.rsquared,
                                          fit.coefficient,
                                          fit.stderr)
        else:
            estimates[fit.voxel_index] = (fit.x, 
                                          fit.y,
                                          fit.sigma,
                                          fit.n,
                                          fit.beta,
                                          fit.hrf_delay,
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
def compute_model_ts(x, y, sigma, n, beta, hrf_delay,
                     deg_x, deg_y, stim_arr, tr_length):
    
    
    """
    The objective function for GaussianFi class.
    
    Parameters
    ----------
    x : float
        The model estimate along the horizontal dimensions of the display.

    y : float
        The model estimate along the vertical dimensions of the display.

    sigma : float
        The model estimate of the dispersion across the the display.
    
    hrf_delay : float
        The model estimate of the relative delay of the HRF.  The canonical
        HRF is assumed to be 5 s post-stimulus [1]_.
    
    beta : float
        The model estimate of the amplitude of the BOLD signal.
    
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
    
    # generate the response
    rf = generate_og_receptive_field(deg_x, deg_y, x, y, sigma)
    
    # normalize by the integral
    rf /= simps(simps(rf))
    
    # raise it to the n
    rf **= n
    
    # extract the response
    response = generate_rf_timeseries(deg_x, deg_y, stim_arr, rf, x, y, sigma)
    # response = beta*generate_og_timeseries(deg_x, deg_y, stim_arr, x, y, sigma)**n
    
    # create the HRF
    hrf = utils.double_gamma_hrf(hrf_delay, tr_length)
    
    # convolve it with the stimulus
    model = fftconvolve(response, hrf)[0:len(response)]
    
    # scale it by beta
    model *= beta
    
    return model*beta

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
    fit = CompressiveSpatialSummationFit(model,
                                          data,
                                          grids,
                                          bounds,
                                          tr_length,
                                          voxel_index,
                                          auto_fit,
                                          verbose)
    return fit


class CompressiveSpatialSummationModel(PopulationModel):
    
    """
    A CSS population receptive field model class
    
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
        
class CompressiveSpatialSummationFit(PopulationFit):
    
    """
    A CSS population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds, tr_length,
                 voxel_index=(1,2,3), auto_fit=True, verbose=True):
        
        
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
        
        self.grids = grids
        self.bounds = bounds
        self.tr_length = tr_length
        self.voxel_index = voxel_index
        self.auto_fit = auto_fit
        self.verbose = verbose
        
        if self.auto_fit:
            
            tic = time.clock()
            self.ballpark;
            self.estimate;
            self.OLS;
            self.rss;
            toc = time.clock()
            
            msg = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RSQ=%.02f  THETA=%.02f  RHO=%.02d   SIGMA=%.02f   N=%.02f   BETA=%.08f  HRF=%.02f" 
                    %(self.voxel_index[0],
                      self.voxel_index[1],
                      self.voxel_index[2],
                      toc-tic,
                      self.rsquared,
                      self.theta,
                      self.rho,
                      self.sigma,
                      self.n,
                      self.beta,
                      self.hrf_delay))
                          
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
        return utils.gradient_descent_search((self.x0, self.y0, self.s0, self.n0, self.beta0, self.hrf0),
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
    def n0(self):
        return self.ballpark[3]
        
    @auto_attr
    def beta0(self):
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
    def n(self):
        return self.estimate[3]
    
    @auto_attr
    def beta(self):
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
        return compute_model_ts(self.x, self.y, self.sigma, self.n, self.beta, self.hrf_delay,
                                self.model.stimulus.deg_x,
                                self.model.stimulus.deg_y,
                                self.model.stimulus.stim_arr,
                                self.tr_length)
    
    @auto_attr
    def OLS(self):
        return sm.OLS(self.data,self.prediction).fit()
    
    @auto_attr
    def coefficient(self):
        return self.OLS.params[0]
    
    @auto_attr
    def rsquared(self):
        return self.OLS.rsquared
    
    @auto_attr
    def stderr(self):
        return np.sqrt(self.OLS.mse_resid)
    
    @auto_attr
    def rss(self):
        return np.sum((self.data - self.prediction)**2)
    
    @auto_attr
    def receptive_field(self):
        rf = generate_og_receptive_field(self.model.stimulus.deg_x,
                                         self.model.stimulus.deg_y,
                                         self.x, self.y, self.sigma/np.sqrt(self.n), self.beta)
        
        return rf
    
    @auto_attr
    def hemodynamic_response(self):
        return utils.double_gamma_hrf(self.hrf_delay, self.tr_length)
