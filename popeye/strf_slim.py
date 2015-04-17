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
from scipy.integrate import trapz, simps
import nibabel
import statsmodels.api as sm

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries

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
    dims.append(7)
    
    # initialize the statmaps
    estimates = np.zeros(dims)
    
    # extract the prf model estimates from the results queue output
    for fit in output:
        
        if polar is True:
            estimates[fit.voxel_index] = (fit.theta,
                                          fit.spatial_sigma,
                                          fit.temporal_sigma,
                                          fit.weight,
                                          fit.rsquared,
                                          fit.coefficient,
                                          fit.stderr)
        else:
            estimates[fit.voxel_index] = (fit.theta, 
                                          fit.spatial_sigma,
                                          fit.temporal_sigma,
                                          fit.weight,
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

def compute_model_ts(x, y, spatial_sigma, temporal_sigma, weight, beta, baseline,
                     deg_x, deg_y, stim_arr, tr_length, projector_hz):
    
    
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
    
    # create Gaussian
    spatial_rf = generate_og_receptive_field(deg_x, deg_y, x, y, spatial_sigma)
    
    # normalize it by integral
    spatial_rf /= 2 * np.pi * spatial_sigma ** 2
    
    # create mask for speed
    distance = (deg_x - x)**2 + (deg_y - y)**2
    mask = np.zeros_like(distance, dtype='uint8')
    mask[distance < (37*spatial_sigma)**2] = 1
    
    # extract the timeseries
    spatial_response = generate_rf_timeseries(stim_arr, spatial_rf, mask)
    
    # get rid of zeros
    spatial_response[spatial_response==0] = spatial_response[0]
    
    # set the coordinate
    t = np.linspace(0,1,projector_hz)
    
    # set the center of the responses
    center = t[len(t)/2]
    
    # create sustain response for 1 TR
    sustained_1vol = np.exp(-((t-center)**2)/(2*temporal_sigma**2))
    sustained_1vol_norm = sustained_1vol/(temporal_sigma*np.sqrt(2*np.pi))
    
    # create transient response for 1 TR
    transient_1vol = np.append(np.diff(sustained_1vol),0)
    transient_1vol_norm = transient_1vol/(simps(np.abs(transient_1vol),t)/2)
    
    if np.round(transient_1vol_norm[0],3) != 0 or np.round(transient_1vol_norm[0],3) != 0:
        return np.inf
    
    # pad spatial response
    spatial_pad = np.tile(spatial_response,3)
    
    # convolve with transient and sustained RF
    sustained_response = fftconvolve(spatial_pad, sustained_1vol_norm, 'same')[len(spatial_response):-len(spatial_response)]
    transient_response = fftconvolve(spatial_pad, transient_1vol_norm, 'same')[len(spatial_response):-len(spatial_response)]
    
    # take the mean of each TR
    sustained_ts = np.array([np.mean(sustained_response[tp:tp+projector_hz*tr_length]) for tp in np.arange(0,len(spatial_response),projector_hz*tr_length)])
    transient_ts = np.array([np.mean(transient_response[tp:tp+projector_hz*tr_length]) for tp in np.arange(0,len(spatial_response),projector_hz*tr_length)])
    
    # pad them
    sustained_pad = np.tile(sustained_ts,3)
    transient_pad = np.tile(transient_ts,3)
    
    # convolve with hrf
    hrf = utils.double_gamma_hrf(0,tr_length)
    sustained_model = fftconvolve(sustained_pad, hrf, 'same')[len(sustained_ts):-len(sustained_ts)]
    transient_model = fftconvolve(transient_pad, hrf, 'same')[len(sustained_ts):-len(sustained_ts)]
    
    # normalize to a range
    sustained_norm = utils.zscore(sustained_model)
    transient_norm = utils.zscore(transient_model)
    
    # mix it together
    model = sustained_norm * weight + transient_norm * (1-weight)
    
    # normalize it
    model = utils.zscore(model)
    
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
    Ns = args[4]
    tr_length = args[5]
    voxel_index = args[6]
    auto_fit = args[7]
    verbose = args[8]
    
    # fit the data
    fit = SpatioTemporalFit(model,
                            data,
                            grids,
                            bounds,
                            Ns,
                            tr_length,
                            voxel_index,
                            auto_fit,
                            verbose)
    return fit


class SpatioTemporalModel(PopulationModel):
    
    """
    A Gaussian population receptive field model class
    
    """
    
    def __init__(self, stimulus):
        
        """
        A spatiotemporal population receptive field model.
        
        Paramaters
        ----------
        
        stimulus : `VisualStimulus` class object
            A class instantiation of the `VisualStimulus` class
            containing a representation of the visual stimulus.
        
        
        """
        
        PopulationModel.__init__(self, stimulus)
        
class SpatioTemporalFit(PopulationFit):
    
    """
    A spatiotemporal population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds, Ns, tr_length,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
        
        
        """
        
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
                
        """
        
        PopulationFit.__init__(self, model, data, grids, bounds, Ns, 
                               tr_length, voxel_index, auto_fit, verbose)
        
        if self.auto_fit:
            
            self.start = time.clock()
            self.ballpark;
            self.estimate;
            self.OLS;
            self.finish = time.clock()
            
            if self.verbose:
                print(self.msg)
        
    @auto_attr
    def ballpark(self):
        return utils.brute_force_search((self.model.stimulus.deg_x_coarse,
                                         self.model.stimulus.deg_y_coarse,
                                         self.model.stimulus.stim_arr_coarse,
                                         self.tr_length,
                                         self.model.stimulus.projector_hz),
                                        self.grids,
                                        self.bounds,
                                        self.Ns,
                                        self.data,
                                        utils.error_function,
                                        compute_model_ts,
                                        self.very_verbose)

    @auto_attr
    def estimate(self):
        return utils.gradient_descent_search((self.x, self.y, self.spatial_s0, self.temporal_s0, self.weight0),
                                             (self.model.stimulus.deg_x,
                                              self.model.stimulus.deg_y,
                                              self.model.stimulus.stim_arr,
                                              self.tr_length,
                                              self.model.stimulus.projector_hz),
                                             self.bounds,
                                             self.data,
                                             utils.error_function,
                                             compute_model_ts,
                                             self.very_verbose)
    
    @auto_attr
    def x0(self):
        return self.ballpark[0]
    
    @auto_attr
    def y0(self):
        return self.ballpark[1]
        
    @auto_attr
    def spatial_s0(self):
        return self.ballpark[2]
    
    @auto_attr
    def temporal_s0(self):
        return self.ballpark[3]
    
    @auto_attr
    def weight0(self):
        return self.ballpark[4]
    
    @auto_attr
    def x(self):
        return self.estimate[0]
    
    @auto_attr
    def y(self):
        return self.estimate[1]
    
    @auto_attr
    def spatial_sigma(self):
        return self.estimate[2]
    
    @auto_attr
    def temporal_sigma(self):
        return self.estimate[3]
    
    @auto_attr
    def weight(self):
        return self.estimate[4]
        
    @auto_attr
    def spatial_rf(self):
        return generate_og_receptive_field(self.model.stimulus.deg_x, 
                                           self.model.stimulus.deg_y, 
                                           self.x, self.y, self.spatial_sigma)
    
    @auto_attr
    def spatial_rf_norm(self):
        return self.spatial_rf / (2 * np.pi * spatial_sigma ** 2)
    
    @auto_attr
    def theta(self):
        return np.mod(np.arctan2(self.y,self.x),2*np.pi)
    
    @auto_attr
    def rho(self):
        return np.sqrt(self.x**2+self.y**2)
    
    @auto_attr
    def prediction(self):
        return compute_model_ts(self.x, self.y, self.spatial_sigma, self.temporal_sigma, self.weight,
                                self.model.stimulus.deg_x,
                                self.model.stimulus.deg_y,
                                self.model.stimulus.stim_arr,
                                self.tr_length,
                                self.model.stimulus.projector_hz)
    
    def generate_prediction(self, x, y, spatial_sigma, temporal_sigma, weight):
        return compute_model_ts(x, y, spatial_sigma, temporal_sigma, weight,
                                self.model.stimulus.deg_x,
                                self.model.stimulus.deg_y,
                                self.model.stimulus.stim_arr,
                                self.tr_length,
                                self.model.stimulus.projector_hz)
        
        
    
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
    def tr_timescale(self):
        return np.linspace(0,self.tr_length,self.model.stimulus.projector_hz*self.tr_length)
    
    @auto_attr
    def response_center(self):
        return self.tr_timescale[len(self.tr_timescale)/2]
    
    @auto_attr
    def sustained_response(self):
        return np.exp(-((self.tr_timescale-self.response_center)**2)/(2*self.temporal_sigma**2))
    
    @auto_attr
    def sustained_response_norm(self):
        return self.sustained_response/(self.temporal_sigma*np.sqrt(2*np.pi))
    
    @auto_attr
    def transient_response(self):
        return np.append(np.diff(self.sustained_response),0)
    
    @auto_attr
    def transient_response_norm(self):
        return self.transient_response/(simps(np.abs(self.transient_response),self.tr_timescale)/2)
    
    @auto_attr
    def hemodynamic_response(self):
        return utils.double_gamma_hrf(self.hrf_delay, self.tr_length)
    
    @auto_attr
    def msg(self):
        txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RSQ=%.02f  THETA=%.02f   RHO=%.02f   SSIGMA=%.02f   TSIGMA=%.02f   WEIGHT=%.02f"
            %(self.voxel_index[0],
              self.voxel_index[1],
              self.voxel_index[2],
              self.finish-self.start,
              self.rsquared,
              self.theta,
              self.rho,
              self.spatial_sigma,
              self.temporal_sigma,
              self.weight,
              self.beta,
              self.baseline))
        return txt
    
