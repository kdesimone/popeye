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

        if fit.__dict__.has_key('fit_stats_'):

            cartes[fit.voxel_index] = (fit.x_,
                                      fit.y_,
                                      fit.sigma_,
                                      fit.hrf_delay_,
                                      fit.beta_,
                                      fit.rss_,
                                      fit.fit_stats_[2])

            polar[fit.voxel_index] = (np.mod(np.arctan2(fit.x_,fit.y_),2*np.pi),
                                     np.sqrt(fit.x_**2+fit.y_**2),
                                     fit.sigma_,
                                     fit.hrf_delay_,
                                     fit.beta_,
                                     fit.rss_,
                                     fit.fit_stats_[2])

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


def compute_model_ts(x, y, sigma, beta, hrf_delay, deg_x, deg_y, stim_arr,
                     tr_length):
    """The objective function for GaussianFit class.

    Parameters
    ----------
    x : float
        The model estimate along the horizontal dimensions of the display.

    y : float
        The model estimate along the vertical dimensions of the display.

    sigma : float
        The model estimate of the dispersion across the the display.

    beta : float
        The model estimate of the amplitude of the BOLD signal.

    hrf_delay : float
        The model estimate of the relative delay of the HRF. The canonical
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

    # generate a prediction
    stim = generate_og_timeseries(deg_x, deg_y, stim_arr, x, y, sigma)

    # scale it
    stim *= beta

    # create the HRF
    hrf = utils.double_gamma_hrf(hrf_delay, tr_length)

    # convolve HRF with the stimulus
    model = fftconvolve(stim, hrf)[0:len(stim)]

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


class GaussianModel(PopulationModel):
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


class GaussianFit(PopulationFit):
    """Gaussian population receptive field fit class"""

    def __init__(self, model, data, search_bounds, fit_bounds, tr_length,
                 voxel_index=(1,2,3), auto_fit=True, uncorrected_rval=0.0,
                 verbose=True):
        """Gaussian population receptive field (pRF) model

        The Gaussian pRF model was introduced in [1]_.

        Paramaters
        ----------
        model : `GaussianModel` class instance containing the representation
            of the visual stimulus.

        data : ndarray
            An array containing the measured BOLD signal.

        search_bounds : tuple
            A tuple indicating the search space for the brute-force grid-search.
            The tuple contains pairs of upper and lower bounds for exploring a
            given dimension. For example `fit_bounds=((-10,10),(0,5),)` will
            search the first dimension from -10 to 10 and the second from 0 to
            5. These values cannot be None.

            For more information, see `scipy.optimize.brute`.

        fit_bounds : tuple
            A tuple containing the upper and lower bounds for each parameter
            in `parameters`.  If a parameter is not bounded, simply use `None`.
            For example, `fit_bounds=((0,None),(-10,10),)` would bound the
            first parameter to be any positive number while the second
            parameter would be bounded between -10 and 10.

        tr_length : float
            The length of the repetition time in seconds.

        voxel_index : tuple
            A tuple containing the index of the voxel being modeled. The
            fitting procedure does not require a voxel index, but collating the
            results across many voxels will does require voxel indices. With
            voxel indices, the brain volume can be reconstructed using the
            newly computed model estimates.

        auto-fit : bool, default True
            A flag for automatically running the fitting procedures once the
            `GaussianFit` object is instantiated.

        uncorrected_rval : float, default 0.0
            The second stage of the model estimation is only performed when the
            model estimates of the first stage fitting procedure correlate
            higher than ``uncorrected_rval`` with the measures BOLD signal.

        verbose : bool, default True
            A flag for printing some summary information about the model
            estiamte after the fitting procedures have completed.

        References
        ----------
        .. [1] Dumoulin SO, Wandell BA. (2008) Population receptive field
               estimates in human visual cortex. NeuroImage 39:647-660
        """

        PopulationFit.__init__(self, model, data)

        self.search_bounds = search_bounds
        self.fit_bounds = fit_bounds
        self.tr_length = tr_length
        self.voxel_index = voxel_index
        self.auto_fit = auto_fit
        self.verbose = verbose
        self.uncorrected_rval = uncorrected_rval

        if self.auto_fit:
            tic = time.clock()
            self.ballpark_estimate;

            if self.fit_stats0_[2] > self.uncorrected_rval:
                self.estimates_;
                self.fit_stats_;
                self.rss_;
                toc = time.clock()

                msg = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   RVAL=%.02f"
                        %(self.voxel_index[0],
                          self.voxel_index[1],
                          self.voxel_index[2],
                          toc-tic,
                          self.fit_stats_[2]))

            else:
                toc = time.clock()

                msg = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   COARSE_RVAL=%.02f"
                        %(self.voxel_index[0],
                          self.voxel_index[1],
                          self.voxel_index[2],
                          toc-tic,
                          self.fit_stats0_[2]))

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
    def estimates_(self):
        return utils.gradient_descent_search((self.x0_, self.y0_, self.sigma0_,
                                              self.beta0_, self.hrf_delay0_),
                                             (self.model.stimulus.deg_x,
                                              self.model.stimulus.deg_y,
                                              self.model.stimulus.stim_arr,
                                              self.tr_length),
                                              self.fit_bounds,
                                              self.data,
                                              utils.error_function,
                                              compute_model_ts)


    @auto_attr
    def x0_(self):
        return self.ballpark_estimate[0]

    @auto_attr
    def y0_(self):
        return self.ballpark_estimate[1]

    @auto_attr
    def sigma0_(self):
        return self.ballpark_estimate[2]

    @auto_attr
    def beta0_(self):
        return self.ballpark_estimate[3]

    @auto_attr
    def hrf_delay0_(self):
        return self.ballpark_estimate[4]

    @auto_attr
    def x_(self):
        return self.estimates_[0]

    @auto_attr
    def y_(self):
        return self.estimates_[1]

    @auto_attr
    def sigma_(self):
        return self.estimates_[2]

    @auto_attr
    def beta_(self):
        return self.estimates_[3]

    @auto_attr
    def hrf_delay_(self):
        return self.estimates_[4]

    @auto_attr
    def prediction0(self):
        """predicted hemodynamic response based on coarse model estimates"""
        return compute_model_ts(self.x0_, self.y0_, self.sigma0_, self.beta0_,
                                self.hrf_delay0_,
                                self.model.stimulus.deg_x_coarse,
                                self.model.stimulus.deg_y_coarse,
                                self.model.stimulus.stim_arr_coarse,
                                self.tr_length)

    @auto_attr
    def prediction(self):
        """predicted hemodynamic response based on model estimates"""
        return compute_model_ts(self.x_, self.y_, self.sigma_, self.beta_,
                                self.hrf_delay_,
                                self.model.stimulus.deg_x,
                                self.model.stimulus.deg_y,
                                self.model.stimulus.stim_arr,
                                self.tr_length)

    @auto_attr
    def fit_stats0_(self):
        """statistical fit based on coarse model estimates"""
        return linregress(self.data, self.prediction0)

    @auto_attr
    def fit_stats_(self):
        """statistical fit based on model estimates"""
        return linregress(self.data, self.prediction)

    @auto_attr
    def rss_(self):
        return np.sum((self.data - self.prediction)**2)

    @auto_attr
    def receptive_field(self):
        rf = generate_og_receptive_field(self.model.stimulus.deg_x,
                                         self.model.stimulus.deg_y,
                                         self.x_, self.y_, self.sigma_)

        rf *= self.beta_

        return rf

    @auto_attr
    def hemodynamic_response(self):
        return utils.double_gamma_hrf(self.hrf_delay_, self.tr_length)
