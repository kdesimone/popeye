"""This module contains various utility methods that support functionality in
other modules.  The multiprocessing functionality also exists in this module,
though that might change with time.

"""

from __future__ import division
import sys, os, time, fnmatch, copy
from multiprocessing import Array
from itertools import repeat
from random import shuffle
import datetime

import numpy as np
import nibabel
from scipy.special import gamma
from scipy.optimize import brute, fmin_powell, fmin
from scipy.integrate import romb, trapz
from scipy import c_, ones, dot, stats, diff
from scipy.linalg import inv, solve, det
from numpy import log, pi, sqrt, square, diagonal
from numpy.random import randn, seed
import sharedmem

# Python 3 compatibility below:
try:  # pragma: no cover
    import cPickle
except ImportError:  # pragma: no cover
    import _pickle as cPickle

try:  # pragma: no cover
    from types import SliceType
except ImportError:  # pragma: no cover
    SliceType = slice

try: # pragma: no cover
    xrange
except NameError:  # pragma: no cover
    xrange = range


def regularizing_error_function(parameter, bundle, p_bounds, thr=0.10):
    
    # if out of bounds
    if parameter < p_bounds[0] or parameter > p_bounds[1]:
            return np.inf
    
    # execute
    output = regularizing_objective_function(parameter, bundle)
    
    # what?
    error = np.mean([o.rss for o in output if not np.isnan(o.rss) and o.rsquared > thr])
    
    # when?
    now = str(datetime.datetime.now()).replace(" ","_").replace(".","_").replace(":","_").replace('-','_')
    
    # verbose!
    print('parameter=%.05f    error=%.05f    now=%s' %(o.model.parameter, error, now))
    
    return error

def regularizing_objective_function(parameter, bundle):
    
        # attach the guess for tau to each of the voxels in the bundle
        for voxel in bundle:
            model = voxel[1]
            model.parameter = parameter
            
        # fit each of the voxels
        num_cpus = sharedmem.cpu_count()-1
        with sharedmem.Pool(np=num_cpus) as pool:
            output = pool.map(parallel_fit, bundle)
        
        return output

def regularizer(bundle, p_grid, p_bounds, Ns=None):
    
    p0 = brute(regularizing_error_function, (p_grid,), args=(bundle, p_bounds), Ns=Ns, finish=None, full_output=True)
    phat = fmin_powell(regularizing_error_function, p0[0], args=(bundle, p_bounds), xtol=1e-4, ftol=1e-4, full_output=True, retall=True)
    
    return p0,phat

def spm_hrf(delay, TR):
    """ An implementation of spm_hrf.m from the SPM distribution

Arguments:

Required:
TR: repetition time at which to generate the HRF (in seconds)

Optional:
p: list with parameters of the two gamma functions:
                                                     defaults
                                                    (seconds)
   p[0] - delay of response (relative to onset)         6
   p[1] - delay of undershoot (relative to onset)      16
   p[2] - dispersion of response                        1
   p[3] - dispersion of undershoot                      1
   p[4] - ratio of response to undershoot               6
   p[5] - onset (seconds)                               0
   p[6] - length of kernel (seconds)                   32

"""
    # default settings
    p=[5,15,1,1,6,0,32]
    p=[float(x) for x in p]

    # delay variation
    p[0] += delay
    p[1] += delay

    fMRI_T = 16.0

    TR=float(TR)
    dt  = TR/fMRI_T
    u   = np.arange(p[6]/dt + 1) - p[5]/dt
    hrf=stats.gamma.pdf(u,p[0]/p[2],scale=1.0/(dt/p[2])) - stats.gamma.pdf(u,p[1]/p[3],scale=1.0/(dt/p[3]))/p[4]
    good_pts=np.array(range(np.int(p[6]/TR)))*fMRI_T
    hrf=hrf[list(good_pts)]
    # hrf = hrf([0:(p(7)/RT)]*fMRI_T + 1);
    hrf /= trapz(hrf)
    return hrf

def grid_slice(start, stop, Ns, dryrun=False):

    # special case
    if Ns == 2:
        step = stop-start
    # all others
    else:
        step = (stop-start) / (Ns-1)

    # if true, this return the ndarray rather than slice object.
    if dryrun: # pragma: no cover
        return arange(start, stop+step, step) # pragma: no cover
    else:
        return slice(start, stop+step, step)

def distance_mask(x, y, sigma, deg_x, deg_y, amplitude=1):

    distance = (deg_x - x)**2 + (deg_y - y)**2
    mask = np.zeros_like(distance, dtype='uint8')
    mask[distance < sigma**2] = 1
    mask *= amplitude

    return mask

def recast_estimation_results(output, grid_parent, overloaded=False):

    # load the gridParent
    dims = list(grid_parent.shape)
    dims = dims[0:3]

    if overloaded == True and output[0].overloaded_estimate is not None:
        dims.append(len(output[0].overloaded_estimate)+1)
    else:
        dims.append(len(output[0].estimate)+1)

    # initialize the statmaps
    estimates = np.zeros(dims)

    # extract the prf model estimates from the results queue output
    for fit in output:

        if not np.isnan(fit.rsquared): # pragma: no cover

            # gather the estimate + stats
            if overloaded == True and fit.overloaded_estimate is not None:
                voxel_dat = list(fit.overloaded_estimate)
            else:
                voxel_dat = list(fit.estimate)

            voxel_dat.append(fit.rsquared)
            voxel_dat = np.array(voxel_dat)

            # assign to
            estimates[fit.voxel_index] = voxel_dat

    # get header information from the gridParent and update for the prf volume
    aff = grid_parent.get_affine()
    hdr = grid_parent.get_header()
    hdr.set_data_shape(dims)

    # recast as nifti
    nifti_estimates = nibabel.Nifti1Image(estimates,aff,header=hdr)

    return nifti_estimates

def make_nifti(data, grid_parent=None):

    if grid_parent:

        # get header information from the gridParent and update for the prf volume
        aff = grid_parent.get_affine()
        hdr = grid_parent.get_header()

        # recast as nifti
        nifti = nibabel.Nifti1Image(data,aff,header=hdr)

    else:
        aff = np.eye(4,4)
        nifti = nibabel.Nifti1Image(data,aff)

    return nifti

def generate_shared_array(unshared_arr,dtype):

    r"""Creates synchronized shared arrays from numpy arrays.

    The function takes a numpy array `unshared_arr` and returns a shared
    memory object, `shared_arr`.  The user also specifies the data-type of
    the values in the array with the `dataType` argument.  See
    multiprocessing.Array and ctypes for details on shared memory arrays and
    the data-types.

    Parameters
    ----------
    unshared_arr : ndarray
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    dtype : ctypes instance
        The data-type specificed has to be an instance of the ctypes library.
        See ctypes for details.

    Returns
    -------
    shared_arr : synchronized shared array
        An array that is read accessible from multiple processes/threads.
    """

    shared_arr = sharedmem.empty(unshared_arr.shape, dtype=dtype)
    shared_arr[:] = unshared_arr[:]
    return shared_arr

# normalize to a specific range
def normalize(array, imin=-1, imax=1, axis=-1):

    r"""A short-hand function for normalizing an array to a desired range.

    Parameters
    ----------
    array : ndarray
        An array to be normalized.

    imin : float
        The desired minimum value in the output array.  Default: -1

    imax : float
        The desired maximum value in the output array.  Default: 1


    Returns
    -------
    array : ndarray
        The normalized array with imin and imax as the minimum and
        maximum values.
    """

    new_arr = array.copy()

    if np.ndim(imin) == 0:
        dmin = new_arr.min()
        dmax = new_arr.max()
        new_arr = new_arr-dmin
        new_arr = new_arr*(imax - imin)
        new_arr = new_arr/(dmax - dmin)
        new_arr = new_arr+imin
    else:
        dmin = new_arr.min(axis=axis)
        dmax = new_arr.max(axis=axis)
        new_arr -= dmin[:,np.newaxis]
        new_arr *= imax[:,np.newaxis] - imin[:,np.newaxis]
        new_arr /= dmax[:,np.newaxis] - dmin[:,np.newaxis]
        new_arr += imin[:,np.newaxis]

    return new_arr


# generic gradient descent
def gradient_descent_search(data, error_function, objective_function, parameters, bounds, verbose):

    r"""A generic gradient-descent error minimization function.

    The values inside `parameters` are used as a seed-point for a
    gradient-descent error minimization procedure [1]_.  The user must
    also supply an `objective_function` for producing a model prediction
    and an `error_function` for relating that prediction to the actual,
    measured `data`.

    In addition, the user may also supply  `fit_bounds`, containing pairs
    of upper and lower bounds for each of the values in parameters.
    If `fit_bounds` is specified, the error minimization procedure in
    `error_function` will return an Inf whether the parameters exceed the
    minimum or maxmimum values specified in `fit_bounds`.


    Parameters
    ----------
    parameters : tuple
        A tuple of values representing a model setting.

    args : tuple
        Extra arguments to `objective_function` beyond those in `parameters`.

    fit_bounds : tuple
        A tuple containing the upper and lower bounds for each parameter
        in `parameters`.  If a parameter is not bounded, simply use
        `None`.  For example, `fit_bounds=((0,None),(-10,10),)` would
        bound the first parameter to be any positive number while the
        second parameter would be bounded between -10 and 10.

    data : ndarray
        The actual, measured time-series against which the model is fit.

    error_function : callable
        The error function that relates the model prediction to the
        measure data.  The error function returns a float that represents
        the residual sum of squared errors between the prediction and the
        data.

    objective_function : callable
        The objective function that takes `parameters` and `args` and
        proceduces a model time-series.

    Returns
    -------
    estimate : tuple
        The model solution given `parameters` and `objective_function`.


    References
    ----------

    .. [1] Fletcher, R, Powell, MJD (1963) A rapidly convergent descent
    method for minimization, Compututation Journal 6, 163-168.

    """

    output = fmin_powell(error_function, parameters,
                         args=(bounds, data, objective_function, verbose),
                         full_output=True, disp=False, retall=True)

    return output

def brute_force_search(data, error_function, objective_function, grids, bounds, Ns=None, verbose=False):

    r"""A generic brute-force grid-search error minimization function.

    The user specifies an `objective_function` and the corresponding
    `args` for generating a model prediction.  The `brute_force_search`
    uses `search_bounds` to dictate the bounds for evenly sample the
    parameter space.

    In addition, the user must also supply  `fit_bounds`, containing pairs
    of upper and lower bounds for each of the values in parameters.
    If `fit_bounds` is specified, the error minimization procedure in
    `error_function` will return an Inf whether the parameters exceed the
    minimum or maxmimum values specified in `fit_bounds`.

    The output of `brute_force_search` can be used as a seed-point for
    the fine-tuned solutions from `gradient_descent_search`.


    Parameters
    ----------
    args : tuple
        Arguments to `objective_function` that yield a model prediction.

    grids : tuple
        A tuple indicating the search space for the brute-force grid-search.
        The tuple contains pairs of upper and lower bounds for exploring a
        given dimension.  For example `grids=((-10,10),(0,5),)` will
        search the first dimension from -10 to 10 and the second from 0 to 5.
        These values cannot be `None`.

        For more information, see `scipy.optimize.brute`.

    bounds : tuple
        A tuple containing the upper and lower bounds for each parameter
        in `parameters`.  If a parameter is not bounded, simply use
        `None`.  For example, `fit_bounds=((0,None),(-10,10),)` would
        bound the first parameter to be any positive number while the
        second parameter would be bounded between -10 and 10.

    Ns : int
        Number of samples per stimulus dimension to sample during the ballpark search.

        For more information, see `scipy.optimize.brute`.

    data : ndarray
       The actual, measured time-series against which the model is fit.

    error_function : callable
       The error function that relates the model prediction to the
       measure data.  The error function returns a float that represents
       the residual sum of squared errors between the prediction and the
       data.

    objective_function : callable
      The objective function that takes `parameters` and `args` and
      proceduces a model time-series.

    Returns
    -------
    estimate : tuple
       The model solution given `parameters` and `objective_function`.

    """

    # if user provides their own grids
    if isinstance(grids[0], SliceType):
        output = brute(error_function,
                       args=(bounds, data, objective_function, verbose),
                       ranges=grids,
                       finish=None,
                       full_output=True,
                       disp=False)

    # otherwise specify (min,max) and Ns for each dimension
    else:
        output = brute(error_function,
               args=(bounds, data, objective_function, verbose),
               ranges=grids,
               Ns=Ns,
               finish=None,
               full_output=True,
               disp=False)


    return output

# generic error function
def error_function(parameters, bounds, data, objective_function, verbose):

    r"""A generic error function with bounding.

    Parameters
    ----------
    parameters : tuple
        A tuple of values representing a model setting.

    args : tuple
        Extra arguments to `objective_function` beyond those in `parameters`.

    data : ndarray
       The actual, measured time-series against which the model is fit.

    objective_function : callable
        The objective function that takes `parameters` and `args` and
        proceduces a model time-series.

    debug : bool
        Useful for debugging a model, will print the parameters and error.

    Returns
    -------
    error : float
        The residual sum of squared errors between the prediction and data.
    """

    ############
    #   NOTE   #
    ############
    # as of now, this will not work if your model has 1 parameter.
    # i think it is because scipy.optimize.brute returns
    # a scalar when num params is 1, and a tuple/list
    # when num params is > 1. have to look into this further

    # check if parameters are inside bounds
    for p, b in zip(parameters,bounds):
        # if not return an inf
        if b[0] and p < b[0]:
            return np.inf
        if b[1] and b[1] < p:
            return np.inf

    # merge the parameters and arguments
    ensemble = []
    ensemble.extend(parameters)

    # compute the RSS
    prediction = objective_function(*ensemble)

    # if nan, return inf
    if np.any(np.isnan(prediction)):
        return np.inf # pragma: no cover

    # else, return RSS
    error = np.sum((data-prediction)**2)

    # print for debugging
    if verbose:
        print(parameters, error)

    return error

def double_gamma_hrf(delay, tr, fptr=1.0, integrator=trapz):

    r"""The double gamma hemodynamic reponse function (HRF).
    The user specifies only the delay of the peak and undershoot.
    The delay shifts the peak and undershoot by a variable number of
    seconds. The other parameters are hardcoded. The HRF delay is
    modeled for each voxel independently. The form of the HRF and the
    hardcoded values are based on previous work [1]_.

    Parameters
    ----------
    delay : float
        The delay of the HRF peak and undershoot.

    tr : float
        The length of the repetition time in seconds.

    fptr : float
        The number of stimulus frames per reptition time.  For a
        60 Hz projector and with a 1 s repetition time, the fptr
        would be equal to 60.  It is possible that you will bin all
        the frames in a single TR, in which case fptr equals 1.

    integrator : callable
        The integration function for normalizing the units of the HRF
        so that the area under the curve is the same for differently
        delayed HRFs.  Set integrator to None to turn off normalization.

    Returns
    -------
    hrf : ndarray
        The hemodynamic response function to convolve with the stimulus
        timeseries.

    Reference
    ----------
    .. [1] Glover, GH (1999) Deconvolution of impulse response in event related
    BOLD fMRI. NeuroImage 9, 416-429.

    """

    # add delay to the peak and undershoot params (alpha 1 and 2)
    alpha_1 = 5.0/tr+delay/tr
    beta_1 = 1.0
    c = 0.1
    alpha_2 = 15.0/tr+delay/tr
    beta_2 = 1.0

    t = np.arange(0,32,tr)

    hrf = ( ( ( t ** (alpha_1) * beta_1 ** alpha_1 * np.exp( -beta_1 * t )) /gamma( alpha_1 )) - c *
            ( ( t ** (alpha_2) * beta_2 ** alpha_2 * np.exp( -beta_2 * t )) /gamma( alpha_2 )) )

    if integrator: # pragma: no cover
        hrf /= integrator(hrf)

    return hrf

def percent_change(ts, ax=-1):

    r"""Returns the % signal change of each point of the times series
    along a given axis of the array timeseries

    Parameters
    ----------
    ts : ndarray
        an array of time series

    ax : int, optional (default to -1)
        the axis of time_series along which to compute means and stdevs

    Returns
    -------
    ndarray
        the renormalized time series array (in units of %)

    Examples
    --------
    >>> np.set_printoptions(precision=4)  # for doctesting
    >>> ts = np.arange(4*5).reshape(4,5)
    >>> ax = 0
    >>> percent_change(ts,ax)
    array([[-100.    ,  -88.2353,  -78.9474,  -71.4286,  -65.2174],
           [ -33.3333,  -29.4118,  -26.3158,  -23.8095,  -21.7391],
           [  33.3333,   29.4118,   26.3158,   23.8095,   21.7391],
           [ 100.    ,   88.2353,   78.9474,   71.4286,   65.2174]])
    >>> ax = 1
    >>> percent_change(ts,ax)
    array([[-100.    ,  -50.    ,    0.    ,   50.    ,  100.    ],
           [ -28.5714,  -14.2857,    0.    ,   14.2857,   28.5714],
           [ -16.6667,   -8.3333,    0.    ,    8.3333,   16.6667],
           [ -11.7647,   -5.8824,    0.    ,    5.8824,   11.7647]])
    """
    ts = np.asarray(ts)

    return (ts / np.expand_dims(np.mean(ts, ax), ax) - 1) * 100


def zscore(time_series, axis=-1):

    r"""Returns the z-score of each point of the time series
    along a given axis of the array time_series.

    Parameters
    ----------
    time_series : ndarray
        an array of time series
    axis : int, optional
        the axis of time_series along which to compute means and stdevs

    Returns
    _______
    zt : ndarray
        the renormalized time series array
    """

    time_series = np.asarray(time_series)
    et = time_series.mean(axis=axis)
    st = time_series.std(axis=axis)
    sl = [slice(None)] * len(time_series.shape)
    sl[axis] = np.newaxis
    if sl == [None]:
        zt = (time_series - et)/st
    else:
        zt = time_series - et[sl]
        zt /= st[sl]

    return zt

def multiprocess_bundle(Fit, model, data, grids, bounds, indices, auto_fit=True, verbose=1, Ns=None):

    # num voxels
    num_voxels = np.shape(data)[0]

    # expand out grids and bounds
    grids = [grids,]*num_voxels
    bounds = [bounds,]*num_voxels

    # package the data structure
    dat = zip(repeat(Fit,num_voxels),
              repeat(model,num_voxels),
              data,
              grids,
              bounds,
              repeat(Ns,num_voxels),
              indices,
              repeat(auto_fit,num_voxels),
              repeat(verbose,num_voxels))

    # return randomized order
    np.random.seed(12345)
    # Consume the zip upfront into a list:
    dat = list(dat)
    idx = np.argsort(np.random.rand(len(dat)))
    dat = [dat[i] for i in idx]

    return dat

def gaussian_2D(X, Y, x0, y0, sigma_x, sigma_y, degrees, amplitude=1):

    theta = degrees*np.pi/180

    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2

    Z = amplitude*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))

    return Z


def parallel_fit(args):

    r"""
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

    fit : `Fit` class object
        A fit object that contains all the inputs and outputs of the
        pRF model estimation for a single voxel.

    """


    # unpackage the arguments
    Fit = args[0]
    model = args[1]
    data = args[2]
    grids = args[3]
    bounds = args[4]
    voxel_index = args[5]
    Ns = args[6]
    auto_fit = args[7]
    verbose = args[8]

    # fit the data
    fit = Fit(model,
              data,
              grids,
              bounds,
              Ns,
              voxel_index,
              auto_fit,
              verbose)
    return fit

def cartes_to_polar(cartes):

    """
    Assumes that the 0th and 1st parameters are cartesian `x` and `y`
    """
    polar = cartes.copy()
    polar[...,0] = np.mod(np.arctan2(cartes[...,1], cartes[...,0]),2*np.pi)
    polar[...,1] = np.sqrt(cartes[...,0]**2 + cartes[...,1]**2)
    return polar


def find_files(directory, pattern):
    names = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                names.append(filename)
    return names

def binner(signal, times, bins):
    binned_response = np.zeros(len(bins)-2)
    bin_width = bins[1] - bins[0]
    for t in xrange(1,len(bins)):
        the_bin = bins[t]
        binned_signal = signal[(times >= the_bin-bin_width) & (times <= the_bin)]
        binned_response[t-2] = np.sum(binned_signal)
    return binned_response

import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    #PEAKDET Detect peaks in a vector
    #        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    #        maxima and minima ("peaks") in the vector V.
    #        MAXTAB and MINTAB consists of two columns. Column 1
    #        contains indices in V, and column 2 the found values.
    #
    #        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    #        in MAXTAB and MINTAB are replaced with the corresponding
    #        X-values.
    #
    #        A point is considered a maximum peak if it has the maximal
    #        value, and was preceded (to the left) by a value lower by
    #        DELTA.
    #
    # Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    # This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None: # pragma: no cover
        x = arange(len(v))

    v = asarray(v)

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

# class ols:
#     """
#     Author: Vincent Nijs (+ ?)
#     Email: v-nijs at kellogg.northwestern.edu
#     Last Modified: Mon Jan 15 17:56:17 CST 2007
#     """
#
#     def __init__(self,y,x,y_varnm = 'y',x_varnm = ''):
#
#         self.y = y
#         self.x = c_[ones(x.shape[0]),x]
#         self.y_varnm = y_varnm
#         if not isinstance(x_varnm,list):
#             self.x_varnm = ['const'] + list(x_varnm)
#         else:
#             self.x_varnm = ['const'] + x_varnm
#
#         # Estimate model using OLS
#         self.estimate()
#
#     def estimate(self):
#
#         # estimating coefficients, and basic stats
#         self.inv_xx = inv(dot(self.x.T,self.x))
#         xy = dot(self.x.T,self.y)
#         self.b = dot(self.inv_xx,xy)                    # estimate coefficients
#
#         self.nobs = self.y.shape[0]                     # number of observations
#         self.ncoef = self.x.shape[1]                    # number of coef.
#         self.df_e = self.nobs - self.ncoef              # degrees of freedom, error
#         self.df_r = self.ncoef - 1                      # degrees of freedom, regression
#
#         self.e = self.y - dot(self.x,self.b)            # residuals
#         self.sse = dot(self.e,self.e)/self.df_e         # SSE
#         self.se = sqrt(diagonal(self.sse*self.inv_xx))  # coef. standard errors
#         self.t = self.b / self.se                       # coef. t-statistics
#         self.p = (1-stats.t.cdf(abs(self.t), self.df_e)) * 2    # coef. p-values
#
#         self.R2 = 1 - self.e.var()/self.y.var()         # model R-squared
#         self.R2adj = 1-(1-self.R2)*((self.nobs-1)/(self.nobs-self.ncoef))   # adjusted R-square
#
#         self.F = (self.R2/self.df_r) / ((1-self.R2)/self.df_e)  # model F-statistic
#         self.Fpv = 1-stats.f.cdf(self.F, self.df_r, self.df_e)  # F-statistic p-value
#
#     def dw(self):
#         """
#         Calculates the Durbin-Waston statistic
#         """
#         de = diff(self.e,1)
#         dw = dot(de,de) / dot(self.e,self.e);
#
#         return dw
#
#     def omni(self):
#         """
#         Omnibus test for normality
#         """
#         return stats.normaltest(self.e)
#
#     def JB(self):
#         """
#         Calculate residual skewness, kurtosis, and do the JB test for normality
#         """
#
#         # Calculate residual skewness and kurtosis
#         skew = stats.skew(self.e)
#         kurtosis = 3 + stats.kurtosis(self.e)
#
#         # Calculate the Jarque-Bera test for normality
#         JB = (self.nobs/6) * (square(skew) + (1/4)*square(kurtosis-3))
#         JBpv = 1-stats.chi2.cdf(JB,2);
#
#         return JB, JBpv, skew, kurtosis
#
#     def ll(self):
#         """
#         Calculate model log-likelihood and two information criteria
#         """
#
#         # Model log-likelihood, AIC, and BIC criterion values
#         ll = -(self.nobs*1/2)*(1+log(2*pi)) - (self.nobs/2)*log(dot(self.e,self.e)/self.nobs)
#         aic = -2*ll/self.nobs + (2*self.ncoef/self.nobs)
#         bic = -2*ll/self.nobs + (self.ncoef*log(self.nobs))/self.nobs
#
#         return ll, aic, bic
#
#     def summary(self):
#         """
#         Printing model output to screen
#         """
#
#         # local time & date
#         t = time.localtime()
#
#         # extra stats
#         ll, aic, bic = self.ll()
#         JB, JBpv, skew, kurtosis = self.JB()
#         omni, omnipv = self.omni()
#
#         # printing output to screen
#         print '\n=============================================================================='
#         print "Dependent Variable: " + self.y_varnm
#         print "Method: Least Squares"
#         print "Date: ", time.strftime("%a, %d %b %Y",t)
#         print "Time: ", time.strftime("%H:%M:%S",t)
#         print '# obs:               %5.0f' % self.nobs
#         print '# variables:     %5.0f' % self.ncoef
#         print '=============================================================================='
#         print 'variable     coefficient     std. Error      t-statistic     prob.'
#         print '=============================================================================='
#         for i in range(len(self.x_varnm)):
#             print '''% -5s          % -5.6f     % -5.6f     % -5.6f     % -5.6f''' % tuple([self.x_varnm[i],self.b[i],self.se[i],self.t[i],self.p[i]])
#         print '=============================================================================='
#         print 'Models stats                         Residual stats'
#         print '=============================================================================='
#         print 'R-squared            % -5.6f         Durbin-Watson stat  % -5.6f' % tuple([self.R2, self.dw()])
#         print 'Adjusted R-squared   % -5.6f         Omnibus stat        % -5.6f' % tuple([self.R2adj, omni])
#         print 'F-statistic          % -5.6f         Prob(Omnibus stat)  % -5.6f' % tuple([self.F, omnipv])
#         print 'Prob (F-statistic)   % -5.6f           JB stat             % -5.6f' % tuple([self.Fpv, JB])
#         print 'Log likelihood       % -5.6f           Prob(JB)            % -5.6f' % tuple([ll, JBpv])
#         print 'AIC criterion        % -5.6f         Skew                % -5.6f' % tuple([aic, skew])
#         print 'BIC criterion        % -5.6f         Kurtosis            % -5.6f' % tuple([bic, kurtosis])
#         print '=============================================================================='
