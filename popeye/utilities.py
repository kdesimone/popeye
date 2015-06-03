"""This module contains various utility methods that support functionality in
other modules.  The multiprocessing functionality also exists in this module,
though that might change with time.

"""

from __future__ import division
import sys, os, time
from multiprocessing import Array
from itertools import repeat

import numpy as np
import nibabel
from scipy.misc import imresize
from scipy.special import gamma
from scipy.optimize import brute, fmin_powell, fmin
from scipy.integrate import romb, trapz
import sharedmem
from statsmodels import api as sm

def recast_estimation_results(output, grid_parent):
    
    # load the gridParent
    dims = list(grid_parent.shape)
    dims = dims[0:3]
    dims.append(len(output[0].estimate)+3)
    
    # initialize the statmaps
    estimates = np.zeros(dims)
    
    # extract the prf model estimates from the results queue output
    for fit in output:
        
        # gather the estimate + stats
        voxel_dat = list(fit.estimate)
        voxel_dat.append(fit.rsquared)
        voxel_dat.append(fit.coefficient)
        voxel_dat.append(fit.stderr)
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

def make_nifti(data, grid_parent):
    
    # get header information from the gridParent and update for the prf volume
    aff = grid_parent.get_affine()
    hdr = grid_parent.get_header()
    
    # recast as nifti
    nifti = nibabel.Nifti1Image(data,aff,header=hdr)
    
    return nifti


def generate_shared_array(unshared_arr,dtype):
    
    """
    Creates synchronized shared arrays from numpy arrays.
    
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
def normalize(array, imin=-1, imax=1):
    
    """
    A short-hand function for normalizing an array to a desired range.

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
    
    
    dmin = array.min()
    dmax = array.max()
    array -= dmin
    array *= imax - imin
    array /= dmax - dmin
    array += imin
    return array
    

# generic gradient descent
def gradient_descent_search(parameters, bounds, data,
                            error_function, objective_function, verbose):
                            
    """
    A generic gradient-descent error minimization function.
    
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
    
    .. [1] Fletcher, R., Powell, M.J.D. (1963) A rapidly convergent descent 
    method for minimization. Compututation Journal 6: 163-168.
    
    """
    
    estimate, err,  _, _, _, warnflag =\
        fmin_powell(error_function, 
                    parameters,
                    args=(bounds, data, objective_function, verbose),
                    full_output=True,
                    disp=False)
    
    return estimate

def brute_force_search(grids, bounds, Ns, data,
                       error_function, objective_function, verbose):
                       
    """
    A generic brute-force grid-search error minimization function.
    
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
                       
    estimate, err,  _, _ =\
        brute(error_function,
              args=(bounds, data, objective_function, verbose),
              ranges=grids,
              Ns=Ns,
              finish=None,
              full_output=True,
              disp=False)
              
    return estimate

# generic error function
def error_function(parameters, bounds, data, objective_function, verbose):
    
    """
    A generic error function with bounding.
    

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
    error = np.sum((data-prediction)**2)
    
    # print for debugging
    if verbose:
        print(parameters, error)
    
    return error

def double_gamma_hrf(delay, tr_length, frames_per_tr=1.0, integrator=trapz):
    
    """
    The double-gamma hemodynamic reponse function (HRF) used to convolve with
    the stimulus time-series.
    
    The user specifies only the delay of the peak and under-shoot.  
    The delay shifts the peak and under-shoot by a variable number of 
    seconds. The other parameters are hard-coded. The HRF delay is 
    modeled for each voxel independently. The double-gamme HRF and 
    hard-coded values are based on previous work [1]_.
    
    
    Parameters
    ----------
    delay : float
        The delay of the HRF peak and under-shoot.
        
    tr_length : float
        The length of the repetition time in seconds.
    
    integrator : callable
        The integration function for normalizing the units of the HRF 
        so that the area under the curve is the same for differently
        delayed HRFs.  Set integrator to None to turn off normalization.
        
    Returns
    -------
    hrf : ndarray
        The hemodynamic response function to convolve with the stimulus
        time-series.
        
        
    Reference
    ----------
    Glover, G.H. (1999) Deconvolution of impulse response in event-related BOLD.
    fMRI. NeuroImage 9: 416-429.
    
    """
    
    # add delay to the peak and undershoot params (alpha 1 and 2)
    # add delay to the peak and undershoot params (alpha 1 and 2)
    alpha_1 = 5.0/tr_length+delay/tr_length
    beta_1 = 1.0
    c = 0.1
    alpha_2 = 15.0/tr_length+delay/tr_length
    beta_2 = 1.0
    
    t = np.arange(0,33/tr_length,tr_length/frames_per_tr)
    scale = 1
    hrf = scale*( ( ( t ** (alpha_1) * beta_1 ** alpha_1 *
                      np.exp( -beta_1 * t )) /gamma( alpha_1 )) - c *
                  ( ( t ** (alpha_2 ) * beta_2 ** alpha_2 * np.exp( -beta_2 * t ))/gamma( alpha_2 ) ) )
    
    if integrator:
        hrf /= integrator(hrf)
    
    return hrf

def multiprocessor(targetMethod,stimData,funcData,metaData):
    """
    Uses the multiprocessing toolbox to parallize the voxel-wise prf estimation
    across a user-specified number of cpus.

    Each voxel is treated as the atomic unit.  Collections of voxels are sent
    to each of the user-specified allocated cpus for prf estimation.
    Currently, the strategy is to use multiprocessing.Process to start each
    batch of voxels on a given cpu.  The results of each are written to a
    multiprocessing.Queue object, collected into a list of results, and
    returned to the user. 


    Parameters
    ----------
    stimData : dict
        A dictionary containing the stimulus array and other stimulus-related
        data.  For details, see config.py 

    funcData : ndarray
        A 4D numpy array containing the functional data to be used for the prf
        estimation. For details, see config.py 

    metaData : dict
        A dictionary containing meta-data about the analysis being performed.
        For details, see config.py.

    Returns
    -------
    output : list
        A list of multiprocessing.Queue objects that house the prf estimates
        for all the voxels analyzed as specified in `metaData`.  The `output`
        will be a list whose length is equal to the number of cpus specified in
        `metaData`.
    """

    # figure out how many voxels are in the mask & the number of jobs we have
    # allocated 
    [xi,yi,zi] = metaData['voxels']
    cpus = metaData['cpus']

    # Set up the voxel lists for each job
    voxelLists = []
    cutOffs = [int(np.floor(i)) for i in np.linspace(0,len(xi),cpus+1)]
    for i in range(len(cutOffs)-1):
        l = range(cutOffs[i],cutOffs[i+1])
        voxelLists.append(l)

    # initialize Queues for managing the outputs of the jobs
    results_q = Queue()

    # start the jobs
    procs = []
    for j in range(cpus):
        voxels = [xi[voxelLists[j]],yi[voxelLists[j]],zi[voxelLists[j]]]
        metaData['core_voxels'] = voxels
        p = Process(target=targetMethod,args=(stimData,funcData,metaData,
                                              results_q))
        procs.append(p)
        p.start()

    # gather the outputs from the queue
    output = []
    for i in range(len(procs)):
        output.append(results_q.get())

    # close the jobs
    for p in procs:
        p.join()

    return output


def percent_change(ts, ax=-1):
    """Returns the % signal change of each point of the times series
    along a given axis of the array time_series

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
    """Returns the z-score of each point of the time series
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

def randomize_voxels(voxels):
    """Returns a set of 3D coordinates that are randomized.
    
    Since the brain is highly spatially correlated and because computational time increases with
    increases in the prf size, we randomize the voxel order so that a particular core doesn't get
    stuck with a disproportionately high number of voxels whose sigma values are large.
    
    Parameters
    ----------
    voxels : ndarray
        an array of voxel coordinates
    
    
    Returns
    _______
    randomized_voxels : ndarray
        the shuffled voxel coordinates 
    """
    
    
    xi,yi,zi = voxels[:]
    randVec = np.random.rand(len(xi))
    randInd = np.argsort(randVec)
    randomized_voxels = tuple((xi[randInd],yi[randInd],zi[randInd]))
    
    return randomized_voxels

    
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
    Fit = args[0]
    model = args[1]
    data = args[2]
    grids = args[3]
    bounds = args[4]
    Ns = args[5]
    voxel_index = args[6]
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


def multiprocess_bundle(Fit, model, data, grids, bounds, Ns, indices, auto_fit, verbose):
    
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
    
    return dat

def gaussian_2D(X, Y, x0, y0, sigma_x, sigma_y, degrees, amplitude=1):
    
    theta = degrees*np.pi/180
        
    a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    
    Z = amplitude*np.exp( - (a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    
    return Z


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
    Ns = args[5]
    voxel_index = args[6]
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

          