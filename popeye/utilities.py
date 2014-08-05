"""This module contains various utility methods that support functionality in
other modules.  The multiprocessing functionality also exists in this module,
though that might change with time.

"""

from __future__ import division
import sys, os, time

import numpy as np
import nibabel
from scipy.misc import imresize
from scipy.special import gamma
from scipy.optimize import brute, fmin_powell
from scipy.integrate import romb, trapz

# normalize to a specific range
def normalize(array, imin=-1, imax=1):
    dmin = array.min()
    dmax = array.max()
    array -= dmin
    array *= imax - imin
    array /= dmax - dmin
    array += imin
    return array
    

# generic gradient descent
def gradient_descent_search(parameters, args, fit_bounds, response,
                            error_function, objective_function):
    
    estimate, err,  _, _, _, warnflag =\
        fmin_powell(error_function, parameters,
                    args=(args, fit_bounds, response, objective_function),
                    full_output=True,
                    disp=False)
    
    return estimate

def brute_force_search(args, search_bounds, fit_bounds, response,
                       error_function, objective_function):
                       
    estimate, err,  _, _ =\
        brute(error_function,
              args=(args, fit_bounds, response, objective_function),
              ranges=search_bounds,
              Ns=4,
              finish=None,
              full_output=True,
              disp=False)
              
    return estimate

# generic error function
def error_function(params, args, bounds, response, func, debug=True):
    
    # check ifparameters are inside bounds
    for p, b in zip(params,bounds):
        
        # if not return an inf
        if b[0] and b[0] > p:
            return np.inf
        if b[1] and b[1] < p:
            return np.inf
    
    # merge the parameters and arguments
    ensemble = []
    ensemble.extend(params)
    ensemble.extend(args)
    
    # compute the RSS
    error = np.sum((response-func(*ensemble))**2)
    
    # print for debugging
    if debug:
        print(params, error)
    
    return error

def double_gamma_hrf(delay, tr_length, frames_per_tr=1.0, integrator=trapz):
    """
    The double-gamma hemodynamic reponse function (HRF) used to convolve with
    the stimulus time-series.
    
    The user specifies only the delay of the peak and under-shoot The delay
    shifts the peak and under-shoot by a variable number of seconds.  The other
    parameters are hard-coded.  The HRF delay is modeled for each voxel
    independently.  The double-gamme HRF andhard-coded values are based on
    previous work (Glover, 1999).
    
    
    Parameters
    ----------
    delay : float
        The delay of the HRF peak and under-shoot.
    tr_length : float
        The length of the repetition time in seconds.
    frames_per_tr : int
        The number number of stimulus frames that are used during a single functional volume.
        
        
    Returns
    -------
    hrf : ndarray
        The hemodynamic response function to convolve with the stimulus
        time-series.
        
        
    Reference
    ----------
    Glover, G.H. (1999) Deconvolution of impulse response in event-related BOLD.
    fMRI. NeuroImage 9: 416 429.
    
    """
    
    # add delay to the peak and undershoot params (alpha 1 and 2)
    alpha_1 = 6.0/tr_length+delay/tr_length
    beta_1 = 1.0
    c = 0.2
    alpha_2 = 16.0/tr_length+delay/tr_length
    beta_2 = 1.0
    
    t = np.arange(0,33/tr_length,tr_length/frames_per_tr)
    scale = 1
    hrf = scale*( ( ( t ** (alpha_1) * beta_1 ** alpha_1 *
                      np.exp( -beta_1 * t )) /gamma( alpha_1 )) - c *
                  ( ( t ** (alpha_2 ) * beta_2 ** alpha_2 * np.exp( -beta_2 * t ))
                      /gamma( alpha_2 ) ) )
                      
    hrf /= integrator(hrf)
    
    return hrf


def recast_estimation_results(output, grid_parent, write=True):
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

def recast_simulation_results_queue(output,funcData,metaData,write=True):
    """
    Recasts the output of the neural RF (nRF) simulation into two nifti_gz
    volumes. 
    
    Takes `output`, a list of multiprocessing.Queue objects containing the
    output of the nRF simulation for each voxel.  The nRF estimates are
    expressed in both polar and Cartesian coordinates.  If the default value
    for the `write` parameter is set to False, then the function returns the
    arrays without writing the nifti files to disk.  Otherwise, if `write` is
    True, then the two nifti files are written to disk. 

    Each voxel contains the following metrics:
    
        0 x / polar angle
        1 y / eccentricity
        2 neural RF estimate
        3 HRF delay
        4 visuotopic scatter
        5 SSE between prf and nRF gaussian
        6 correlation of the model-actual fit
        7 percent change from old to new sigma
        
    Parameters
    ----------
    output : list
        A collection of multiprocessing.Queue objects, with one object per voxel.
    metaData : dict
        A dictionary containing meta-data about the analysis being performed.
        For details, see config.py. 
        
        
    Returns
    -------
    cartesFileName : string
        The absolute path of the recasted prf estimation output in Cartesian
        coordinates. 
    polarFileName : string
        The absolute path of the recasted prf estimation output in polar
        coordinates. 
        
    """
    
    # load the gridParent
    gridParent = nibabel.load(metaData['maskPath'])
    dims = list(gridParent.get_shape())
    dims.append(8)
    nRF_polar = np.zeros(dims)
    nRF_cartes = np.zeros(dims)
    
    
    # extract the nRF model estimates from the results queue output
    for job in output:
        for voxel in job:
            xi,yi,zi = voxel[0:3]
            x, y = funcData['prf_cartes'][xi,yi,zi,0:2]
            phi, rho = funcData['prf_polar'][xi,yi,zi,0:2]
            d = funcData['prf_cartes'][xi,yi,zi,3]
            rval = funcData['prf_cartes'][xi,yi,zi,6]
            sigma = voxel[3]
            SSE = voxel[4]
            meanScatter = voxel[5]
            percentChange = voxel[6]
            nRF_cartes[xi,yi,zi,:] = (x, y, sigma, d, meanScatter, SSE, rval,
                                      percentChange)
            nRF_polar[xi,yi,zi,:] = (phi, rho, sigma, d, meanScatter, SSE, rval,
                                     percentChange)
            
    # get header information from the gridParent and update for the nRF volume
    aff = gridParent.get_affine()
    hdr = gridParent.get_header()
    hdr.set_data_shape(dims)
    voxelDims = list(hdr.get_zooms())
    voxelDims[-1] = 8
    hdr.set_zooms(voxelDims)
        
    # write the files
    now = time.strftime('%Y%m%d_%H%M%S')
    nif_polar = nibabel.Nifti1Image(nRF_polar,aff,header=hdr)
    nif_polar.set_data_dtype('float32')
    polarFileName = '%s/%s_polar_ts%d.nii.gz'%(metaData['outputPath'],
                                               metaData['baseFileName'],
                                               metaData['temporal_smooth'])

    nif_cartes = nibabel.Nifti1Image(nRF_cartes,aff,header=hdr)
    nif_cartes.set_data_dtype('float32')
    cartesFileName = '%s/%s_cartes_ts%d.nii.gz' %(metaData['outputPath'],
                                                  metaData['baseFileName'],
                                                  metaData['temporal_smooth'])
    
    if write:
        nibabel.save(nif_polar,polarFileName)
        nibabel.save(nif_cartes,cartesFileName)
        return polarFileName,cartesFileName
    else:
        return nRF_cartes,nRF_polar

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

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y


