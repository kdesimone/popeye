"""This module contains various utility methods that support functionality in
other modules.  The multiprocessing functionality also exists in this module,
though that might change with time.

"""

from __future__ import division
import sys, os, time
import ctypes
from multiprocessing import Process, Queue, Array
import numpy as np
import nibabel
from scipy.misc import imresize

def generate_shared_array(unsharedArray,dataType):
    """Creates synchronized shared arrays from numpy arrays.

    The function takes a numpy array `unsharedArray` and returns a shared
    memory object, `sharedArray_mem`.  The user also specifies the data-type of
    the values in the array with the `dataType` argument.  See
    multiprocessing.Array and ctypes for details on shared memory arrays and
    the data-types.
    
    Parameters
    ----------
    unsharedArray : ndarray
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    dataType : ctypes instance
        The data-type specificed has to be an instance of the ctypes library.
        See ctypes for details.
        
    Returns
    -------
    sharedArray : syncrhonized shared array
        An array that is read/write accessible from multiple processes/threads. 
    """
    
    shared_array_base = Array(dataType,np.prod(np.shape(unsharedArray)))
    sharedArray = np.ctypeslib.as_array(shared_array_base.get_obj())
    sharedArray = np.reshape(sharedArray,np.shape(unsharedArray))
    sharedArray[:] = unsharedArray[:]
    
    return sharedArray
    
def resample_stimulus(stim_arr,scale_factor):
    """Resamples the visual stimulus
    
    The function takes an ndarray `stim_arr` and resamples it by the user
    specified `scale_factor`.  The stimulus array is assumed to be a three
    dimensional ndarray representing the stimulus, in screen pixel coordinates,
    over time.  The first two dimensions of `stim_arr` together represent the
    exent of the visual display (pixels) and the last dimensions represents
    time (TRs).
    
    Parameters
    ----------
    stim_arr : ndarray
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.
    scale_factor : float
        The scale factor by which the stimulus is resampled.  The scale factor
        must be a float, and must be greater than 0.
        
    Returns
    -------
    resampledStim : ndarray
        An array that is resampled according to the user-specified scale factor.
    """
    
    dims = np.shape(stim_arr)
    resampledStim = np.zeros((dims[0]*scale_factor,dims[1]*scale_factor,dims[2]))
    for tp in range(dims[2]):
        resampledStim[:,:,tp] = imresize(stim_arr[:,:,tp],scale_factor)
    
    resampledStim[resampledStim>0] = 1
    
    return resampledStim.astype('short')
    
def generate_coordinate_matrices(pixels_across,pixels_down,ppd,
                                 scale_factor):
    """Creates coordinate matrices for representing the visual field in terms
       of degrees of visual angle.
       
    This function takes the screen dimensions, the pixels per degree, and a
    scaling factor in order to generate a pair of ndarrays representing the
    horizontal and vertical extents of the visual display in degrees of visual
    angle.
    
    Parameters
    ----------
    pixels_across : int
        The number of pixels along the horizontal extent of the visual display.
    pixels_down : int
        The number of pixels along the vertical extent of the visual display.
    ppd: float
        The number of pixels that spans 1 degree of visual angle.  This number
        is computed using the display width and the viewing distance.  See the
        config.init_config for details. 
    scale_factor : float
        The scale factor by which the stimulus is resampled.  The scale factor
        must be a float, and must be greater than 0.
        
    Returns
    -------
    deg_x : ndarray
        An array representing the horizontal extent of the visual display in
        terms of degrees of visual angle.
    deg_y : ndarray
        An array representing the vertical extent of the visual display in
        terms of degrees of visual angle.
    """
    
    [X,Y] = np.meshgrid(np.arange(np.round(pixels_across*scale_factor)),
                        np.arange(np.round(pixels_down*scale_factor)))
                        
                        
    deg_x = (X-np.round(pixels_across*scale_factor)/2)/(ppd*scale_factor)
    deg_y = (Y-np.round(pixels_down*scale_factor)/2)/(ppd*scale_factor)
    
    deg_x = deg_x + 0.5/(ppd*scale_factor)
    deg_y = deg_y + 0.5/(ppd*scale_factor)
        
    return deg_x, deg_y
    
def recast_estimation_results_queue(output,metaData,write=True):
    """
    Recasts the output of the pRF estimation into two nifti_gz volumes.
    
    Takes `output`, a list of multiprocessing.Queue objects containing the
    output of the pRF estimation for each voxel.  The pRF estimates are
    expressed in both polar and Cartesian coordinates.  If the default value
    for the `write` parameter is set to False, then the function returns the
    arrays without writing the nifti files to disk.  Otherwise, if `write` is
    True, then the two nifti files are written to disk.

    Each voxel contains the following metrics: 

        0 x / polar angle
        1 y / eccentricity
        2 sigma
        3 HRF delay
        4 slope of the model-actual fit
        5 standard error of the model-actual fit
        6 correlation of the model-actual fit
        7 two-tailed p-value of the model-actual fit
        
    Parameters
    ----------
    output : list
        A collection of multiprocessing.Queue objects, with one object per
        voxel. 
    metaData : dict
        A dictionary containing meta-data about the analysis being performed.
        For details, see config.py. 

    Returns
    -------
    cartesFileName : string
        The absolute path of the recasted pRF estimation output in Cartesian
        coordinates. 
    polarFileName : string
        The absolute path of the recasted pRF estimation output in polar
        coordinates.  

    """
    # load the gridParent
    gridParent = nibabel.load(metaData['maskPath'])
    dims = list(gridParent.get_shape())
    dims.append(8)
    pRF_polar = np.zeros(dims)
    pRF_cartes = np.zeros(dims)

    # extract the pRF model estimates from the results queue output
    for job in output:
        for voxel in job:
            xi,yi,zi = voxel[0:3]
            x,y,s,d = voxel[3:7]
            stats = voxel[7]
            slope,intercept,rval,pval,stderr = stats[:]
            pRF_cartes[xi,yi,zi,:] = x,y,s,d,slope,stderr,rval,pval
            pRF_polar[xi,yi,zi,:] = (np.mod(np.arctan2(x,y),2*np.pi),
                                np.sqrt(x**2+y**2),s,d,slope,stderr,rval,pval)
    
    # get header information from the gridParent and update for the pRF volume
    aff = gridParent.get_affine()
    hdr = gridParent.get_header()
    hdr.set_data_shape(dims)
    voxelDims = list(hdr.get_zooms())
    voxelDims[-1] = 8
    hdr.set_zooms(voxelDims)

    # write the files
    now = time.strftime('%Y%m%d_%H%M%S')
    nif_polar = nibabel.Nifti1Image(pRF_polar,aff,header=hdr)
    nif_polar.set_data_dtype('float32')
    polarFileName = '%s/%s_polar_%s.nii.gz' %(metaData['outputPath'],
                                              metaData['baseFileName'],now)
    nif_cartes = nibabel.Nifti1Image(pRF_cartes,aff,header=hdr)
    nif_cartes.set_data_dtype('float32')
    cartesFileName = '%s/%s_cartes_%s.nii.gz' %(metaData['outputPath'],
                                                metaData['baseFileName'],now)
    
    if write:
        nibabel.save(nif_polar,polarFileName)
        nibabel.save(nif_cartes,cartesFileName)
        return polarFileName,cartesFileName
    else:
        return pRF_cartes,pRF_polar

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
        5 SSE between pRF and nRF gaussian
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
        The absolute path of the recasted pRF estimation output in Cartesian
        coordinates. 
    polarFileName : string
        The absolute path of the recasted pRF estimation output in polar
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
            x, y = funcData['pRF_cartes'][xi,yi,zi,0:2]
            phi, rho = funcData['pRF_polar'][xi,yi,zi,0:2]
            d = funcData['pRF_cartes'][xi,yi,zi,3]
            rval = funcData['pRF_cartes'][xi,yi,zi,6]
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
    Uses the multiprocessing toolbox to parallize the voxel-wise pRF estimation
    across a user-specified number of cpus.

    Each voxel is treated as the atomic unit.  Collections of voxels are sent
    to each of the user-specified allocated cpus for pRF estimation.
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
        A 4D numpy array containing the functional data to be used for the pRF
        estimation. For details, see config.py 

    metaData : dict
        A dictionary containing meta-data about the analysis being performed.
        For details, see config.py.

    Returns
    -------
    output : list
        A list of multiprocessing.Queue objects that house the pRF estimates
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
    increases in the pRF size, we randomize the voxel order so that a particular core doesn't get
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

