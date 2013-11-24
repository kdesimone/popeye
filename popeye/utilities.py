"""This module contains various utility methods that support functionality in other modules.  The
multiprocessing functionality also exists in this module, though that might change with time.

"""

from __future__ import division
import sys, os, time
import ctypes
from multiprocessing import Process, Queue, Array
import numpy as np
import nibabel
from scipy.misc import imresize
from estimation import compute_prf_estimate

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

def resample_stimulus(stimArray,scaleFactor):
    """Resamples the visual stimulus

    The function takes an ndarray `stimArray` and resamples it by the user
    specified `scaleFactor`.  The stimulus array is assumed to be a three
    dimensional ndarray representing the stimulus, in screen pixel coordinates,
    over time.  The first two dimensions of `stimArray` together represent the
    exent of the visual display (pixels) and the last dimensions represents
    time (TRs).

    Parameters
    ----------
    stimArray : ndarray
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.
    scaleFactor : float
        The scale factor by which the stimulus is resampled.  The scale factor
        must be a float, and must be greater than 0.

    Returns
    -------
    resampledStim : ndarray
        An array that is resampled according to the user-specified scale factor.
    """

    dims = np.shape(stimArray)
    resampledStim = np.zeros((dims[0]*scaleFactor,dims[1]*scaleFactor,dims[2]))
    for tp in range(dims[2]):
        resampledStim[:,:,tp] = imresize(stimArray[:,:,tp],scaleFactor)
    
    resampledStim[resampledStim>0] = 1
    
    return resampledStim.astype('short')

def generate_coordinate_matrices(pixelsAcross,pixelsDown,pixelsPerDegree,
                                 scaleFactor):
    """Creates coordinate matrices for representing the visual field in terms
       of degrees of visual angle.

    This function takes the screen dimensions, the pixels per degree, and a
    scaling factor in order to generate a pair of ndarrays representing the
    horizontal and vertical extents of the visual display in degrees of visual
    angle. 

    Parameters
    ----------
    pixelsAcross : int
        The number of pixels along the horizontal extent of the visual display.
    pixelsDown : int
        The number of pixels along the vertical extent of the visual display.
    pixelsPerDegree: float
        The number of pixels that spans 1 degree of visual angle.  This number
        is computed using the display width and the viewing distance.  See the
        config.init_config for details. 
    scaleFactor : float
        The scale factor by which the stimulus is resampled.  The scale factor
        must be a float, and must be greater than 0.

    Returns
    -------
    degX : ndarray
        An array representing the horizontal extent of the visual display in
        terms of degrees of visual angle.
    degY : ndarray
        An array representing the vertical extent of the visual display in
        terms of degrees of visual angle.
    """

    [X,Y] = np.meshgrid(np.arange(pixelsAcross*scaleFactor),
                        np.arange(pixelsDown*scaleFactor))
    degX = (X-np.shape(X)[1]/2)/(pixelsPerDegree*scaleFactor).astype('double')
    degY = (Y-np.shape(Y)[0]/2)/(pixelsPerDegree*scaleFactor).astype('double')

    return degX,degY

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
