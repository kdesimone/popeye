from __future__ import division
import time
import ctypes
import shutil
from multiprocessing import Process, Queue, Array

import numpy as np
from scipy.interpolate import interp1d

import nibabel

def reconstruct_stimulus(metaData,stimData,funcData,verbose=True):
    
    # grab voxel indices
    xi,yi,zi = metaData['core_voxels']
    
    # set up the time vector for interpolation of the time-series based on the
    # HRF tau estimate 
    runLength = np.shape(stimData['stimRecon'])[-1]
    usTime = np.array([np.round(item,1) for item in np.r_[0:runLength-1:0.1]])
    
    # printing niceties
    numVoxels = len(xi)
    voxelCount = 1
    printLength = len(xi)/10
    
    # printing niceties
    numVoxels = len(xi)
    voxelCount = 1
    printLength = len(xi)/10
    
    for xvoxel,yvoxel,zvoxel in zip(xi,yi,zi):
        
        # Grab timestamp
        toc = time.clock()
        
        # grab the pRF estimate for this voxel
        pRFx,pRFy,pRFs,pRFd = funcData['pRF_polar'][xvoxel,yvoxel,zvoxel,0:4]
        pRFcov = funcData['pRF_polar'][xvoxel,yvoxel,zvoxel,6]
        
        # create the receptive field from the metaData['pRF_polar'] estimate
        rf = MakeFastRF(stimData['degXFine'],stimData['degYFine'],pRFx,pRFy,pRFs)

        # grab the voxel's time-series
        tsActual = funcData['bold'][xvoxel,yvoxel,zvoxel,:]
        
        # create the upsampled time-series
        f = interp1d(np.arange(runLength), tsActual, kind='cubic')
        usActual = f(usTime)

        for tr in range(runLength):
            usTimePoint = np.nonzero(usTime == np.round(tr+pRFd+5,1))[0]
            if usTimePoint < runLength*10:
                intensity = usActual[usTimePoint][0]
                stimData['stimRecon'][:,:,tr] += intensity*rf
        
        # Grab a timestamp
        tic = time.clock()
        
        if verbose:
            percentDone = (voxelCount/numVoxels)*100
            print("%.02d%%  VOXEL=(%.03d,%.03d,%.03d)  TIME=%.03E" 
                  %(percentDone,
                    xvoxel,
                    yvoxel,
                    zvoxel,
                    tic-toc))
                    
    return None
    
def reconstruct_stimulus_realtime(voxels,stimData,funcData,pRFs,verbose=True):
    
    # grab voxel indices
    xi,yi,zi = voxels[:]
    
    # printing niceties
    numVoxels = len(xi)
    voxelCount = 1
    
    # initialize an empty frame to store the reconstruction
    emptyFrame = np.zeros_like(stimData['stimArrayFine'][:,:,0])
    
    for voxel in range(numVoxels):
        
        # grab voxel coordinate
        xvoxel,yvoxel,zvoxel = voxels[0][voxel],voxels[1][voxel],voxels[2][voxel]
        
        # grab the RF from the pRFs array
        rf = pRFs[:,:,voxel]
        
        # smooth the time-series
        intensity = funcData[xvoxel,yvoxel,zvoxel]
        
        # create the receptive field from the pRF estimatetimate
        emptyFrame += rf*intensity
        
        if verbose:
            print("VOXEL=(%.03d,%.03d,%.03d)" %(xvoxel,yvoxel,zvoxel))
            
    return emptyFrame
    
def reconstruct_stimulus_realtime_smoothing(voxels,stimData,funcData,pRFs,verbose=True):

    # grab voxel indices
    xi,yi,zi = voxels[:]

    # printing niceties
    numVoxels = len(xi)
    voxelCount = 1

    # initialize an empty frame to store the reconstruction
    emptyFrame = np.zeros_like(stimData['stimArrayFine'][:,:,0])

    for voxel in range(numVoxels):

        # grab voxel coordinate
        xvoxel,yvoxel,zvoxel = voxels[0][voxel],voxels[1][voxel],voxels[2][voxel]

        # grab the RF from the pRFs array
        rf = pRFs[:,:,voxel]

        # smooth the time-series
        ts = funcData[xvoxel,yvoxel,zvoxel,:]
        intensity = wiener(ts,5)[-1]

        # create the receptive field from the pRF estimate
        emptyFrame += rf*intensity

        if verbose:
            print("VOXEL=(%.03d,%.03d,%.03d,%.03f,%.03f,%.03f)" %(xvoxel,yvoxel,zvoxel))

    return emptyFrame

def multiprocess_stimulus_reconstruction(stimData,funcData,metaData):

    # figure out how many voxels are in the mask & the number of jobs we have allocated
    [xi,yi,zi] = metaData['voxels']
    cpus = metaData['cpus']
    
    # Set up the voxel lists for each job
    voxelLists = []
    cutOffs = [int(np.floor(i)) for i in np.linspace(0,len(xi),cpus+1)]
    for i in range(len(cutOffs)-1):
        l = range(cutOffs[i],cutOffs[i+1])
        voxelLists.append(l)
        
    # start the jobs
    procs = []
    for j in range(cpus):
        voxels = [xi[voxelLists[j]],yi[voxelLists[j]],zi[voxelLists[j]]]
        metaData['core_voxels'] = voxels
        p = Process(target=reconstruct_stimulus,args=(metaData,stimData,funcData))
        procs.append(p)
        p.start()
        
    # close the jobs
    for p in procs:
        p.join()
        
    return None
    

def multiprocess_stimulus_reconstruction_realtime(stimData,funcData,metaData,
                                                  pRF,tr):
    
    # figure out how many voxels are in the mask & the number of jobs we have
    # allocated s
    [xi,yi,zi] = metaData['voxels']
    cpus = metaData['cpus']
    
    # Set up the voxel lists for each job
    voxelLists = []
    cutOffs = [int(np.floor(i)) for i in np.linspace(0,len(xi),cpus+1)]
    for i in range(len(cutOffs)-1):
        l = range(cutOffs[i],cutOffs[i+1])
        voxelLists.append(l)

    # start the jobs
    procs = []
    for j in range(cpus):
        voxels = [xi[voxelLists[j]],yi[voxelLists[j]],zi[voxelLists[j]]]
        metaData['core_voxels'] = voxels
        p = Process(target=reconstruct_stimulus_realtime,args=(metaData,stimData,funcData,pRF,tr))
        procs.append(p)
        p.start()

    # close the jobs
    for p in procs:
        p.join()

    return None
