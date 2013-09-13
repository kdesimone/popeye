from __future__ import division
import ctypes
import shutil
from multiprocessing import Process, Queue, Array
import numpy as np
from pylab import find
from scipy.interpolate import interp1d
import nibabel
from MakeFastRF import MakeFastRF
from utilities import generate_shared_array

def reconstruct_stimulus(voxels,stimData,funcData,pRF,verbose=True):
    
    # grab voxel indices
    xi,yi,zi = voxels[:]
    
    # set up the time vector for interpolation of the time-series based on the HRF tau estimate
    runLength = np.shape(stimData['stimRecon'])[-1]
    usTime = np.array([np.round(item,1) for item in np.r_[0:runLength-1:0.1]])
    
    # printing niceties
    numVoxels = len(xi)
    voxelCount = 1
    printLength = len(xi)/10
    
    for xvoxel,yvoxel,zvoxel in zip(xi,yi,zi):
        
        # grab the pRF estimate for this voxel
        pRFx,pRFy,pRFs,pRFd = pRF[xvoxel,yvoxel,zvoxel,0:4]
        pRFcov = pRF[xvoxel,yvoxel,zvoxel,-1]
        
        # create the receptive field from the pRF estimate
        rf = MakeFastRF(stimData['degXFine'],stimData['degYFine'],pRFx,pRFy,pRFs)
        
        # grab the voxel's time-series
        tsActual = funcData[xvoxel,yvoxel,zvoxel,:]
        
        # create the upsampled time-series
        f = interp1d(arange(runLength), tsActual, kind='cubic')
        usActual = f(usTime)
        
        for tr in range(runLength):
            usTimePoint = np.nonzero(usTime == np.round(tr+pRFd+5,1))[0]
            if usTimePoint < runLength*10:
                intensity = usActual[usTimePoint][0]
                stimData['stimRecon'][:,:,tr] += intensity*rf*pRFcov
        
        if verbose:
            print("VOXEL=(%.03d,%.03d,%.03d)" %(xvoxel,yvoxel,zvoxel))
    
    
    return None
    
def multiprocess_stimulus_reconstruction(stimData,funcData,metaData):
    
    # figure out how many voxels are in the mask & the number of jobs we have allocated
    [xi,yi,zi] = metaData['voxels']
    cpus = metaData['cpus']
    
    # Grab the pRF estimation results and create a shared array from it
    if not shutil.os.path.exists(metaData['pRF_path']):
        sys.exit('The pRF dataset %s cannot be found!' %(metaData['pRF_path']))
    pRF = nibabel.load(metaData['pRF_path']).get_data()
    pRF = generate_shared_array(pRF,ctypes.c_double)
    
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
        p = Process(target=reconstruct_stimulus,args=(voxels,stimData,funcData,pRF))
        procs.append(p)
        p.start()
        
    # close the jobs
    for p in procs:
        p.join()
        
    return None
                