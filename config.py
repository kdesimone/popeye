""""
This is a configuration file that sets filepaths for loading the stimulus, functional, and mask datasets.  In addition, 
it creates the visuotopic arrays for plotting the 2D gaussians in stimulus-referred coordinates. Shared-memory arrays will
be created for reading data from a multiprocessing.Array.  Results are garnered from a multiprocessing.Queue. 

TODO:  Create a multiprocessing.Queue for work-to-be-done.  This would pop jobs off the to-do stack and feed them into the main
pRF estimator method.  This will maximize the use of the user-specified available CPUs.


"""

import shutil, sys
import ctypes
from multiprocessing import Array
import numpy as np
import nibabel
from utilities import resample_stimulus, generate_coordinate_matrices, generate_shared_array

def init_config():
    
    ######################
    ###    Metadata    ###
    ######################
    # User-specified meta-data
    metaData = {}
    metaData['subject_id'] = ''
    metaData['cpus'] = 23
    metaData['outputPath'] = ''
    metaData['basePath'] = ''
    metaData['baseFileName'] = ''
    metaData['maskPath'] = ''
    metaData['funcPath'] = ''
    metaData['Bounds'] = () 
 
    ######################
    ###    Stimulus    ###
    ######################
    
    # set the visuotopic stimulus array path
    stimArrayPath = '%s/barsArray.npy' %(metaData['basePath'])
    
    # make sure the file exists
    if not shutil.os.path.exists(stimArrayPath):
        sys.exit('The stimulus array %s cannot be found!' %(stimArrayPath))
    
    # stimulus display parameters
    monitorWidth = 25.0 # distance across the width of the image on the projection screen in cm
    viewingDistance = 38.0 # viewing distance from the subject's eye to the projection screen in cm
    pixelsAcross = 800 # display resolution across in pixels
    pixelsDown = 600 # display resolution down in pixels
    pixelsPerDegree = np.pi * pixelsAcross / np.arctan(monitorWidth/viewingDistance/2.0) / 360.0 # degrees of visual angle
    clipNumber = 10 # TRs to remove at the beginning
    rollNumber = -2 # TRs to rotate the time-series.
    fineScaleFactor = 1.0 # Decimal describing how much to down-sample the stimulus for increased fitting speed
    coarseScaleFactor = 0.05 # Decimal describing how much to down-sample the stimulus for increased fitting speed
    
    # the non-resampled stimulus array
    stimArray = np.load(stimArrayPath)
    stimArray = stimArray[:,:,clipNumber::]
    stimArray = np.roll(stimArray,rollNumber,axis=-1)
    stimArrayFine = resample_stimulus(stimArray,fineScaleFactor)
    degXFine,degYFine = generate_coordinate_matrices(pixelsAcross,pixelsDown,pixelsPerDegree,fineScaleFactor)
    
    # the resampled stimulus array
    stimArrayCoarse = resample_stimulus(stimArray,coarseScaleFactor)
    degXCoarse,degYCoarse = generate_coordinate_matrices(pixelsAcross,pixelsDown,pixelsPerDegree,coarseScaleFactor)
    
    # create shared stimulus arrays and package them into a dict
    stimData = {}
    stimData['stimArrayFine'] = generate_shared_array(stimArrayFine,ctypes.c_short)
    stimData['stimArrayCoarse'] = generate_shared_array(stimArrayCoarse,ctypes.c_short)
    stimData['degXFine'] = generate_shared_array(degXFine,ctypes.c_double)
    stimData['degXCoarse'] = generate_shared_array(degXCoarse,ctypes.c_double)
    stimData['degYFine'] = generate_shared_array(degYFine,ctypes.c_double)
    stimData['degYCoarse'] = generate_shared_array(degYCoarse,ctypes.c_double)
    stimData['stimRecon'] = generate_shared_array(np.zeros_like(stimData['stimArrayFine'],dtype='double'),ctypes.c_double)
    
    
    ######################
    ###   Functional   ###
    ######################
    
    # make sure it is accessible on the file-system
    if not shutil.os.path.exists(metaData['funcPath']):
        sys.exit('The functional dataset %s cannot be found!' %(metaData['funcPath']))
    
    # load and trim the leading TRs
    bold = nibabel.load(metaData['funcPath']).get_data()
    
    # FIX THIS -- when 3dVol2Surf -> SurfSmooth -> 3dSurf2Vol, the EPI ends up being 5-D with a dummy dimension.  Haven't figured out how to fix this.
    if len(np.shape(bold)) > 4:
        bold = bold[:,:,:,0,:]
    
    # clip the first N-tps off the beginning, created shared array, and store the data into a dict
    funcData = {}
    funcData['bold'] = generate_shared_array(bold[:,:,:,clipNumber::],ctypes.c_double)
    
    # load the pRFs if they've been specified
    if metaData.has_key('pRF_cartes') and metaData['pRF_cartes']:
        funcData['pRF_cartes'] = generate_shared_array(nibabel.load(metaData['pRF_cartes']).get_data(),ctypes.c_double)
    if metaData.has_key('pRF_polar') and metaData['pRF_polar']:
        funcData['pRF_polar'] = generate_shared_array(nibabel.load(metaData['pRF_polar']).get_data(),ctypes.c_double)
    
    
    ######################
    ###      Mask      ###
    ######################
    
    # make sure it is accessible on the file-system
    if not shutil.os.path.exists(metaData['maskPath']):
        sys.exit('The mask dataset %s cannot be found!' %(metaData['maskPath']))
    
    # load the mask
    maskData = nibabel.load(metaData['maskPath']).get_data()
    
    # make sure the mask being used is in the same space as the functional data
    if funcData['bold'].shape[0:3] != maskData.shape[0:3]:
        sys.exit('The mask and functional datasetsar e not the same shape!\n\n%s\n%s' %(metaData['maskPath'],metaData['funcPath']))
    
    metaData['voxels'] = np.nonzero(maskData)
    
    return stimData,funcData,metaData
