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
    ###    Stimulus    ###
    ######################
    
    # set the visuotopic stimulus array path
    stimArrayPath = '/Users/kevin/Desktop/Etc/Analyzed_Data/2_41_42_43/pRF/barsArray_2_41.npy'
    
    # make sure the file exists
    if not shutil.os.path.exists(stimArrayPath):
        sys.exit('The stimulus array %s cannot be found!' %(stimArrayPath))
    
    # stimulus display parameters
    monitorWidth = 25.0 # distance across the width of the image on the projection screen in mm
    viewingDistance = 38.0 # viewing distance from the subject's eye to the projection screen in mm
    pixelsAcross = 800 # display resolution across in pixels
    pixelsDown = 600 # display resolution down in pixels
    pixelsPerDegree = np.pi * pixelsAcross / np.arctan(monitorWidth/viewingDistance/2.0) / 360.0 # degrees of visual angle
    clipNumber = 10 # TRs to remove at the beginning
    rollNumber = -2 # TRs to rotate the time-series.
    scaleFactor = 0.05 # Decimal describing how much to down-sample the stimulus for increased fitting speed
    
    # the non-resampled stimulus array
    stimArrayFine = np.load(stimArrayPath)
    stimArrayFine = stimArrayFine[:,:,clipNumber::]
    stimArrayFine = np.roll(stimArrayFine,rollNumber,axis=-1)
    stimArrayFine = resample_stimulus(stimArrayFine,1.0)
    degXFine,degYFine = generate_coordinate_matrices(pixelsAcross,pixelsDown,pixelsPerDegree,1.0)
    
    # the resampled stimulus array
    stimArrayCoarse = resample_stimulus(stimArrayFine,scaleFactor)
    degXCoarse,degYCoarse = generate_coordinate_matrices(pixelsAcross,pixelsDown,pixelsPerDegree,scaleFactor)
    
    
    ######################
    ###   Functional   ###
    ######################
    
    # set functional dataset path
    funcPath = '/Users/kevin/Desktop/Etc/Analyzed_Data/2_41_42_43/pRF/2_41_42_43.nii.gz'
    
    # make sure it is accessible on the file-system
    if not shutil.os.path.exists(funcPath):
        sys.exit('The functional dataset %s cannot be found!' %(funcPath))
    
    # load and trim the leading TRs
    func = nibabel.load(funcPath)
    funcData = func.get_data()
    funcData = funcData[:,:,:,clipNumber::]
    
    ######################
    ###      Mask      ###
    ######################
    
    # set mask dataset path -- any binary mask where voxels-to-be analyzed are non-zero
    maskPath = '/Users/kevin/Desktop/Etc/Analyzed_Data/2_41_42_43/pRF/ROIs/anat_EPI_LGN_dil1x.nii.gz'
    
    # Make sure it is accessible on the file-system
    if not shutil.os.path.exists(maskPath):
        sys.exit('The mask dataset %s cannot be found!' %(maskPath))
    mask = nibabel.load(maskPath)
    maskData = mask.get_data()
    
    #######################
    ### Multiprocessing ###
    #######################
    
    # number of CPUs over which to distribute the work
    cpus = 23
    
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
    ###    Metadata    ###
    ######################
    
    # Set the output variables
    metaData = {}
    metaData['cpus'] = cpus
    metaData['outputPath'] = '/Users/kevin/Desktop/Etc/Analyzed_Data/2_41_42_43/pRF/'
    metaData['baseFileName'] = 'pRF_2_41_42_43'
    metaData['voxels'] = np.nonzero(maskData==1)
    metaData['maskPath'] = maskPath
    metaData['funcPath'] = funcPath
    metaData['verbose'] = True
    metaData['pRF_path'] = '/Users/kevin/Desktop/Etc/Analyzed_Data/2_41_42_43/pRF/pRF_2_41_42_43_tsmooth0_cartes.nii.gz'
    
    return stimData,funcData,metaData