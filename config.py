""""
This is a configuration file that sets filepaths for loading the stimulus, functional, and mask datasets.  In addition, 
it creates the visuotopic arrays for plotting the 2D gaussians in stimulus-referred coordinates. Shared-memory arrays will
be created for reading data from a multiprocessing.Array.  Results are garnered from a multiprocessing.Queue. 

TODO:  Create a multiprocessing.Queue for work-to-be-done.  This would pop jobs off the to-do stack and feed them into the main
pRF estimator method.  This will maximize the use of the user-specified available CPUs.


"""

import shutil, sys
import ctypes
from multiprocessing import Array, Manager
import numpy as np
import nibabel
from utilities import resample_stimulus, generate_coordinate_matrices

def init_config():
	
	######################
	###    Stimulus    ###
	######################
	
	# set the visuotopic stimulus array path
	stimArrayPath = '/Users/kevin/Desktop/Etc/Analyzed_Data/2_38_39/pRF/barsArray_mod.npy'
	
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
	degXFine,degYFine = generate_coordinate_matrices(pixelsAcross,pixelsDown,pixelsPerDegree,1.0)
	
	# the resampled stimulus array
	stimArrayCoarse = resample_stimulus(stimArrayFine,scaleFactor)
	degXCoarse,degYCoarse = generate_coordinate_matrices(pixelsAcross,pixelsDown,pixelsPerDegree,scaleFactor)
	
	
	######################
	###   Functional   ###
	######################
	
	# set functional dataset path
	funcPath = '/Users/kevin/Desktop/Etc/Analyzed_Data/2_38_39/pRF/2_38_39.nii.gz'
	
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
	maskPath = '/Users/kevin/Desktop/Etc/Analyzed_Data/2_38_39/pRF/anat_EPI_mask_2mm.nii.gz'
	
	# Make sure it is accessible on the file-system
	if not shutil.os.path.exists(maskPath):
		sys.exit('The mask dataset %s cannot be found!' %(maskPath))
	mask = nibabel.load(maskPath)
	maskData = mask.get_data()
	
	
	#######################
	### Multiprocessing ###
	#######################
	
	# number of CPUs over which to distribute the work
	jobs = 23 
	
	# Stimulus arrays
	shared_array_base = Array(ctypes.c_short,np.prod(np.shape(stimArrayFine)))
	stimArrayFine_mem = np.ctypeslib.as_array(shared_array_base.get_obj())
	stimArrayFine_mem = np.reshape(stimArrayFine_mem,np.shape(stimArrayFine))
	stimArrayFine_mem[:,:,:] = stimArrayFine[:,:,:]
	stimArrayFine = stimArrayFine_mem
	stimArrayFine_mem = []
	
	shared_array_base = Array(ctypes.c_short,np.prod(np.shape(stimArrayCoarse)))
	stimArrayCoarse_mem = np.ctypeslib.as_array(shared_array_base.get_obj())
	stimArrayCoarse_mem = np.reshape(stimArrayCoarse_mem,np.shape(stimArrayCoarse))
	stimArrayCoarse_mem[:,:,:] = stimArrayCoarse[:,:,:]
	stimArrayCoarse = stimArrayCoarse_mem
	stimArrayCoarse_mem = []
	
	# Visuotpic coordinate matrices
	shared_array_base = Array(ctypes.c_double,np.prod(np.shape(degXFine)))
	degXFine_mem = np.ctypeslib.as_array(shared_array_base.get_obj())
	degXFine_mem = np.reshape(degXFine_mem,np.shape(degXFine))
	degXFine_mem[:,:] = degXFine[:,:]
	degXFine = degXFine_mem
	degXFine_mem = []
	
	shared_array_base = Array(ctypes.c_double,np.prod(np.shape(degXCoarse)))
	degXCoarse_mem = np.ctypeslib.as_array(shared_array_base.get_obj())
	degXCoarse_mem = np.reshape(degXCoarse_mem,np.shape(degXCoarse))
	degXCoarse_mem[:,:] = degXCoarse[:,:]
	degXCoarse = degXCoarse_mem
	degXCoarse_mem = []
	
	shared_array_base = Array(ctypes.c_double,np.prod(np.shape(degYFine)))
	degYFine_mem = np.ctypeslib.as_array(shared_array_base.get_obj())
	degYFine_mem = np.reshape(degYFine_mem,np.shape(degYFine))
	degYFine_mem[:,:] = degYFine[:,:]
	degYFine = degYFine_mem
	degYFine_mem = []
	
	shared_array_base = Array(ctypes.c_double,np.prod(np.shape(degYCoarse)))
	degYCoarse_mem = np.ctypeslib.as_array(shared_array_base.get_obj())
	degYCoarse_mem = np.reshape(degYCoarse_mem,np.shape(degYCoarse))
	degYCoarse_mem[:,:] = degYCoarse[:,:]
	degYCoarse = degYCoarse_mem
	degYCoarse_mem = []
	
	# package the arrays into a dict
	stimData = {}
	stimData['stimArrayFine'] = stimArrayFine
	stimData['stimArrayCoarse'] = stimArrayCoarse
	stimData['degXFine'] = degXFine
	stimData['degXCoarse'] = degXCoarse
	stimData['degYFine'] = degYFine
	stimData['degYCoarse'] = degYCoarse
	
	# functional data
	shared_array_base = Array(ctypes.c_double,np.prod(np.shape(funcData)))
	funcData_mem =  np.ctypeslib.as_array(shared_array_base.get_obj())
	funcData_mem = np.reshape(funcData_mem,np.shape(funcData))
	funcData_mem[:,:,:,:] = funcData[:,:,:,:]
	funcData = funcData_mem
	funcData_mem = []
	
	######################
	###    Metadata    ###
	######################
	
	# Set the output variables
	metaData = {}
	metaData['jobs'] = jobs
	metaData['outputPath'] = '/Users/kevin/Desktop/'
	metaData['baseFileName'] = 'pRF_2_38_39'
	metaData['voxels'] = np.nonzero(maskData)
	metaData['maskPath'] = maskPath
	metaData['funcPath'] = funcPath
	
	return stimData,funcData,metaData