import datetime
import time
import ctypes
import numpy as np
import nibabel
from config import init_config
from utilities import multiprocessor, recast_simulation_results_queue, generate_shared_array, recast_estimation_results_queue
from estimation import compute_prf_estimate
from simulation import simulate_neural_sigma
from reconstruction import multiprocess_stimulus_reconstruction

# initialize the datasets as per config.py
stimData,funcData,metaData = init_config()

# Randomize the voxel indices so that a particular core doesn't get stuck with all large sigmas ...
[xi,yi,zi] = metaData['voxels']
randVec = np.random.rand(len(xi))
randInd = np.argsort(randVec)
metaData['voxels'] = tuple((xi[randInd],yi[randInd],zi[randInd]))

# # Run the pRF estimation
tic = datetime.datetime.now()
output = multiprocessor(compute_prf_estimate,stimData,funcData,metaData)
toc = datetime.datetime.now()
print("The pRF estimation took %s" %(toc-tic))
 
# # Write the results out as nifti_gz
recast_estimation_results_queue(output,metaData,True)

