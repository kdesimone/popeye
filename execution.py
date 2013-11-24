import datetime
import numpy as np

from config import init_config
import popeye

# initialize the datasets as per config.py
stimData,funcData,metaData = init_config()

# Randomize the voxel indices so that a particular core doesn't get stuck with
# all large sigmas ... 
[xi,yi,zi] = metaData['voxels']
randVec = np.random.rand(len(xi))
randInd = np.argsort(randVec)
metaData['voxels'] = tuple((xi[randInd],yi[randInd],zi[randInd]))

# # Run the pRF estimation
tic = datetime.datetime.now()
output = popeye.utilities.multiprocessor(popeye.estimation.compute_prf_estimate,
                                stimData['degXFine'],
                                stimData['degYFine'],
                                stimData['stimArrayFine'],
                                funcData,
                                metaData['bounds'],
                                metaData['core_voxels'],
                                metaData['uncorrected_rval'])
toc = datetime.datetime.now()
print("The pRF estimation took %s" %(toc-tic))
 
# # Write the results out as nifti_gz
popeye.utilities.recast_estimation_results_queue(output,metaData,True)

