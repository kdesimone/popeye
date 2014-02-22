import datetime
import numpy as np

from config import init_config
from popeye.estimation import voxel_prf
import popeye.utilities as utils


# initialize the datasets as per config.py
func,meta = init_config()

# initialize a results list
output = []

# loop over each time-series and compute the prf
for voxel in meta['voxels']:
    
    # grab the voxel's time-series
    ts_vox = func['bold'][voxel,:]
    ts_vox = utils.zscore(ts_vox)
    
    # compute the prf estimate
    x, y, sigma, hrf_delay, err, stats, ts_model = voxel_prf(ts_vox, 
                                                             stimulus,
                                                             meta['bounds'],
                                                             norm_func=utils.zscore,
                                                             uncorrected_rval=0)
    
    # store the results
    output.append([voxel,x, y, sigma, hrf_delay, err, stats])
