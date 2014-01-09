import datetime
import numpy as np

from config import init_config
from popeye.estimation import voxel_prf
import popeye.utilities as utils


# initialize the datasets as per config.py
stim,func,meta = init_config()

# initialize a results list
output = []

# loop over each time-series and compute the prf
for voxel in meta['voxels']:
    
    # grab the voxel's time-series
    ts_vox = func['bold'][voxel,:]
    ts_vox = utils.zscore(ts_vox)
    
    # compute the prf estimate
    x, y, sigma, hrf_delay, err, stats = voxel_prf(ts_vox, 
                                                   stim['deg_x_coarse'],
                                                   stim['deg_y_coarse'],
                                                   stim['deg_x_fine'],
                                                   stim['deg_y_fine'],
                                                   stim['stim_arr_coarse'],
                                                   stim['stim_arr_fine'],
                                                   meta['bounds'],
                                                   norm_func=utils.zscore,
                                                   uncorrected_rval=0)
    
    # store the results
    output.append([voxel,x, y, sigma, hrf_delay, err, stats])
