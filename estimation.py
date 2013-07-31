#! /usr/bin/env python2.6

from config import init_config
from utilities import multiprocess_prf_estimates, recast_results_queue_output

# initialize the datasets as per config.py
stimData,funcData,metaData = init_config()

# Run the pRF estimation 
output = multiprocess_prf_estimates(stimData,funcData,metaData)

# Write the results out as nifti_gz
recast_results_queue_output(output,metaData)