""""
This is a configuration file that sets filepaths for 
loading the sample stimulus and BOLD functional time-series data.
In addition, it creates the visuotopic arrays for plotting the 2D 
gaussians in stimulus-referred coordinates. 

"""

import shutil, sys, os
import ctypes
from multiprocessing import Array
import numpy as np
import nibabel
import popeye as prf

def init_config():
    
    ######################
    ###    Metadata    ###
    ######################
    
    # User-specified meta-data
    
    meta_data = {}
    meta_data['subject_id'] = ''
    meta_data['cpus'] = 1
    meta_data['output_path'] = os.path.join(os.path.expanduser('~'), '.popeye/popeye')
    meta_data['base_path'] =  meta_data['output_path']
    meta_data['base_filename'] = ''
    meta_data['mask_path'] = ''
    meta_data['func_path'] = '%s/sample_response.npy' %(meta_data['base_path'])
    meta_data['stim_path'] = '%s/sample_stimulus.npy' %(meta_data['base_path'])
    meta_data['bounds'] = ((-10,10),(-10,10),(0.25,5.25),(-4,4))
    
    ######################
    ###   Functional   ###
    ######################
    
    # load and trim the leading TRs
    bold = np.load(meta_data['func_path'])
    
    # clip the first N-tps off the beginning, created shared array, and store
    # the data into a dict
    func_data = {}
    func_data['bold'] = prf.utilities.generate_shared_array(bold[:,clip_number::],ctypes.c_double)
    
    ######################
    ###      Mask      ###
    ######################
    
    # load the mask -- here we're just taking all the voxels in the sample dataset
    maskData = np.ones_like(func_data['bold'][:,0])
    
    # grab the indices
    meta_data['voxels'] = np.nonzero(maskData)[0]
    
    return func_data,meta_data