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
    ###    Stimulus    ###
    ######################
    
    # stimulus display parameters
    monitor_width = 25.0 # distance across the width of the image on the
                        # projection screen in cm 
    viewing_distance = 38.0 # viewing distance from the subject's eye to the
                           # projection screen in cm 
    pixels_across = 800 # display resolution across in pixels
    pixels_down = 600 # display resolution down in pixels
    ppd = np.pi*pixels_across/np.arctan(monitor_width/viewing_distance/2.0)/360.0 # degrees of visual angle
    clip_number = 10 # TRs to remove at the beginning
    roll_number = -2 # TRs to rotate the time-series.
    fine_scale = 1.0 # Decimal describing how much to down-sample the
                          # stimulus for increased fitting speed 
    coarse_scale = 0.05 # Decimal describing how much to down-sample the
                             # stimulus for increased fitting speed 
    
    # the non-resampled stimulus array
    stim_arr = np.load(meta_data['stim_path'])
    stim_arr = stim_arr[:,:,clip_number::]
    stim_arr = np.roll(stim_arr,roll_number,axis=-1)
    stim_arr_fine = prf.utilities.resample_stimulus(stim_arr,fine_scale)
    deg_x_fine,deg_y_fine = prf.utilities.generate_coordinate_matrices(pixels_across,
                                                                       pixels_down,
                                                                       ppd,
                                                                       fine_scale)
    
    # the resampled stimulus array
    stim_arr_coarse = prf.utilities.resample_stimulus(stim_arr,coarse_scale)
    deg_x_coarse,deg_y_coarse = prf.utilities.generate_coordinate_matrices(pixels_across,
                                                                           pixels_down,
                                                                           ppd,
                                                                           coarse_scale)
    
    # package it
    stim_data = {}
    stim_data['deg_x_coarse'] = deg_x_coarse
    stim_data['deg_y_coarse'] = deg_y_coarse
    stim_data['deg_x_fine'] = deg_x_fine
    stim_data['deg_y_fine'] = deg_y_fine
    stim_data['stim_arr_coarse'] = stim_arr_coarse
    stim_data['stim_arr_fine'] = stim_arr_fine
    
    
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
    
    return stim_data,func_data,meta_data