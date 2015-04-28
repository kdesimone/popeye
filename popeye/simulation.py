from __future__ import division
from random import shuffle
from itertools import repeat
import time
import ctypes
import numpy as np

from scipy import ndimage
from scipy.optimize import fmin_powell, fmin
import nibabel

from popeye import og
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus, resample_stimulus
from popeye.spinach import generate_og_receptive_field, generate_og_receptive_fields


def recast_simulation_results(output, grid_parent):
    
    # load the gridParent
    dims = list(grid_parent.shape)
    dims = dims[0:3]
    dims.append(3)
    
    # initialize the statmaps
    estimates = np.zeros(dims)
    
    # extract the prf model estimates from the results queue output
    for o in output:
        voxel_index = o[0]
        estimates[voxel_index] = (o[1],float(o[2][0]),o[3])

        
    # get header information from the gridParent and update for the prf volume
    aff = grid_parent.get_affine()
    hdr = grid_parent.get_header()
    hdr.set_data_shape(dims)
    
    # recast as nifti
    nifti_estimates = nibabel.Nifti1Image(estimates,aff,header=hdr)
    
    return nifti_estimates

def error_function(neural_sigma, voxel_sigma, voxel_rf, deg_x, deg_y, xs, ys):
    
    if neural_sigma <= 0:
        return np.inf
    if neural_sigma > voxel_sigma:
        return np.inf
    
    # create all the neural rfs
    neural_rfs = generate_og_receptive_fields(deg_x, deg_y, xs, ys, neural_sigma)
    
    # normalize each rf by integral
    neural_rfs /= 2 * np.pi * neural_sigma ** 2
    
    # sum and divide by the number of neurons
    neural_rf = np.sum(neural_rfs,axis=-1)/neural_rfs.shape[-1]
    
    # RSS between neural and voxel
    error = np.sum((neural_rf-voxel_rf)**2)
    
    return error

def simulate_neural_sigma(estimate, scatter, deg_x, deg_y, voxel_index, num_neurons=1000, verbose=True):
    
    # timestamp
    start = time.clock()
    
    # unpack
    x = estimate[0]
    y = estimate[1]
    sigma = estimate[2]
    
    # create the Gaussian
    voxel_rf = generate_og_receptive_field(deg_x, deg_y, x, y, sigma)
    
    # normalize by integral
    voxel_rf /= 2*np.pi*sigma**2
    
    # generate random angles and scatters
    angles = np.random.uniform(0,2*np.pi,num_neurons)
    lengths = np.random.uniform(0,scatter,num_neurons)
    
    # convert to cartesian coords
    xs = x + np.sin(angles)*lengths
    ys = y + np.cos(angles)*lengths
    
    sigma_phat = fmin_powell(error_function, sigma, args=(sigma, voxel_rf, deg_x, deg_y, xs, ys),full_output=True,disp=False)
    
    # timestamp
    finish = time.clock()
    
    # progress
    if verbose:
        txt = ("VOXEL=(%.03d,%.03d,%.03d)   TIME=%.03d   OLD=%.02f  NEW=%.02f   SCATTER=%.02f" 
            %(voxel_index[0],
              voxel_index[1],
              voxel_index[2],
              finish-start,
              sigma,
              sigma_phat[0],
              scatter))
        
        print(txt)
    
    return (voxel_index,sigma,sigma_phat,scatter)

def parallel_simulate_neural_sigma(args):
    
    estimate = args[0]
    scatter = args[1]
    model = args[2]
    voxel_index = args[3]
    num_neurons = args[4]
    
    neural_sigma = simulate_neural_sigma(estimate, scatter, model.stimulus.deg_x, model.stimulus.deg_y, voxel_index, num_neurons)
    
    return neural_sigma

def generate_scatter_volume(roi, prf, threshold, iterations):
    
    scatter = np.zeros_like(roi,dtype='double')
    
    # get indices
    xi,yi,zi = np.nonzero((roi>0) & (prf[...,-3]>threshold))
    
    for voxel in range(len(xi)):
        
        # get index
        xind = xi[voxel]
        yind = yi[voxel]
        zind = zi[voxel]
        
        # set target to 1
        mask = np.zeros_like(roi)
        mask[xind,yind,zind] = 1
        
        # get neighborhood
        hood = ndimage.binary_dilation(mask,iterations=iterations)
        hood[xind,yind,zind] = 0
        
        fit = prf[mask==1][0]
        fits = prf[hood==1]
        scatter[xind,yind,zind] = np.mean(np.sqrt((fits[:,0]-fit[np.newaxis,0])**2 + (fits[:,1]-fit[np.newaxis,1])**2))/2
    
    return scatter