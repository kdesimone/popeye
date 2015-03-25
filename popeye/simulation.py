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
    
    print(error,neural_sigma)
    
    return error

def simulate_neural_sigma(estimate, scatter, deg_x, deg_y, num_neurons=1000):
    
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
        
    return sigma_phat[0]

def parallel_simulate_neural_sigma(args):
    
    estimate = args[0]
    scatter = args[1]
    model = args[2]
    num_neurons = args[3]
    
    neural_sigma = simulate_neural_sigma(estimate, scatter, model.stimulus.deg_x, model.stimulus.deg_y, num_neurons)
    
    return neural_sigma

def generate_scatter_volume(roi, prf, threshold):
    
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
        hood = ndimage.binary_dilation(mask)
        hood[xind,yind,zind] = 0
        
        fit = prf[mask==1][0]
        fits = prf[hood==1]
        scatter[xind,yind,zind] = np.mean(np.sqrt((fits[:,0]-fit[np.newaxis,0])**2 + (fits[:,1]-fit[np.newaxis,1])**2))/2
    
    return scatter
    
# stimulus features
pixels_across = 800
pixels_down = 600
viewing_distance = 38
screen_width = 25
thetas = np.arange(0,360,360)
num_blank_steps = 20
num_bar_steps = 40
ecc = 12
tr_length = 1.0
frames_per_tr = 1.0
scale_factor = 0.10
resample_factor = 0.50
dtype = ctypes.c_uint8

# insert blanks
# thetas = list(thetas)
# thetas.insert(0,-1)
# thetas.insert(2,-1)
# thetas.insert(5,-1)
# thetas.insert(8,-1)
# thetas.insert(11,-1)
# thetas.append(-1)
# thetas = np.array(thetas)

# create the sweeping bar stimulus in memory
bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                            screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
    
# resample the stimulus
bar = resample_stimulus(bar, resample_factor)

# create an instance of the Stimulus class
stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, dtype)
model = og.GaussianModel(stimulus)
bar = []

# load the data
roi = nibabel.load('/Users/kevin/Desktop/rois_hemisphere.nii.gz').get_data()
prf = nibabel.load('/Users/kevin/Desktop/prf.nii.gz').get_data()

# extract voxels
mask = np.zeros_like(roi)
mask[roi>0] = 1
[xi,yi,zi] = np.nonzero((mask>0) & (prf[...,-3]>0.0625))
indices = [(xi[i],yi[i],zi[i]) for i in range(len(xi))]
num_voxels = len(xi)

# compute the scatter volume
scatter_vol = generate_scatter_volume(roi,prf,0.0625)

# get scatter and prfs
scatter = scatter_vol[xi,yi,zi]
estimates = prf[xi,yi,zi]

# package it for parallel
dat = zip(estimates,
          scatter,
          repeat(model, num_voxels),
          repeat(100, num_voxels))

# shuffle(dat)

# single voxel scatter and estimate
# xind, yind, zind = 73, 17, 24
# scatter = scatter_vol[xind,yind,zind]
# estimate = prf[xind,yind,zind]
# neural_sigma = simulate_neural_sigma(estimate, scatter, stimulus.deg_x, stimulus.deg_y, 5)




