import multiprocessing, time, shutil, ctypes
from itertools import repeat

import numpy as np
import nibabel

import popeye.utilities as utils
from popeye.visual_stimulus import VisualStimulus
import popeye.gaussian as gaussian
from popeye.utilities import recast_estimation_results

# stimulus features
clip_number = 10
viewing_distance = 38
screen_width = 25
num_steps = 20
tr_length = 1.5

# load the bar stimulus
bar = np.load('/Users/kevin/Desktop/Etc/Analyzed_Data/2_48_49_50/pRF/stim_digital.npy')

# create the stimulus model object
stim = VisualStimulus(bar, viewing_distance, screen_width, 0.05, clip_number, 0)

# set up bounds for the grid search
bounds = ((-10,10),(-10,10),(0.25,5.25),(-5,5),)

# initialize the gaussian model
model = gaussian.GaussianModel(stim)

# load the mask
mask = nibabel.load('/Users/kevin/Desktop/Etc/Analyzed_Data/2_48_49_50/pRF/bounding_box.nii.gz').get_data()
indices = np.nonzero(mask)
num_voxels = np.sum(mask)

# load the functional data
nii = nibabel.load('/Users/kevin/Desktop/Etc/Analyzed_Data/2_48_49_50/pRF/2_48_49_50_ss2_ts5.nii.gz')
hdr = nii.get_header()
func = nii.get_data()[:,:,:,clip_number::]
timeseries = func[indices]

# flush the func
func = []

# package the tuples, can't figure out how to do this with repeat and nested tuples
all_bounds = []
all_indices = []
for i in range(num_voxels):
    all_bounds.append(bounds)
    all_indices.append((indices[0][i],indices[1][i],indices[2][i]))

# package the data structure
dat = zip(timeseries,
          repeat(model,num_voxels),
          all_bounds,
          repeat(tr_length,num_voxels),
          all_indices,
          repeat(0.20,num_voxels),
          repeat(True,num_voxels))

# run analysis
num_cpus = multiprocessing.cpu_count()-1
pool = multiprocessing.Pool(num_cpus)
output = pool.map(gaussian.parallel_fit,dat)
pool.close()
pool.join()

# recast the results
grid_parent = nibabel.load('/Users/kevin/Desktop/Etc/Analyzed_Data/2_48_49_50/pRF/MGN_dil.nii.gz') 
cartes, polar = recast_estimation_results(output, grid_parent)


