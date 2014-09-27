import multiprocessing, time, shutil, ctypes, gc
from itertools import repeat

import numpy as np
import nibabel
from scipy.signal import decimate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

import popeye.utilities as utils
from popeye.auditory_stimulus_1D import AuditoryStimulus
import popeye.strf as strf

# stimulus features
lo_freq = 20
hi_freq = 12000
Fs = hi_freq*2
NFFT = 4096
noverlap = NFFT*0.5
tr_length = 1.0
clip_number = 0
period = 32

base_path = '/Users/kevin/Desktop/2_51/pRF/'

# load the mask
mask = nibabel.load('%s/both_gm.nii.gz' %(base_path)).get_data()

# load the functional
nii = nibabel.load('%s/2_51_both_fwhm4.nii.gz' %(base_path))
hdr = nii.get_header()
dims = list(nii.shape[0:3])
num_timepoints = nii.shape[-1]

# load the signal representing the auditory stimulus
signal = np.load('/Users/kevin/Desktop/chirp.npy')
signal = signal[clip_number*Fs::]
    
# instantiate an instance of the Stimulus class
stimulus = AuditoryStimulus(signal, NFFT, Fs, noverlap, tr_length)

# spectrogram = stimulus.spectrogram
# times = stimulus.times
# freqs = stimulus.freqs
# freq_center, freq_sigma = 5000,500
# gaussian = np.exp(-(freqs - freq_center)**2 / freq_sigma**2)

# initialize the gaussian model
model = strf.SpectrotemporalModel(stimulus)

# load a statmap to subselect indices
statmap = nibabel.load('%s/2_51_both_fwhm4_fft.nii.gz' %(base_path)).get_data()
indices = np.nonzero((statmap[...,1] > 0.5) & (np.abs(statmap[...,1]) != np.inf))
num_voxels = len(indices[0])

# load the functional data
func = nii.get_data()[:,:,:,clip_number::]
timeseries = utils.zscore(func[indices])
func = []

# set some bounds
search_bounds = [((lo_freq, hi_freq),(lo_freq, hi_freq/2),)]*num_voxels
fit_bounds = [((lo_freq, hi_freq),(lo_freq,hi_freq/2),)]*num_voxels

# package the tuples, can't figure out how to do this with repeat and nested tuples
all_indices = []
for i in range(num_voxels):
    all_indices.append((indices[0][i],indices[1][i],indices[2][i]))



# save the results
# prf = strf.recast_estimation_results(output,nii)
# nibabel.save(prf,'/Users/kevin/Desktop/Etc/Analyzed_Data/S5_audio_chirp/pRF/prf_fwhm4_fb%d_tr1.5_no_noise.nii.gz' %(freq_window))
# nibabel.save(prf,'/Users/kevin/Desktop/tonotopy/prf_fwhm4_fb%d_tr1.nii.gz' %(freq_window))