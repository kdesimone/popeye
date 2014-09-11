"""

First pass at a stimulus model for abstracting the qualities and functionality of a stimulus
into an abstract class.  For now, we'll assume the stimulus model only pertains to visual 
stimuli on a visual display over time (i.e., 3D).  Hopefully this can be extended to other stimuli
with an arbitrary number of dimensions (e.g., auditory stimuli).

"""
from __future__ import division
import ctypes, gc

import numpy as np
from numpy.lib import stride_tricks
from scipy.misc import imresize
from popeye.onetime import auto_attr
import nibabel

from popeye.base import StimulusModel

def compute_stft(signal, freq_window, overlap=0.5, window=np.hanning):
    
    win = window(freq_window)
    hop_size = int(freq_window - np.floor(overlap * freq_window))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(freq_window/2.0)), signal)
    
    # cols for windowing
    cols = np.ceil( (len(samples) - freq_window) / float(hop_size)) + 1
    
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(freq_window))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, freq_window), strides=(samples.strides[0]*hop_size, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)
    
def logscale_stft(stft, sapmling_rate, scale, timebins):
    
    # create stft with new freq bins
    log_stft = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            log_stft[:,i] = np.sum(stft[:,scale[i]:], axis=1)
        else:
            log_stft[:,i] = np.sum(stft[:,scale[i]:scale[i+1]], axis=1)
    
    log_stft[log_stft == -np.inf] = 0
    
    return log_stft


def generate_spectrogram(log_stft, all_freqs, num_timepoints, tr_sampling_rate):
    
    s = 20.*np.log10(np.abs(log_stft)/10e-6)
    s[s == -np.inf] = 0
    s = np.transpose(s)
    s /= np.max(s)
    im = imresize(s,(len(all_freqs), num_timepoints * tr_sampling_rate))
    
    return im
    
def generate_coordinate_matrices(tr_times, all_freqs):
    
    return np.meshgrid(tr_times, all_freqs)

def rescale_frequencies(stimulus, scale_factor):
    
    dims = stimulus.shape
    scale_x = np.int(dims[0]*scale_factor)
    scale_y = dims[1]
    dims = (scale_x,scale_y)
    
    return imresize(stimulus, dims)
    
    

# This should eventually be VisualStimulus, and there would be an abstract class layer
# above this called Stimulus that would be generic for n-dimentional feature spaces.
class AuditoryStimulus(StimulusModel):
    
    """ Abstract class for stimulus model """
    
    
    def __init__(self, stim_arr, tr_length, freq_window = 2**10,
                 time_window = 0.1, freq_factor=1.0, sampling_rate=44100,
                 tr_sampling_rate=100, scale_factor=1.0):
        
        # this is a weird notation
        StimulusModel.__init__(self, stim_arr)
        
        if time_window > tr_length / 2:
            print("You must give a time window size in seconds that is half your TR length or less.")
            return None
        
        # absorb the vars
        self.tr_length = tr_length # the TR in s
        self.freq_factor = freq_factor # the scaling factor of the freqs, i think its only for plotting
        self.sampling_rate = sampling_rate # the sampling rate of the wav
        self.tr_sampling_rate = tr_sampling_rate # the number of samples from a single TR
        self.scale_factor = scale_factor
        self.time_window = time_window # in s, this is the number of slices we'll make for each TR
        self.freq_window = freq_window
        
        # share the arrays via memmap to reduce size  CHANGE THESE TO SINGLE?
        spectrogram = generate_spectrogram(self.log_stft, self.all_freqs, self.num_timepoints, self.tr_sampling_rate)
        self.spectrogram = np.memmap('%s%s_%d.npy' %('/tmp/','spectrogram',self.__hash__()),dtype = np.double, mode = 'w+',shape = np.shape(spectrogram))
        self.spectrogram[:] = spectrogram[:]
        
        time_coord, freq_coord = generate_coordinate_matrices(self.tr_times, self.all_freqs)
        self.time_coord = np.memmap('%s%s_%d.npy' %('/tmp/','time_coord',self.__hash__()),dtype = np.double, mode = 'w+',shape = np.shape(time_coord))
        self.time_coord[:] = time_coord[:]
        
        self.freq_coord = np.memmap('%s%s_%d.npy' %('/tmp/','freq_coord',self.__hash__()),dtype = np.double , mode = 'w+',shape = np.shape(freq_coord))
        self.freq_coord[:] = freq_coord[:]
        
        # create coarse versions of these
        spectrogram_coarse = rescale_frequencies(spectrogram, self.scale_factor)
        self.spectrogram_coarse = np.memmap('%s%s_%d.npy' %('/tmp/','spectrogram_coarse',self.__hash__()),dtype = np.double, mode = 'w+',shape = np.shape(spectrogram_coarse))
        self.spectrogram_coarse[:] = spectrogram_coarse[:]
        
        time_coord_coarse = rescale_frequencies(time_coord, self.scale_factor)
        self.time_coord_coarse = np.memmap('%s%s_%d.npy' %('/tmp/','time_coord_coarse',self.__hash__()),dtype = np.double, mode = 'w+',shape = np.shape(time_coord_coarse))
        self.time_coord_coarse[:] = time_coord_coarse[:]
        
        freq_coord_coarse = rescale_frequencies(freq_coord, self.scale_factor)
        self.freq_coord_coarse = np.memmap('%s%s_%d.npy' %('/tmp/','freq_coord_coarse',self.__hash__()),dtype = np.double, mode = 'w+',shape = np.shape(freq_coord_coarse))
        self.freq_coord_coarse[:] = time_coord_coarse[:]
        
        # collect garbage
        gc.collect()
        
        
    @auto_attr
    def num_timepoints(self):
        return np.int(self.stim_arr.shape[0]/self.sampling_rate)
    
    @auto_attr
    def tr_times(self):
        return np.linspace(self.all_freqs[0],self.all_freqs[-1],self.tr_sampling_rate)
    
    @auto_attr
    def stft(self):
        return compute_stft(self.stim_arr, self.freq_window)
    
    @auto_attr
    def timebins(self):
        return np.shape(self.stft)[0]
    
    @auto_attr 
    def freqbins(self):
        return np.shape(self.stft)[1]
    
    @auto_attr
    def scale(self):
        scale = np.linspace(0, 1, self.freqbins) ** self.freq_factor
        scale *= (self.freqbins-1)/max(scale)
        scale = np.unique(np.round(scale))
        return scale
    
    @auto_attr
    def all_freqs(self):
        return np.abs(np.fft.fftfreq(self.freqbins*2, 1./self.sampling_rate)[:self.freqbins+1])
    
    @auto_attr
    def scaled_freqs(self):
        freqs = []
        for i in np.arange(0, len(self.scale)):
            if i == len(self.scale)-1:
                freqs += [np.mean(self.all_freqs[self.scale[i]:])]
            else:
                freqs += [np.mean(self.all_freqs[self.scale[i]:self.scale[i+1]])]
        return freqs
    
    @auto_attr
    def all_times(self):
        return np.arange(0,self.num_timepoints,self.time_window)
    
    @auto_attr
    def log_stft(self):
        return logscale_stft(self.stft, self.sampling_rate, self.scale, self.timebins)
    
    
    @auto_attr
    def scaled_spectrogram(self):
        if self.scale_factor < 1:
            return imresize(self.spectrogram,self.scale_factor)
        else:
            return None
    
    @auto_attr
    def scaled_coordinate_matrices(self):
        if self.scale_factor < 1:
            ind = np.arange(0, len(self.all_freqs), 1/self.scale_factor).astype('int16')
            freqs = self.all_freqs[ind]
            X,Y = np.meshgrid(self.tr_times, self.all_freqs[ind])
            return X,Y
        else:
            return None
    
    @auto_attr
    def scaled_freq_coord(self):
         return self.scaled_coordinate_matrices[1]
         
    @auto_attr
    def scaled_time_coord(self):
        return self.scaled_coordinate_matrices[0]
    
    @auto_attr
    def scaled_num_timepoints(self):
        if self.scale_factor < 1:
            return self.scaled_spectrogram.shape[-1]
        else:
            return None
