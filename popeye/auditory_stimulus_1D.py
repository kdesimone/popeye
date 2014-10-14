"""

First pass at a stimulus model for abstracting the qualities and functionality of a stimulus
into an abstract class.  For now, we'll assume the stimulus model only pertains to visual 
stimuli on a visual display over time (i.e., 3D).  Hopefully this can be extended to other stimuli
with an arbitrary number of dimensions (e.g., auditory stimuli).

"""
from __future__ import division
import ctypes, gc

import numpy as np
from pylab import specgram
from numpy.lib import stride_tricks
from scipy.misc import imresize
import nibabel

from popeye.base import StimulusModel
from popeye.onetime import auto_attr
import popeye.utilities as utils

def generate_spectrogram(signal, NFFT, Fs, noverlap):
    
    spectrogram, freqs, times, handle = specgram(signal,NFFT=NFFT,Fs=Fs,noverlap=noverlap);
    
    return spectrogram, freqs, times

# This should eventually be VisualStimulus, and there would be an abstract class layer
# above this called Stimulus that would be generic for n-dimentional feature spaces.
class AuditoryStimulus(StimulusModel):
    
    """ Abstract class for stimulus model """
    
    
    def __init__(self, stim_arr, NFFT, Fs, noverlap, tr_length):
        
        # this is a weird notation
        StimulusModel.__init__(self, stim_arr)
        
        # absorb the vars
        self.NFFT = NFFT
        self.Fs = Fs
        self.noverlap = noverlap
        self.tr_length = tr_length
                
        spectrogram, freqs, times = generate_spectrogram(self.stim_arr, self.NFFT, self.Fs, self.noverlap)
        self.spectrogram = np.memmap('%s%s_%d.npy' %('/tmp/','spectrogram',self.__hash__()),dtype = np.double, mode = 'w+',shape = np.shape(spectrogram))
        self.spectrogram[:] = spectrogram[:]
        
        self.freqs = np.memmap('%s%s_%d.npy' %('/tmp/','freqs',self.__hash__()),dtype = np.double, mode = 'w+',shape = np.shape(freqs))
        self.freqs[:] = freqs[:]
        
        self.times = np.memmap('%s%s_%d.npy' %('/tmp/','times',self.__hash__()),dtype = np.double, mode = 'w+',shape = np.shape(times))
        self.times[:] = times[:]
        
        