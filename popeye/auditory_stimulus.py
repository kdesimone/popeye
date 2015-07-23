"""

First pass at a stimulus model for abstracting the qualities and functionality of a stimulus
into an abstract class.  For now, we'll assume the stimulus model only pertains to visual 
stimuli on a visual display over time (i.e., 3D).  Hopefully this can be extended to other stimuli
with an arbitrary number of dimensions (e.g., auditory stimuli).

"""
from __future__ import division
import ctypes

import matplotlib
matplotlib.use('Agg')
from matplotlib.mlab import specgram

import numpy as np
from numpy.lib import stride_tricks
from scipy.misc import imresize
import nibabel

from popeye.base import StimulusModel
from popeye.onetime import auto_attr
import popeye.utilities as utils

def generate_spectrogram(signal, NFFT, Fs, noverlap):
    
    """
    A Gaussian population receptive field model [1]_.
    
    Paramaters
    ----------
    
    signal : ndarray
        A 1D array containg the monophonic auditory stimulus.
    
    NFFT : integer
      The number of data points used in each block for the FFT.
      Must be even; a power 2 is most efficient.  The default value is 256.
      This should *NOT* be used to get zero padding, or the scaling of the
      result will be incorrect. Use *pad_to* for this instead.
    
     Fs : scalar
      The sampling frequency (samples per time unit).  It is used
      to calculate the Fourier frequencies, freqs, in cycles per time
      unit. The default value is 2.
    
    
    noverlap : integer
        The number of points of overlap between blocks.  The default value 
        is 128.
    
    
    For more information, see help of `pylab.specgram`.

    """
    
    
    spectrogram, freqs, times = specgram(signal,NFFT=NFFT,Fs=Fs,noverlap=noverlap);
    
    print(spectrogram.shape)
    print(np.unique(spectrogram))
    
    return spectrogram, freqs, times

class AuditoryStimulus(StimulusModel):
    
    
    def __init__(self, stim_arr, NFFT, Fs, noverlap, dtype,tr_length):
        
        
        """
        A child of the StimulusModel class for auditory stimuli.
        
        Paramaters
        ----------
        
        signal : ndarray
            A 1D array containg the monophonic auditory stimulus.
        
        NFFT : integer
          The number of data points used in each block for the FFT.
          Must be even; a power 2 is most efficient.  The default value is 256.
          This should *NOT* be used to get zero padding, or the scaling of the
          result will be incorrect. Use *pad_to* for this instead.
        
         Fs : scalar
          The sampling frequency (samples per time unit).  It is used
          to calculate the Fourier frequencies, freqs, in cycles per time
          unit. The default value is 2.
        
        noverlap : integer
            The number of points of overlap between blocks.  The default value 
            is 128.
        
        dtype : string
            Sets the data type the stimulus array is cast into.
        
        tr_length : float
            The repetition time (TR) in seconds.
            
        """
        
        # this is a weird notation
        StimulusModel.__init__(self, stim_arr, dtype, tr_length)
        
        # absorb the vars
        self.NFFT = NFFT
        self.Fs = Fs
        self.noverlap = noverlap
        
        # create the vars via matplotlib
        spectrogram, freqs, times = generate_spectrogram(self.stim_arr, self.NFFT, self.Fs, self.noverlap)
        
        # share them
        self.spectrogram = utils.generate_shared_array(spectrogram, ctypes.c_double)
        self.freqs = utils.generate_shared_array(freqs, ctypes.c_double)
        
        # # why don't the times returned from specgram start at 0? they are time bin centers?
        # self.target_times = utils.generate_shared_array(target_times, ctypes.c_double)