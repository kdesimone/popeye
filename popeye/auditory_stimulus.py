"""

First pass at a stimulus model for abstracting the qualities and functionality of a stimulus
into an abstract class.  For now, we'll assume the stimulus model only pertains to visual 
stimuli on a visual display over time (i.e., 3D).  Hopefully this can be extended to other stimuli
with an arbitrary number of dimensions (e.g., auditory stimuli).

"""
from __future__ import division
import ctypes

from scipy.signal import spectrogram
import numpy as np

from popeye.base import StimulusModel
from popeye.onetime import auto_attr
import popeye.utilities as utils

def generate_spectrogram(signal, Fs, tr_length, noverlap=0, bins_per_octave = 5*20, min_freq = 20, logspace=True):
    
    # window size is 1 TR x samples per seconds
    win = Fs*tr_length
    
    # find num freqs
    nfft=win
    
    # get spectrum
    F,T,Sx = spectrogram(signal, Fs, nperseg=win, noverlap=noverlap, nfft=nfft)
    
    
    if logspace:
        
        # Ratio between adjacent frequencies in log-f axis
        fratio = 2**(1./octaves)
        
        # How many bins in log-f axis
        nbins = floor( np.log10((Fs/2)/freq_min) / np.log10(fratio) )
        
        # Freqs corresponding to each bin in FFT
        fftfrqs = np.arange(nfft/2)*(Fs/nfft)
        nfftbins = nfft/2;
        
        # Freqs corresponding to each bin in log F output
        logffrqs = freq_min * np.exp(np.log10(2)*np.arange(nbins)/octaves);
        
        # Bandwidths of each bin in log F
        logfbws = logffrqs * (fratio - 1)
        
        # # .. but bandwidth cannot be less than FFT binwidth
        # logfbws = np.max(logfbws, Fs/nfft,0)
        
        ovfctr = 0.5475;   # Adjusted by hand to make sum(mx'*mx) close to 1.0
        
        
        
        
        # Weighting matrix mapping energy in FFT bins to logF bins
        # is a set of Gaussian profiles depending on the difference in 
        # frequencies, scaled by the bandwidth of that bin
        A = np.tile(logffrqs,(nfftbins,1)).T
        B = np.tile(fftfrqs,(nbins,1))
        C = np.tile(ovfctr*logfbws,(nfftbins,1)).T
        freqdiff = ( A - B )/C;
        
        
        # % Normalize rows by sqrt(E), so multiplying by mx' gets approx orig spectrum back
        mx = exp( -0.5*freqdiff**2 );
        D = np.sqrt(2*np.sum(mx**2,1))
        E = mx / np.tile(D, (nfftbins,1)).T
        Px = np.matrix(Sx[1::])
        spec = np.array(np.matrix(mx) * Px)
        
        # output
        times = np.arange(spec.shape[-1])
        
        return spec, freqs, times
        
class AuditoryStimulus(StimulusModel):
    
    
    def __init__(self, signal, dtype, tr_length):
        
        r"""A child of the StimulusModel class for auditory stimuli.
        
        Paramaters
        ----------
        
        
        dtype : string
            Sets the data type the stimulus array is cast into.
        
        tr_length : float
            The repetition time (TR) in seconds.
            
        """
        
        # this is a weird notation
        StimulusModel.__init__(self, signal, Fs, tr_length, dtype)
        
        # absorb the vars
        self.signal = utils.generate_shared_array(signal, ctypes.c_double)
        self.Fs = Fs
        self.tr_length = tr_length
        
        # # create spectrogram
        specgram, freqs, times, = generate_spectrogram(signal, Fs, tr_length)
        
        # share them
        self.spectrogram = utils.generate_shared_array(specgram, ctypes.c_double)
        self.freqs = utils.generate_shared_array(freqs, ctypes.c_double)
        self.times = utils.generate_shared_array(times, ctypes.c_int16)
