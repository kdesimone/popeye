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

# def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
#     win = window(frameSize)
#     hopSize = int(frameSize - np.floor(overlapFac * frameSize))
#     
#     # zeros at beginning (thus center of 1st window should be for sample nr. 0)
#     samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
#     # cols for windowing
#     cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
#     # zeros at end (thus samples can be fully covered by frames)
#     samples = np.append(samples, np.zeros(frameSize))
#     
#     frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
#     frames *= win
#     
#     return np.fft.rfft(frames)
# 
# def logscale_spec(spec, sr=44100, factor=20.):
#     timebins, freqbins = np.shape(spec)
#     
#     scale = np.linspace(0, 1, freqbins) ** factor
#     scale *= (freqbins-1)/max(scale)
#     scale = np.unique(np.round(scale))
#     
#     # create spectrogram with new freq bins
#     newspec = np.complex128(np.zeros([timebins, len(scale)]))
#     for i in range(0, len(scale)):
#         if i == len(scale)-1:
#             newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
#         else:        
#             newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
#             
#     # list center freq of bins
#     allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
#     freqs = []
#     for i in range(0, len(scale)):
#         if i == len(scale)-1:
#             freqs += [np.mean(allfreqs[scale[i]:])]
#         else:
#             freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
#             
#     return newspec, freqs
# 
# def generate_spectrogram(signal, NFFT, Fs, noverlap):
#     
#     s = stft(signal, NFFT)
#     sshow, freqs = logscale_spec(s, factor=1.0, sr=Fs)
#     spectrogram = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
#     
#     return spectrogram, freqs

def generate_spectrogram(signal, NFFT, Fs, noverlap):
    
    spectrogram, freqs, times, handle = specgram(signal,NFFT=NFFT,Fs=Fs,noverlap=noverlap);
    
    # # grab the data from the figure handle
    # dat = im.get_array().data
    # 
    # # set all infs to 0
    # dat[np.isinf(dat)] = 0
    # 
    # # the zeros cause a problem so we'll mask them
    # [dx,dy] = np.nonzero(dat==0)
    # 
    # # normalize to a known range
    # dat_pos = dat - np.max(dat)
    # dat_norm = utils.normalize(dat_pos,0,255)
    # 
    # # zero out the mask we created earlier
    # dat_norm[dx,dy] = 0
    # 
    # # # resize to the time-series data
    # # spectrogram = imresize(dat_norm,(freqs.shape[0],int(len(signal)/Fs)))
    # 
    # # flip it and standardize the range
    # spectrogram = utils.normalize(np.flipud(dat_norm),0,255)
    
    # return the spectrogram as well as the freqs
    # spectrogram = np.flipud(imresize(Pxx,(freqs.shape[0],int(len(signal)/Fs))))
    
    return spectrogram, freqs, times

def generate_stimulus_regressor(signal,Fs):
    
    # figure out how many time bins we have
    times = np.arange(0,len(signal),Fs)
    
    # digitize signal
    signal[signal!=0] = 1
    
    # initialize the output
    regressor = np.zeros(len(times))
    
    # loop over bins and binarize regressor
    for t in np.arange(len(times)):
        
        # pluck the time
        the_time = times[t]
        
        # look for any signal in that bin and store it
        signal_bin = signal[the_time:the_time+Fs]
        signal_sum = np.sum(signal_bin)/Fs
        regressor[t] = np.round(signal_sum)
    
    
    regressor = np.diff(regressor)
    regressor[regressor>0] = 1
    regressor = np.concatenate((np.zeros(1),regressor))
    
    return regressor
    

    
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
        
        regressor = generate_stimulus_regressor(self.stim_arr, self.Fs)
        self.regressor = np.memmap('%s%s_%d.npy' %('/tmp/','regressor',self.__hash__()),dtype = np.uint8, mode = 'w+',shape = np.shape(regressor))
        self.regressor[:] = regressor[:]
        
        self.freqs = np.memmap('%s%s_%d.npy' %('/tmp/','freqs',self.__hash__()),dtype = np.double, mode = 'w+',shape = np.shape(freqs))
        self.freqs[:] = freqs[:]
        
        self.times = np.memmap('%s%s_%d.npy' %('/tmp/','times',self.__hash__()),dtype = np.double, mode = 'w+',shape = np.shape(times))
        self.times[:] = times[:]
        
        