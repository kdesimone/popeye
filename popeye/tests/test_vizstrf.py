from __future__ import division
import os
import multiprocessing
from itertools import repeat
import ctypes
import glob

import numpy as np
import numpy.testing as npt
import nose.tools as nt
from scipy.signal import fftconvolve, welch
from scipy.integrate import simps
from scipy.fftpack import fft

import popeye.utilities as utils
import popeye.og as og
from popeye.visual_stimulus import VisualStimulus, simulate_checkerboard_bar, resample_stimulus, simulate_movie_bar
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries    

pixels_across = 200
pixels_down = 150
viewing_distance = 38
screen_width = 25
thetas = np.arange(0,360,45)
blank_steps = 20
bar_steps = 30
bar_width = 6
ecc = 12
tr_length = 1
flicker_hz = 10
projector_hz = 60
threshold = 0.33
scale_factor = 0.25
dtype = ctypes.c_uint8
clip = 0.33 
fps = projector_hz
cpd = 0.50

thetas = np.insert(thetas,0,-1)
thetas = np.insert(thetas,2,-1)
thetas = np.insert(thetas,5,-1)
thetas = np.insert(thetas,8,-1)
thetas = np.insert(thetas,11,-1)
thetas = np.insert(thetas,len(thetas),-1)

frames_per_vol = tr_length*projector_hz
total_trs = len(thetas[thetas==-1])*blank_steps + len(thetas[thetas!=-1])*bar_steps
total_frames = frames_per_vol * total_trs
total_secs = total_trs*tr_length

# check =  simulate_checkerboard_bar(pixels_across, pixels_down, 
#                                  viewing_distance, screen_width, 
#                                  thetas, bar_steps, blank_steps,
#                                  bar_width, ecc, tr_length, cpd, flicker_hz,
#                                  projector_hz, dtype, clip)

images = glob.glob('/Users/kevin/Desktop/Etc/battleship_frames/*png')
# images = images[-total_secs*projector_hz::]
images = images[57000:57000+total_secs*projector_hz]

movie = simulate_movie_bar(images, pixels_across, pixels_down, 
                          viewing_distance, screen_width, 
                          thetas, bar_steps, blank_steps,
                          bar_width, ecc, tr_length, flicker_hz, 
                          projector_hz, dtype, clip)

# create an instance of the Stimulus class
stimulus = VisualStimulus(movie, viewing_distance, screen_width, scale_factor, dtype)

# spatial
x = 0
y = 0
s_sigma = 1
deg_x = stimulus.deg_x
deg_y = stimulus.deg_y
stim_arr = stimulus.stim_arr

# temporal
frames = np.arange(0,60)
mu = len(frames)/2
t_sigma = 2
hrf_delay = 0

# create Gaussian
srf = generate_og_receptive_field(deg_x, deg_y, x, y, s_sigma)

# divide by numeric integral
srf /= simps(simps(srf))

# extract the timeseries
response = generate_rf_timeseries(deg_x, deg_y, stim_arr, srf, x, y, s_sigma)


# A = np.abs(fft(response))
# plt.clf()
# plt.subplot(211)
# plt.plot(response)
# plt.xlim(0,len(response))
# plt.subplot(212)
# plt.plot(A[1:projector_hz/2])
# # plt.plot(A[1:len(A)/2])
# plt.grid('on')
# plt.show()



 

# # create sustained response for 1 TR
# sustained_1vol = exp(-((frames-mu)**2)/(2*t_sigma**2))
# sustained_1vol = utils.normalize(sustained_1vol,0,1)
# 
# # create transient response for 1 TR
# transient_1vol = np.diff(sustained_1vol)
# transient_1vol = utils.normalize(transient_1vol,-1,1)
# transient_1vol = np.append(transient_1vol,0)
# transient_1vol[0] = 0
# 
# # tile each response across the run
# transient = np.tile(transient_1vol,stim_arr.shape[-1]/fps)
# sustained = np.tile(sustained_1vol,stim_arr.shape[-1]/fps)
# 
# # extact the signal for each response
# sustained_response = response * sustained
# transient_response = response * transient
# 
# # take the mean of each 60 image TR
# sustained_ts = np.array([np.mean(sustained_response[t:t+fps]) for t in np.arange(0,len(response),fps)])
# transient_ts = np.array([np.mean(transient_response[t:t+fps]) for t in np.arange(0,len(response),fps)])
# spatial_ts = np.array([np.mean(response[t:t+fps]) for t in np.arange(0,len(response),fps)])
# 
# # add a constant to avoid division by zero
# sustained_ts += 127
# transient_ts += 127
# spatial_ts += 127
# 
# # demean
# sustained_pct = ((sustained_ts - np.mean(sustained_ts[15:20]))/np.mean(sustained_ts[15:20]))*100
# spatial_pct = ((spatial_ts - np.mean(spatial_ts[15:20]))/np.mean(spatial_ts[15:20]))*100
# transient_pct = ((transient_ts - np.mean(transient_ts[15:20]))/np.mean(transient_ts[15:20]))*100
# 
# #####################
# # convolution style #
# #####################
# sustained_response_conv = fftconvolve(response,sustained_1vol)[60:-60]
# transient_response_conv = fftconvolve(response,transient_1vol)[60:-60]
# 
# # take the mean of each 60 image TR
# sustained_conv_ts = np.array([np.mean(sustained_response_conv[t:t+fps]) for t in np.arange(0,len(response),fps)])
# transient_conv_ts = np.array([np.mean(transient_response_conv[t:t+fps]) for t in np.arange(0,len(response),fps)])
# 
# # add a constant to avoid division by zero
# sustained_ts += 127
# transient_ts += 127
# spatial_ts += 127
# 
# # demean
# sustained_conv_pct = ((sustained_ts - np.mean(sustained_ts[14:19]))/np.mean(sustained_ts[15:20]))*100
# transient_conv_pct = ((transient_ts - np.mean(transient_ts[14:19]))/np.mean(transient_ts[15:20]))*100
# 
# # create the HRF
# hrf = utils.double_gamma_hrf(hrf_delay, tr_length)
# 
# # convolve it with the stimulus
# sustained_conv_model = fftconvolve(sustained_conv_pct, hrf)[0:len(sustained_conv_pct)]
# transient_conv_model = fftconvolve(transient_conv_pct, hrf)[0:len(transient_conv_pct)]
# 
# # create the HRF
# hrf = utils.double_gamma_hrf(hrf_delay, tr_length)
# 
# # convolve it with the stimulus
# sustained_model = fftconvolve(sustained_pct, hrf)[0:len(sustained_pct)]
# transient_model = fftconvolve(transient_pct, hrf)[0:len(transient_pct)]
# spatial_model = fftconvolve(spatial_pct, hrf)[0:len(spatial_pct)]
# 
# # normalize it to 1
# sustained_norm = sustained_model/sustained_model.max()
# transient_norm = transient_model/transient_model.max()
# spatial_norm = spatial_model/spatial_model.max()
# 
# 
# 
# 
# 
# params = {'legend.fontsize': 10,
#           'legend.linewidth': 2}
# plt.rcParams.update(params)
# 
# fig = plt.figure()
# 
# ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
# # ax1.plot(spatial_norm,'b',lw=2,label='spatial')
# ax1.plot(sustained_norm,'k',lw=2,label='sustained')
# ax1.plot(transient_norm,'r',lw=2,label='transient')
# ax1.set_xlim(0,360)
# ax1.set_xlabel('Volumes')
# ax1.set_ylabel('Pct Change')
# ax1.set_title('Modeled Timeseries')
# ax1.legend(loc=0)
# 
# ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
# ax2.plot(response,'b',lw=2,label='spatial')
# ax2.set_xlim(0,len(response))
# ax2.set_xlabel('Frames')
# ax2.set_ylabel('Luminance')
# ax2.set_title('Stimulus Timeseries')
# 
# ax3 = plt.subplot2grid((3,3), (1, 2),rowspan=1,colspan=1)
# flip_idx = np.nonzero(full_run==1)[0][0]
# ax3.imshow(stim_arr[:,:,30*60+flip_idx],cmap=cm.gray,vmin=0,vmax=255)
# ax3.set_title('Frame 1')
# 
# ax4 = plt.subplot2grid((3,3), (2, 2),rowspan=1,colspan=1)
# flip_idx = np.nonzero(full_run==0)[0][0]
# ax4.imshow(stim_arr[:,:,30*60+flip_idx],cmap=cm.gray,vmin=0,vmax=255)
# ax4.set_title('Frame 2')
# 
# ax5 = plt.subplot2grid((3,3), (2, 0),rowspan=1,colspan=1)
# ax5.imshow(srf)
# ax5.set_title('Spatial tuning')
# 
# ax6 = plt.subplot2grid((3,3), (2, 1),rowspan=1,colspan=1)
# ax6.plot(sustained_1vol,'k',lw=2)
# ax6.plot(transient_1vol,'r',lw=2)
# ax6.set_xlim(0,60)
# ax6.set_title('Temporal tuning')
# ax6.set_ylabel('Amplitude')
# ax6.set_xlabel('Frames')
# 
# plt.tight_layout()
# 





