cimport cython
from cython.parallel import prange, parallel, threadid
import numpy as np
cimport numpy as np
from scipy.signal.sigtools import _correlateND
import ctypes

DTYPE = np.int
ctypedef np.int_t DTYPE_t

DTYPE2 = np.double
ctypedef np.double_t DTYPE2_t

# DTYPE3 = np.short
# ctypedef np.short DTYPE3_t

from libc.math cimport sin, cos, exp, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
def two_dimensional_og(np.ndarray[DTYPE2_t, ndim=2] deg_x,
                       np.ndarray[DTYPE2_t, ndim=2] deg_y,
                       DTYPE2_t x,
                       DTYPE2_t y,
                       DTYPE2_t sigma_x,
                       DTYPE2_t sigma_y,
                       DTYPE2_t degrees,
                       DTYPE2_t amplitude):
    
    # cdef's
    cdef int i,j,k
    cdef DTYPE2_t s_factor2
    cdef DTYPE2_t s_factor3
    cdef int xlim = deg_x.shape[0]
    cdef int ylim = deg_y.shape[1]
    cdef DTYPE2_t d
    
    # convert degress to radians
    cdef DTYPE2_t theta = degrees * np.pi/180
    
    # figure out which sigma is bigger and limit our search to within that
    if sigma_x > sigma_y:
        s_factor2 = (2.0*sigma_x**2)
        s_factor3 = (3.0*sigma_x)**2
    else:
        s_factor2 = (2.0*sigma_y**2)
        s_factor3 = (3.0*sigma_y)**2
    
    # set up the transformation matrix
    cdef DTYPE2_t a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    cdef DTYPE2_t b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    cdef DTYPE2_t c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    
    # initialize output variable
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] rf = np.zeros((xlim,ylim),dtype=DTYPE2)
    
    # the loop
    for i in xrange(xlim):
        for j in xrange(ylim):
            d = (deg_x[i,j]-x)**2 + (deg_y[i,j]-y)**2
            if d <= s_factor3:
                rf[i,j] = np.exp( - (a*(deg_x[i,j]-x)**2 + 2*b*(deg_x[i,j]-x)*(deg_y[i,j]-y) + c*(deg_y[i,j]-y)**2))
    
    # return it
    return rf

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_rf_timeseries(np.ndarray[DTYPE2_t, ndim=2] deg_x,
                           np.ndarray[DTYPE2_t, ndim=2] deg_y,
                           np.ndarray[short, ndim=3] stim_arr, 
                           np.ndarray[DTYPE2_t, ndim=2] rf,
                           DTYPE2_t x, DTYPE2_t y, DTYPE2_t sigma):
    
    # cdef's
    cdef int i,j,k
    cdef DTYPE2_t s_factor3 = (3.0*sigma)**2
    cdef int xlim = stim_arr.shape[0]
    cdef int ylim = stim_arr.shape[1]
    cdef int zlim = stim_arr.shape[2]
    cdef DTYPE2_t d, gauss1D
    
    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(zlim,dtype=DTYPE2)
    
    # the loop
    for i in xrange(xlim):
        for j in xrange(ylim):
            d = (deg_x[i,j]-x)**2 + (deg_y[i,j]-y)**2
            if d <= s_factor3:
                for k in xrange(zlim):
                    stim[k] += stim_arr[i,j,k]*rf[i,j]
    
    return stim

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_dog_timeseries(np.ndarray[DTYPE2_t, ndim=2] deg_x,
                                 np.ndarray[DTYPE2_t, ndim=2] deg_y,
                                 np.ndarray[short, ndim=3] stim_arr,
                                 DTYPE2_t x, DTYPE2_t y, 
                                 DTYPE2_t sigma_center, 
                                 DTYPE2_t sigma_surround):
                                 
    """
    Generate a time-series given a stimulus array and Gaussian parameters.
    
    Parameters
    ----------
    deg_x : 2D array
            The coordinate matrix along the horizontal dimension of the display (degrees)
    deg_y : 2D array
            The coordinate matrix along the vertical dimension of the display (degrees)
    x : float
       The x coordinate of the center of the Gaussian (degrees)
    y : float
       The y coordinate of the center of the Gaussian (degrees)
    s : float
       The dispersion of the Gaussian (degrees)
    beta : float
        The amplitude of the Gaussian
       
    Returns
    
    stim : ndarray
        The 1D array containing the stimulus energies given the Gaussian coordinates
    
    """
    
    # cdef's
    cdef int i,j,k,num_center_pixels, num_surround_pixels
    
    cdef DTYPE2_t sigma_center_factor2 = (2.0*sigma_center**2)
    cdef DTYPE2_t sigma_center_factor3 = (3.0*sigma_center)**2
    
    cdef DTYPE2_t sigma_surround_factor2 = (2.0*sigma_surround**2)
    cdef DTYPE2_t sigma_surround_factor3 = (3.0*sigma_surround)**2
    
    cdef int xlim = deg_x.shape[0]
    cdef int ylim = deg_x.shape[1]
    cdef int zlim = stim_arr.shape[2]
    cdef int add_center = 0
    cdef int add_surround = 0
    
    cdef DTYPE2_t d, gauss1D_center, gauss1D_surround
    cdef DTYPE2_t sum_gauss_center = 0.0
    cdef DTYPE2_t sum_gauss_surround = 0.0

    
    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim_center = np.zeros(zlim,dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim_surround = np.zeros(zlim,dtype=DTYPE2)

    
    # the loop
    for i in xrange(xlim):
        for j in xrange(ylim):
            
            # find out the distance of the pixel from the prf
            d = (deg_x[i,j]-x)**2 + (deg_y[i,j]-y)**2
            
            # if its inside the surround ...
            if d <= sigma_surround_factor3:
                gauss1D_surround = exp(-d/sigma_surround_factor2)
                sum_gauss_surround += gauss1D_surround
                add_surround = 1
            else:
                add_surround = 0
            
            # if its inside the center ...
            if d <= sigma_center_factor3:
                gauss1D_center = exp(-d/sigma_center_factor2)
                sum_gauss_center += gauss1D_center
                add_center = 1
            else:
                add_center = 0
            
            # filter the pixel timeseries by the gauss
            for k in xrange(zlim):
                if add_center == 1:
                    stim_center[k] += stim_arr[i,j,k]*gauss1D_center
                    
                if add_surround == 1:
                    stim_surround[k] += stim_arr[i,j,k]*gauss1D_surround
            
            
    # scale it by the integral
    stim_center /= sum_gauss_center
    stim_surround /= sum_gauss_surround
    
    return stim_center, stim_surround
    
@cython.boundscheck(False)
@cython.wraparound(False)
def generate_og_timeseries(np.ndarray[DTYPE2_t, ndim=2] deg_x,
                           np.ndarray[DTYPE2_t, ndim=2] deg_y,
                           np.ndarray[short, ndim=3] stim_arr,
                           DTYPE2_t x, DTYPE2_t y, DTYPE2_t s):
                                 
    """
    Generate a time-series given a stimulus array and Gaussian parameters.
    
    Parameters
    ----------
    deg_x : 2D array
            The coordinate matrix along the horizontal dimension of the display (degrees)
    deg_y : 2D array
            The coordinate matrix along the vertical dimension of the display (degrees)
    x : float
       The x coordinate of the center of the Gaussian (degrees)
    y : float
       The y coordinate of the center of the Gaussian (degrees)
    s : float
       The dispersion of the Gaussian (degrees)
    beta : float
        The amplitude of the Gaussian
       
    Returns
    
    stim : ndarray
        The 1D array containing the stimulus energies given the Gaussian coordinates
    
    """
    
    # cdef's
    cdef int i,j,k,num_pixels_used
    cdef int xlim = deg_x.shape[0]
    cdef int ylim = deg_x.shape[1]
    cdef int zlim = stim_arr.shape[2]
    cdef DTYPE2_t pi = 3.14159265
    cdef DTYPE2_t gauss_area = 2.0*s**2
    cdef DTYPE2_t gauss_distance = (3.0*s)**2
    cdef DTYPE2_t gauss_integral = 2.0*pi*s**2
    cdef DTYPE2_t d, gauss1D
    
    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(zlim,dtype=DTYPE2)
    
    # the loop
    for i in xrange(xlim):
        for j in xrange(ylim):
            d = (deg_x[i,j]-x)**2 + (deg_y[i,j]-y)**2
            if d <= gauss_distance:
                gauss1D = exp(-d/gauss_area)
                for k in xrange(zlim):
                    stim[k] += stim_arr[i,j,k]*gauss1D/gauss_integral
                    
    return stim

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_strf_timeseries(np.ndarray[DTYPE2_t, ndim=1] freqs,
                                 np.ndarray[DTYPE2_t, ndim=2] spectrogram,
                                 DTYPE2_t center_freq, DTYPE2_t sd):
                                 
    """
    Generate a time-series given a stimulus array and Gaussian parameters.
    
    Parameters
    ----------
    deg_x : 2D array
            The coordinate matrix along the horizontal dimension of the display (degrees)
    deg_y : 2D array
            The coordinate matrix along the vertical dimension of the display (degrees)
    x : float
       The x coordinate of the center of the Gaussian (degrees)
    y : float
       The y coordinate of the center of the Gaussian (degrees)
    s : float
       The dispersion of the Gaussian (degrees)
    beta : float
        The amplitude of the Gaussian
       
    Returns
    
    stim : ndarray
        The 1D array containing the stimulus energies given the Gaussian coordinates
    
    """
    
    # cdef's
    cdef int i,j
    cdef DTYPE2_t s_factor2 = (2.0*sd**2)
    cdef DTYPE2_t s_factor3 = (3.0*sd)**2
    cdef int xlim = spectrogram.shape[0]
    cdef int ylim = spectrogram.shape[1]
    cdef DTYPE2_t d, gauss1D
    cdef DTYPE2_t sum_gauss = 0.0
    
    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(ylim,dtype=DTYPE2)
    
    # the loop
    for i in xrange(xlim):
        
        d = (freqs[i]-center_freq)**2

        for j in xrange(ylim):
            
            if d <= s_factor3:
                
                #compute gauss
                gauss1D = exp(-d/s_factor2)
                
                # filter spectrogram
                stim[j] += spectrogram[i,j]*gauss1D
                
    return stim
    
@cython.boundscheck(False)
@cython.wraparound(False)
def generate_og_receptive_field(np.ndarray[DTYPE2_t, ndim=2] deg_x,
                                np.ndarray[DTYPE2_t, ndim=2] deg_y,
                                DTYPE2_t x, DTYPE2_t y, DTYPE2_t sigma):
    """
    Generate a Gaussian.
    
    Parameters
    ----------
    deg_x : 2D array
            The coordinate matrix along the horizontal dimension of the display (degrees)
    deg_y : 2D array
            The coordinate matrix along the vertical dimension of the display (degrees)
    x : float
       The x coordinate of the center of the Gaussian (degrees)
    y : float
       The y coordinate of the center of the Gaussian (degrees)
    s : float
       The dispersion of the Gaussian (degrees)
    beta : float
        The amplitude of the Gaussian
       
    Returns
    
    stim : ndarray
        The 1D array containing the stimulus energies given the Gaussian coordinates
    
    """
    
    # cdef's
    cdef int i,j,k
    cdef DTYPE2_t s_factor2 = (2.0*sigma**2)
    cdef DTYPE2_t s_factor3 = (3.0*sigma)**2
    cdef int xlim = deg_x.shape[0]
    cdef int ylim = deg_x.shape[1]
    cdef DTYPE2_t d, gauss1D
    cdef DTYPE2_t pi = 3.14159265

    
    # initialize output variable
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] rf = np.zeros((xlim,ylim),dtype=DTYPE2)
    
    # the loop
    for i in xrange(xlim):
        for j in xrange(ylim):
            d = (deg_x[i,j]-x)**2 + (deg_y[i,j]-y)**2
            if d <= s_factor3:
                rf[i,j] = exp(-d/s_factor2)

    return rf

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_gabor_receptive_field(np.ndarray[DTYPE2_t, ndim=2] deg_x,
                                   np.ndarray[DTYPE2_t, ndim=2] deg_y,
                                   DTYPE2_t x0,
                                   DTYPE2_t y0,
                                   DTYPE2_t s0,
                                   DTYPE2_t theta,
                                   DTYPE2_t phi,
                                   DTYPE2_t cpd):
                  
    # cdef's
    cdef int i,j,k
    cdef DTYPE2_t s_factor2 = (2.0*s0**2)                                 
    cdef DTYPE2_t s_factor3 = (3.0*s0)**2
    cdef int xlim = deg_x.shape[0]
    cdef int ylim = deg_y.shape[1]
    cdef DTYPE2_t pi_180 = np.pi/180
    cdef DTYPE2_t theta_rad = theta * pi_180
    cdef DTYPE2_t phi_rad = phi * pi_180
    cdef DTYPE2_t pi_2 = np.pi * 2
    cdef DTYPE2_t d
    
    # fodder
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] XYt = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] XYf = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] grating = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] gauss = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] gabor = np.zeros((xlim,ylim),dtype=DTYPE2)
    
    # the loop
    for i in xrange(xlim):
        for j in xrange(ylim):
            
            # only compute inside the sigma*3
            d = (deg_x[i,j]-x0)**2 + (deg_y[i,j]-y0)**2
            
            if d <= s_factor3:
                
                # creating the grating
                XYt[i,j] =  (deg_x[i,j] * cos(theta_rad)) + (deg_y[i,j] * sin(theta_rad))
                XYf[i,j] = XYt[i,j] * cpd * pi_2
                grating[i,j] = sin(XYf[i,j] + phi_rad)
                
                # create the gaussian
                gauss[i,j] =  exp(-d/s_factor2)
                
                # create the gabor
                gabor[i,j] = gauss[i,j] * grating[i,j]
                
    return gabor

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_gabor_timeseries(np.ndarray[DTYPE2_t, ndim=2] deg_x,
                              np.ndarray[DTYPE2_t, ndim=2] deg_y,
                              np.ndarray[short, ndim=3] stim_arr,
                              DTYPE2_t x0,
                              DTYPE2_t y0,
                              DTYPE2_t s0,
                              DTYPE2_t theta,
                              DTYPE2_t phi,
                              DTYPE2_t cpd):
                  
    # cdef's
    cdef int i,j,k
    cdef DTYPE2_t s_factor2 = (2.0*s0**2)                                 
    cdef DTYPE2_t s_factor3 = (3.0*s0)**2
    cdef int xlim = deg_x.shape[0]
    cdef int ylim = deg_y.shape[1]
    cdef int zlim = stim_arr.shape[-1]
    cdef DTYPE2_t pi_180 = np.pi/180
    cdef DTYPE2_t theta_rad = theta * pi_180
    cdef DTYPE2_t phi_rad = phi * pi_180
    cdef DTYPE2_t pi_2 = np.pi * 2
    cdef DTYPE2_t d
    
    # fodder
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] XYt = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] XYf = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] grating = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] gauss = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] gabor = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(zlim,dtype=DTYPE2)
    
    # the loop
    for i in xrange(xlim):
        for j in xrange(ylim):
            
            # only compute inside the sigma*3
            d = (deg_x[i,j]-x0)**2 + (deg_y[i,j]-y0)**2
            
            if d <= s_factor3:
                
                # creating the grating
                XYt[i,j] =  (deg_x[i,j] * cos(theta_rad)) + (deg_y[i,j] * sin(theta_rad))
                XYf[i,j] = XYt[i,j] * cpd * pi_2
                grating[i,j] = sin(XYf[i,j] + phi_rad)
                
                # create the gaussian
                gauss[i,j] =  exp(-d/s_factor2)
                
                # create the gabor
                gabor[i,j] = gauss[i,j] * grating[i,j]
                
                # for each TR in the stimulus
                for k in xrange(zlim):
                    stim[k] += stim_arr[i,j,k]*gabor[i,j]
                
    return stim
    


@cython.boundscheck(False)
@cython.wraparound(False)
def generate_og_receptive_fields(np.ndarray[DTYPE2_t, ndim=2] deg_x,
                                       np.ndarray[DTYPE2_t, ndim=2] deg_y,
                                       np.ndarray[DTYPE2_t, ndim=1] xs,
                                       np.ndarray[DTYPE2_t, ndim=1] ys,
                                       DTYPE2_t s, DTYPE2_t beta):
    # cdef's
    cdef int i,j,k
    cdef DTYPE2_t s_factor2 = (2.0*s**2)
    cdef DTYPE2_t s_factor3 = (3.0*s)**2
    cdef int xlim = deg_x.shape[0]
    cdef int ylim = deg_x.shape[1]
    cdef int zlim = len(xs)
    
    # initilize the output variable
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] rf = np.zeros((xlim,ylim),dtype=DTYPE2)
    
    for i in xrange(xlim):
        for j in xrange(ylim):
            for k in xrange(zlim):
                d = (deg_x[i,j]-xs[k])**2 + (deg_y[i,j]-ys[k])**2
                if d <= s_factor3:
                    rf[i,j] += exp(-d/s_factor2)*beta

    return rf


@cython.boundscheck(False)
@cython.wraparound(False)
def MakeFastAudioPrediction(np.ndarray[DTYPE2_t, ndim=2] spectrogram,
                            np.ndarray[DTYPE2_t, ndim=2] gaussian,
                            np.ndarray[DTYPE2_t, ndim=2] time_coord,
                            np.ndarray[DTYPE2_t, ndim=2] freq_coord,
                            DTYPE2_t freq_center,
                            DTYPE2_t freq_sigma,
                            DTYPE_t num_timepoints):
    
    # iterators
    cdef int t, f, tr_num, from_slice, to_slice, conv_i, conv_j
    cdef DTYPE2_t conv_sum
    
    # limit our computations to meaningful portions of the gaussian
    s_factor_3 = (3.0*freq_sigma)**2
    
    # loop limiters
    cdef int t_lim = spectrogram.shape[1]
    cdef int f_lim = spectrogram.shape[0]
    cdef int frames_per_tr = t_lim/num_timepoints
    
    # initialize arrays for the loops
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(num_timepoints, dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] g_vector = np.zeros(frames_per_tr, dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] f_vector = np.zeros(frames_per_tr, dtype=DTYPE2)
    
    # loop over each TR
    for tr in np.arange(frames_per_tr, t_lim,t_lim/num_timepoints):
        
        tr_num = tr/(t_lim/num_timepoints)
        from_slice = tr - frames_per_tr/2
        to_slice = tr + frames_per_tr/2
                
        # grab the sound frame an normalize it to 1
        sound_frame = spectrogram[:,from_slice:to_slice]
        
        conv_sum = 0.0
        
        # loop over each frequency and convolve
        for f in range(f_lim):
            
            f_vector = sound_frame[f,:]
            g_vector = gaussian[f,:]
            
            # the hard way
            for conv_i in np.arange(frames_per_tr):
                d = (time_coord[f,conv_i]-time_coord[f,conv_i])**2 + (freq_coord[f,conv_i]-freq_center)**2
                if d <= s_factor_3:
                    for conv_j in np.arange(frames_per_tr):
                        conv_sum += f_vector[conv_i] * g_vector[conv_i-conv_j+1]
            
            # the easy way
            # conv_sum += np.sum(ss.fftconvolve(g_vector,f_vector))
            
        stim[tr_num] = np.mean(conv_sum)
        
    return stim


@cython.boundscheck(False)
@cython.wraparound(False)
def MakeFast_1D_Audio_Prediction(DTYPE_t freq_center, DTYPE_t freq_sigma,
                                 np.ndarray[DTYPE2_t, ndim=2] spectrogram,
                                 np.ndarray[DTYPE2_t, ndim=1] gaussian,
                                 np.ndarray[DTYPE2_t, ndim=1] times,
                                 np.ndarray[DTYPE2_t, ndim=1] freqs,
                                 DTYPE_t num_timepoints):
    
    # typedef counters etc
    cdef int t_ind, f_ind, tr
    cdef int f_lim = spectrogram.shape[0]
    cdef int t_lim = spectrogram.shape[1]
    
    # output stuff
    cdef DTYPE2_t gauss_sum = 0.0
    cdef DTYPE2_t counter = 0.0
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(num_timepoints, dtype=DTYPE2)
    cdef np.ndarray[long,ndim=1,mode='c'] ind
    
    # censoring calculatings
    cdef DTYPE2_t s_factor2, d
    s_factor3 = (3.0*freq_sigma)**2
    
    # main loop
    for tr in xrange(num_timepoints-1):
        
        # empty the tally
        gauss_sum = 0.0
        counter = 0.0
        
        ind = np.nonzero((times <= tr+1) & (times>tr))[0]
        
        # time loop
        for t_ind in xrange(len(ind)):
            
            # freq loop
            for f_ind in xrange(f_lim):
                
                # don't compute points outside 3 sigmas
                d = sqrt((freqs[f_ind] - freq_center)**2)
                if d <= s_factor3:
                    
                    # multiply that by the spectrogram at our time and freq bin
                    gauss_sum += gaussian[f_ind] * spectrogram[f_ind,ind[t_ind]]
                    counter += 1
                
        stim[tr] = gauss_sum
    
    return stim
