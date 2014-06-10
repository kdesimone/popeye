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

from libc.math cimport sin, cos, exp

@cython.boundscheck(False)
@cython.wraparound(False)
def MakeFastGaussPrediction(np.ndarray[DTYPE2_t, ndim=2] degX,
                            np.ndarray[DTYPE2_t, ndim=2] degY,
                            np.ndarray[short, ndim=3] stimArray,
                            DTYPE2_t x, DTYPE2_t y, DTYPE2_t s):

    """
    Predict a time-series of a voxels response, for a specific Gaussian
    pRF. Note that this is the response before convolving with the voxel's HRF 

    Parameters
    ----------
    degX : 2D array
           The X coordinates of points in the stimulus (in degrees)
    degY : 2D array
           The Y coordinates of points in the stimulus (in degrees)
    stimArray : 3D array
    x : float
       The x coordinate of the center of the pRF (in dva)
    y : float
       The y coordinate of the center of the pRF (in dva)
    s : float
       The sigma variable, determining the size of the pRF (in dva).

    Returns
    """
    # cdef's
    cdef int i,j,k # iteration variables
    cdef DTYPE2_t s_factor2 = (2.0*s**2) # precalculate these constants because
                                         # i don't if the compiler would
                                         # optimize this, so i'll just do it
    cdef DTYPE2_t s_factor3 = (3.0*s)**2
    cdef int xlim = degX.shape[0] # for the python-for-range-loop -->
                                  # c-for-loop optimization to kick in,
                                  # range()'s argument must be some size of
                                  # c-integer
    cdef int ylim = degX.shape[1] # it might also optimize xrange, but im not
                                  # sure, so i just stick with range.
    cdef int zlim = stimArray.shape[2]
    cdef DTYPE2_t d, gauss1D # d for distance

    # go

    # now that we are using cython, arithmetic using c-types is fast, so we
    # aren't restricted to using array operations and vectorization for speed
    # then, we don't have to use intermediate lists/arrays, and just apply the
    # result once we find a "good" index (i,j).

    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] tsStim = np.zeros(zlim,
                                                                dtype=DTYPE2)
    # mode='c' (or mode='fortran') is another hint to cython, which guarantees
    # that the array is stored contiguously making indexing just a little
    # faster. i know there are some resources about this...
    cdef DTYPE2_t sum_gauss = 0.0

    for i in xrange(xlim):
        for j in xrange(ylim):
            d = (degX[i,j]-x)**2 + (degY[i,j]-y)**2
            if d <= s_factor3:
                gauss1D = exp(-d/s_factor2)
                sum_gauss += gauss1D
                for k in xrange(zlim):
                    tsStim[k] += stimArray[i,j,k]*gauss1D
                    
    tsStim /= sum_gauss # using the regular numpy broadcasting stuff
    
    return tsStim
    
@cython.boundscheck(False)
@cython.wraparound(False)
def MakeFastRF(np.ndarray[DTYPE2_t, ndim=2] degX,
               np.ndarray[DTYPE2_t, ndim=2] degY,
               DTYPE2_t x, DTYPE2_t y, DTYPE2_t s):
    """
    
    """
    # cdef's
    cdef int i,j # iteration variables
    cdef DTYPE2_t s_factor2 = (2.0*s**2) # precalculate these constants because
                                         # i don't if the compiler would
                                         # optimize this, so i'll just do it
    cdef DTYPE2_t s_factor3 = (3.0*s)**2
    cdef int xlim = degX.shape[0] # for the python-for-range-loop -->
                                  # c-for-loop optimization to kick in,
                                  # range()'s argument must be some size of
                                  # c-integer
    cdef int ylim = degX.shape[1] # it might also optimize xrange, but im not
                                  # sure, so i just stick with range.
    cdef DTYPE2_t d # d for distance

    # now that we are using cython, arithmetic using c-types is fast, so we
    # aren't restricted to using array operations and vectorization for speed
    # then, we don't have to use intermediate lists/arrays, and just apply the
    # result once we find a "good" index (i,j).

    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] rf = np.zeros((xlim,ylim),dtype=DTYPE2)
    # mode='c' (or mode='fortran') is another hint to cython, which guarantees
    # that the array is stored contiguously making indexing just a little
    # faster. i know there are some resources about this...

    for i in xrange(xlim):
        for j in xrange(ylim):
            d = (degX[i,j]-x)**2 + (degY[i,j]-y)**2
            if d <= s_factor3:
                rf[i,j] = exp(-d/s_factor2)

    return rf

@cython.boundscheck(False)
@cython.wraparound(False)
def MakeFastGaussian2D(np.ndarray[DTYPE2_t, ndim=2] X,
                       np.ndarray[DTYPE2_t, ndim=2] Y,
                       DTYPE2_t x0,
                       DTYPE2_t y0,
                       DTYPE2_t sigma_x,
                       DTYPE2_t sigma_y,
                       DTYPE2_t degrees):
                       
    # iterators
    cdef int i,j,k
    cdef DTYPE2_t s_factor2
    cdef DTYPE2_t s_factor3
    
    # convert degress to radians
    cdef DTYPE2_t theta = degrees * np.pi/180
    
    # figure out which sigma is bigger and limit our search to within that
    if sigma_x > sigma_y:
        s_factor2 = (2.0*sigma_x**2)
        s_factor3 = (3.0*sigma_x)**2
    else:
        s_factor2 = (2.0*sigma_y**2)
        s_factor3 = (3.0*sigma_y)**2
    
    # limiters on the the loops
    cdef int xlim = X.shape[0] # for the python-for-range-loop -->
                                  # c-for-loop optimization to kick in,
                                  # range()'s argument must be some size of
                                  # c-integer
    cdef int ylim = Y.shape[1] # it might also optimize xrange, but im not
                                  # sure, so i just stick with range.
    
    # distance from center ...
    cdef DTYPE2_t d
    
    # set up the transformation matrix
    cdef DTYPE2_t a = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
    cdef DTYPE2_t b = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
    cdef DTYPE2_t c = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2
    
    # # setup the output
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] rf = np.zeros((xlim,ylim),dtype=DTYPE2)

    for i in xrange(xlim):
        for j in xrange(ylim):
            d = (X[i,j]-x0)**2 + (Y[i,j]-y0)**2
            if d <= s_factor3:
                rf[i,j] = np.exp( - (a*(X[i,j]-x0)**2 + 2*b*(X[i,j]-x0)*(Y[i,j]-y0) + c*(Y[i,j]-y0)**2))
    
    # return it
    return rf
       
@cython.boundscheck(False)
@cython.wraparound(False)
def MakeFastAudioPrediction(np.ndarray[DTYPE2_t, ndim=2] spectrogram,
                            np.ndarray[DTYPE2_t, ndim=2] gaussian,
                            np.ndarray[DTYPE2_t, ndim=2] time_coord,
                            np.ndarray[DTYPE2_t, ndim=2] freq_coord,
                            DTYPE2_t freq_center,
                            DTYPE2_t freq_sigma,
                            DTYPE2_t hrf_delay,
                            DTYPE_t num_timepoints):
    
    # iterators
    cdef int t, f, tr_num, from_slice, to_slice
    
    # loop limiters
    cdef int t_lim = spectrogram.shape[1]
    cdef int f_lim = spectrogram.shape[0]
    
    cdef int time_bins = t_lim/num_timepoints
    
    # limit our computations to meaningful portions of the gaussian
    s_factor_3 = (3.0*freq_sigma)**2
    
    # initialize arrays for the loops
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(num_timepoints, dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] g_vector = np.zeros(time_bins, dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] f_vector = np.zeros(time_bins, dtype=DTYPE2)
    
    # convolution variables
    cdef int conv_i, conv_j, conv_window
    cdef DTYPE2_t conv_sum
    conv_window = time_bins + time_bins - 1
    
    for t in xrange(num_timepoints):
        
        # grab the frame from spectrogram that we'll analyze at this time-step
        from_slice = t * time_bins
        to_slice = t * time_bins + time_bins
        
        # grab the sound frame an normalize it to 1
        sound_frame = spectrogram[:,from_slice:to_slice]
        
        # initialize the convolution tally
        conv_sum = 0.0
        
        # loop over each frequency and convolve
        for f in range(spectrogram.shape[0]):
            
            f_vector = sound_frame[f,:]
            g_vector = gaussian[f,:]
            
            # something is wrong here?????
            for conv_i in xrange(1,time_bins):
                d = (time_coord[f,conv_i]-time_coord[f,conv_i])**2 + (freq_coord[f,conv_i]-freq_center)**2
                if d <= s_factor_3:
                    for conv_j in xrange(1,time_bins):
                        conv_sum += f_vector[conv_i] * g_vector[conv_i-conv_j+1]
                        
        stim[t] = np.mean(conv_sum)
        
    return stim

@cython.boundscheck(False)
@cython.wraparound(False)
def MakeFastRFs(np.ndarray[DTYPE2_t, ndim=2] degX,
                np.ndarray[DTYPE2_t, ndim=2] degY,
                np.ndarray[DTYPE2_t, ndim=1] xs,
                np.ndarray[DTYPE2_t, ndim=1] ys,
                DTYPE2_t s):
    # cdef's
    cdef int i,j,k # iteration variables
    cdef DTYPE2_t s_factor2 = (2.0*s**2) # precalculate these constants
                                         # because i don't if the compiler
                                         # would optimize this, so i'll just do
                                         # it
    cdef DTYPE2_t s_factor3 = (3.0*s)**2
    cdef int xlim = degX.shape[0] # for the python-for-range-loop -->
                                  # c-for-loop optimization to kick in,
                                  # range()'s argument must be some size of
                                  # c-integer
    cdef int ylim = degX.shape[1] # it might also optimize xrange, but im not
                                  # sure, so i just stick with range.
    cdef int zlim = len(xs)
    cdef DTYPE2_t d # d for distance

    # now that we are using cython, arithmetic using c-types is fast, so we
    # aren't restricted to using array operations and vectorization for speed
    # then, we don't have to use intermediate lists/arrays, and just apply the
    # result once we find a "good" index (i,j).

    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] rf = np.zeros((xlim,ylim),dtype=DTYPE2)
    # mode='c' (or mode='fortran') is another hint to cython, which guarantees
    # that the array is stored contiguously making indexing just a little
    # faster. i know there are some resources about this...

    for i in xrange(xlim):
        for j in xrange(ylim):
            for k in xrange(zlim):
                d = (degX[i,j]-xs[k])**2 + (degY[i,j]-ys[k])**2
                if d <= s_factor3:
                    rf[i,j] += exp(-d/s_factor2)

    return rf

@cython.boundscheck(False)
@cython.wraparound(False)
def MakeFastGabor(np.ndarray[DTYPE2_t, ndim=2] X,
                  np.ndarray[DTYPE2_t, ndim=2] Y,
                  DTYPE2_t x0,
                  DTYPE2_t y0,
                  DTYPE2_t s0,
                  DTYPE2_t theta,
                  DTYPE2_t phi,
                  DTYPE2_t cpd):
                  
    # cdef's
    cdef int i,j,k # iteration variables
    cdef DTYPE2_t s_factor2 = (2.0*s0**2) # precalculate these constants because
                                         # i don't if the compiler would
                                         # optimize this, so i'll just do it
    cdef DTYPE2_t s_factor3 = (3.0*s0)**2

    cdef int xlim = X.shape[0] # for the python-for-range-loop -->
                                  # c-for-loop optimization to kick in,
                                  # range()'s argument must be some size of
                                  # c-integer
    cdef int ylim = Y.shape[1] # it might also optimize xrange, but im not
                                  # sure, so i just stick with range.


    cdef DTYPE2_t pi_180 = np.pi/180
    cdef DTYPE2_t theta_rad = theta * pi_180
    cdef DTYPE2_t phi_rad = phi * pi_180
    cdef DTYPE2_t pi_2 = np.pi * 2

    cdef DTYPE2_t d # d for distance
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] XYt = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] XYf = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] grating = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] gauss = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] gabor = np.zeros((xlim,ylim),dtype=DTYPE2)

    for i in xrange(xlim):
        for j in xrange(ylim):

            # only compute inside the sigma*3
            d = (X[i,j]-x0)**2 + (Y[i,j]-y0)**2
            if d <= s_factor3:

                # creating the grating
                XYt[i,j] =  (X[i,j] * cos(theta_rad)) + (Y[i,j] * sin(theta_rad))
                XYf[i,j] = XYt[i,j] * cpd * pi_2
                grating[i,j] = sin(XYf[i,j] + phi_rad)

                # create the gaussian
                gauss[i,j] =  exp(-d/s_factor2)

                # create the gabor
                gabor[i,j] = gauss[i,j] * grating[i,j]

    return gabor


@cython.boundscheck(False)
@cython.wraparound(False)
def MakeFastGaborPrediction(np.ndarray[DTYPE2_t, ndim=2] X,
                            np.ndarray[DTYPE2_t, ndim=2] Y,
                            np.ndarray[short, ndim=3] stim_arr,
                            DTYPE2_t x0,
                            DTYPE2_t y0,
                            DTYPE2_t s0,
                            DTYPE2_t theta,
                            DTYPE2_t phi,
                            DTYPE2_t cpd):
                            
    # cdef's
    cdef int i,j,k # iteration variables
    cdef DTYPE2_t s_factor2 = (2.0*s0**2) # precalculate these constants because
                                         # i don't if the compiler would
                                         # optimize this, so i'll just do it
    cdef DTYPE2_t s_factor3 = (3.0*s0)**2
    
    cdef int xlim = stim_arr.shape[0]
    cdef int ylim = stim_arr.shape[1]
    cdef int zlim = stim_arr.shape[2]
    
    cdef DTYPE2_t pi_180 = np.pi/180
    cdef DTYPE2_t theta_rad = theta * pi_180
    cdef DTYPE2_t phi_rad = phi * pi_180
    cdef DTYPE2_t pi_2 = np.pi * 2
    
    cdef DTYPE2_t d # d for distance
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] XYt = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] XYf = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] grating = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] gauss = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t, ndim=2, mode='c'] gabor = np.zeros((xlim,ylim),dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] ts_stim = np.zeros(zlim,dtype=DTYPE2)
    
    for i in xrange(xlim):
        for j in xrange(ylim):
            
            # only compute inside the sigma*3
            d = (X[i,j]-x0)**2 + (Y[i,j]-y0)**2
            if d <= s_factor3:
                
                # creating the grating
                XYt[i,j] =  (X[i,j] * cos(theta_rad)) + (Y[i,j] * sin(theta_rad))
                XYf[i,j] = XYt[i,j] * cpd * pi_2
                grating[i,j] = sin(XYf[i,j] + phi_rad)
                
                # create the gaussian
                gauss[i,j] =  exp(-d/s_factor2)
                
                # create the gabor
                gabor[i,j] = gauss[i,j] * grating[i,j]
                
                # for each TR in the stimulus
                for k in xrange(zlim):
                    ts_stim[k] += stim_arr[i,j,k]*gabor[i,j]
                        
                
    return ts_stim