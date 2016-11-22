cimport cython
from cython.parallel import prange, parallel, threadid
import numpy as np
cimport numpy as np
from scipy.signal.sigtools import _correlateND
import ctypes

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

DTYPE2 = np.double
ctypedef np.double_t DTYPE2_t

DTYPE3 = np.short
ctypedef short DTYPE3_t

from libc.math cimport sin, cos, exp, sqrt

try:
    xrange
except NameError:  # python3
    xrange = range


@cython.boundscheck(False)
@cython.wraparound(False)
def binner(np.ndarray[DTYPE2_t, ndim=1] signal,
           np.ndarray[DTYPE2_t, ndim=1] times,
           np.ndarray[DTYPE2_t, ndim=1] bins):

    # type defs
    cdef int t, t_lim
    cdef DTYPE2_t bin_width

    # vars
    t_lim = len(bins)
    bin_width = bins[1] - bins[0]

    # output
    cdef np.ndarray[DTYPE2_t, ndim=1, mode='c'] binned_response = np.zeros((t_lim)-2)

    for t in xrange(1,t_lim):
        the_bin = bins[t]
        binned_signal = signal[(times >= the_bin-bin_width) & (times <= the_bin)]
        binned_response[t-2] = np.sum(binned_signal)

    return binned_response

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
def generate_mp_timeseries(np.ndarray[DTYPE2_t, ndim=1] spatial_ts,
                           np.ndarray[DTYPE2_t, ndim=1] m_amp,
                           np.ndarray[DTYPE2_t, ndim=1] p_amp,
                           np.ndarray[DTYPE_t, ndim=1] flicker_vec):
    # cdef's
    cdef int t
    cdef int tlim = spatial_ts.shape[0]

    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] m_ts = np.zeros(tlim,dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] p_ts = np.zeros(tlim,dtype=DTYPE2)

    # the loop
    for t in xrange(tlim):
        amp = spatial_ts[t]
        flicker = flicker_vec[t]
        if flicker == 1:
            m_ts[t] = m_amp[0] * amp
            p_ts[t] = p_amp[0] * amp
        if flicker == 2:
            m_ts[t] = m_amp[1] * amp
            p_ts[t] = p_amp[1] * amp

    return m_ts, p_ts

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_strf_timeseries(np.ndarray[DTYPE2_t, ndim=1] stim_ts,
                             np.ndarray[DTYPE2_t, ndim=2] m_resp,
                             np.ndarray[DTYPE2_t, ndim=2] p_resp,
                             np.ndarray[DTYPE_t, ndim=1] flicker_vec):
    # cdef's
    cdef int s,t
    cdef int slim = stim_ts.shape[0]
    cdef int tlim = m_resp.shape[0]

    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] m_ts = np.zeros(slim,dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] p_ts = np.zeros(slim,dtype=DTYPE2)

    # the loop
    for s in xrange(slim):
        amp = stim_ts[s]
        for t in xrange(tlim):
                m_ts[s] += m_resp[t,flicker_vec[s]-1] * amp
                p_ts[s] += p_resp[t,flicker_vec[s]-1] * amp

    return m_ts, p_ts

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_strf_betas_timeseries(np.ndarray[DTYPE2_t, ndim=1] stim_ts,
                                    np.ndarray[DTYPE2_t, ndim=2] m_resp,
                                    np.ndarray[DTYPE2_t, ndim=2] p_resp,
                                    np.ndarray[DTYPE_t, ndim=1] flicker_vec,
                                    DTYPE2_t m_beta, DTYPE2_t p_beta):
    # cdef's
    cdef int s,t
    cdef int slim = stim_ts.shape[0]
    cdef int tlim = m_resp.shape[0]

    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(slim,dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] buff = np.zeros(slim,dtype=DTYPE2)

    # the loop
    for s in xrange(slim):
        amp = stim_ts[s]
        for t in xrange(tlim):
                stim[s] += m_resp[t,flicker_vec[s]-1] * m_beta * amp + p_resp[t,flicker_vec[s]-1] * p_beta * amp

    return stim

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_strf_beta_ratio_timeseries(np.ndarray[DTYPE2_t, ndim=1] stim_ts,
                                        np.ndarray[DTYPE2_t, ndim=2] m_resp,
                                        np.ndarray[DTYPE2_t, ndim=2] p_resp,
                                        np.ndarray[DTYPE_t, ndim=1] flicker_vec,
                                        DTYPE2_t beta, DTYPE2_t beta_ratio):
    # cdef's
    cdef int s,t
    cdef int slim = stim_ts.shape[0]
    cdef int tlim = m_resp.shape[0]
    cdef DTYPE2_t beta_2 = beta_ratio/beta

    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(slim,dtype=DTYPE2)

    # the loop
    for s in xrange(slim):
        amp = stim_ts[s]
        for t in xrange(tlim):
                stim[s] += m_resp[t,flicker_vec[s]-1] * beta * amp + p_resp[t,flicker_vec[s]-1] * beta_2 * amp

    return stim

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_strf_weight_timeseries(np.ndarray[DTYPE2_t, ndim=1] stim_ts,
                                    np.ndarray[DTYPE2_t, ndim=2] m_resp,
                                    np.ndarray[DTYPE2_t, ndim=2] p_resp,
                                    np.ndarray[DTYPE_t, ndim=1] flicker_vec,
                                    DTYPE2_t weight):
    # cdef's
    cdef int s,t
    cdef int slim = stim_ts.shape[0]
    cdef int tlim = m_resp.shape[0]

    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(slim,dtype=DTYPE2)
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] buff = np.zeros(slim,dtype=DTYPE2)

    # the loop
    for s in xrange(slim):
        amp = stim_ts[s]
        for t in xrange(tlim):
                stim[s] += m_resp[t,flicker_vec[s]-1] * (1-weight) * amp + p_resp[t,flicker_vec[s]-1] * weight * amp

    return stim

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_rf_timeseries_nomask(np.ndarray[DTYPE3_t, ndim=3] stim_arr,
                                  np.ndarray[DTYPE2_t, ndim=2] rf):

    # cdef's
    cdef int i,j,k
    cdef int xlim = stim_arr.shape[0]
    cdef int ylim = stim_arr.shape[1]
    cdef int zlim = stim_arr.shape[2]

    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(zlim,dtype=DTYPE2)

    # the loop
    for i in xrange(xlim):
        for j in xrange(ylim):
            for k in xrange(zlim):
                stim[k] += stim_arr[i,j,k]*rf[i,j]

    return stim

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_rf_timeseries(np.ndarray[DTYPE3_t, ndim=3] stim_arr,
                           np.ndarray[DTYPE2_t, ndim=2] rf,
                           np.ndarray[DTYPE_t, ndim=2] mask):

    # cdef's
    cdef int i,j,k
    cdef int xlim = stim_arr.shape[0]
    cdef int ylim = stim_arr.shape[1]
    cdef int zlim = stim_arr.shape[2]

    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(zlim,dtype=DTYPE2)

    # the loop
    for i in xrange(xlim):
        for j in xrange(ylim):
            if mask[i,j] == 1:
                for k in xrange(zlim):
                    stim[k] += stim_arr[i,j,k]*rf[i,j]

    return stim

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_rf_timeseries_1D(np.ndarray[DTYPE2_t, ndim=2] stim_arr,
                              np.ndarray[DTYPE2_t, ndim=1] rf,
                              np.ndarray[DTYPE_t, ndim=1] mask):

    # cdef's
    cdef int i,j,k
    cdef int xlim = stim_arr.shape[0]
    cdef int ylim = stim_arr.shape[1]

    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(ylim,dtype=DTYPE2)

    # the loop
    for i in xrange(xlim):
        if mask[i] == 1:
            for j in xrange(ylim):
                stim[j] += stim_arr[i,j]*rf[i]

    return stim

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_dog_timeseries(np.ndarray[DTYPE2_t, ndim=2] deg_x,
                                 np.ndarray[DTYPE2_t, ndim=2] deg_y,
                                 np.ndarray[DTYPE_t, ndim=3] stim_arr,
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
                           np.ndarray[DTYPE_t, ndim=3] stim_arr,
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
    cdef DTYPE2_t sum_gauss = 0.0

    # initialize output variable
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] stim = np.zeros(zlim,dtype=DTYPE2)

    # the loop
    for i in xrange(xlim):
        for j in xrange(ylim):
            d = (deg_x[i,j]-x)**2 + (deg_y[i,j]-y)**2
            if d <= gauss_distance:
                gauss1D = exp(-d/gauss_area)
                sum_gauss += gauss1D
                for k in xrange(zlim):
                    stim[k] += stim_arr[i,j,k]*gauss1D

    return stim

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_spectrotemporal_timeseries(np.ndarray[DTYPE2_t, ndim=1] freqs,
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
def generate_og_receptive_field(DTYPE2_t x, DTYPE2_t y, DTYPE2_t sigma,
                                np.ndarray[DTYPE2_t, ndim=2] deg_x,
                                np.ndarray[DTYPE2_t, ndim=2] deg_y):
    """
    Generate a Gaussian.

    Parameters
    ----------
    x : float
       The x coordinate of the center of the Gaussian (degrees)
    y : float
       The y coordinate of the center of the Gaussian (degrees)
    s : float
       The dispersion of the Gaussian (degrees)
    beta : float
        The amplitude of the Gaussian
    deg_x : 2D array
            The coordinate matrix along the horizontal dimension of the display (degrees)
    deg_y : 2D array
            The coordinate matrix along the vertical dimension of the display (degrees)


    Returns

    stim : ndarray
        The 1D array containing the stimulus energies given the Gaussian coordinates

    """

    # cdef's
    cdef int i,j,k
    cdef DTYPE2_t s_factor2 = (2.0*sigma**2)
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
            rf[i,j] = exp(-d/s_factor2)

    return rf

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_gabor_receptive_field(DTYPE2_t x0,
                                   DTYPE2_t y0,
                                   DTYPE2_t s0,
                                   DTYPE2_t theta,
                                   DTYPE2_t phi,
                                   DTYPE2_t cpd,
                                   np.ndarray[DTYPE2_t, ndim=2] deg_x,
                                   np.ndarray[DTYPE2_t, ndim=2] deg_y):

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
def generate_gabor_receptive_field(DTYPE2_t x0,
                                   DTYPE2_t y0,
                                   DTYPE2_t s0,
                                   DTYPE2_t theta,
                                   DTYPE2_t phi,
                                   DTYPE2_t cpd,
                                   np.ndarray[DTYPE2_t, ndim=2] deg_x,
                                   np.ndarray[DTYPE2_t, ndim=2] deg_y):

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
def generate_og_receptive_fields(
                                 np.ndarray[DTYPE2_t, ndim=1] xs,
                                 np.ndarray[DTYPE2_t, ndim=1] ys,
                                 np.ndarray[DTYPE2_t, ndim=1] ss,
                                 np.ndarray[DTYPE2_t, ndim=1] amps,
                                 np.ndarray[DTYPE2_t, ndim=2] deg_x,
                                 np.ndarray[DTYPE2_t, ndim=2] deg_y):
    # cdef's
    cdef int i,j,k
    cdef int xlim = deg_x.shape[0]
    cdef int ylim = deg_x.shape[1]
    cdef int zlim = len(xs)
    cdef DTYPE2_t s_factor2 = (2.0*ss[0]**2)

    # initilize the output variable
    cdef np.ndarray[DTYPE2_t, ndim=3, mode='c'] rfs = np.zeros((xlim,ylim,zlim),dtype=DTYPE2)

    for i in xrange(xlim):
        for j in xrange(ylim):
            for k in xrange(zlim):
                s_factor2 = (2.0*ss[k]**2)
                d = (deg_x[i,j]-xs[k])**2 + (deg_y[i,j]-ys[k])**2
                rfs[i,j,k] += exp(-d/s_factor2) * amps[k]

    return rfs
