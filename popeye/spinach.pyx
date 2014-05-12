cimport cython
from cython.parallel import prange, parallel, threadid
import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

DTYPE2 = np.double
ctypedef np.double_t DTYPE2_t

cdef extern from "math.h": # using a function from c's math.h
    #double exp(double)
    DTYPE2_t exp(DTYPE2_t)
    # i don't know how to use ctypedef properly,
    # so i always just use c-types, like double
    
@cython.boundscheck(False)
@cython.wraparound(False)
def MakeFastPrediction(np.ndarray[DTYPE2_t, ndim=2] degX,
                       np.ndarray[DTYPE2_t, ndim=2] degY,
                       np.ndarray[DTYPE2_t, ndim=3] stimArray,
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

    # cdef's
    
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

    
