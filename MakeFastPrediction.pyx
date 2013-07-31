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
                       np.ndarray[short,ndim=3] stimArray,
                       DTYPE2_t x, DTYPE2_t y, DTYPE2_t s):
                       # i just like splitting up the function args onto different lines
                       
    # cdef's
    cdef int i,j,k # iteration variables
    cdef DTYPE2_t s_factor2 = (2.0*s**2) # precalculate these constants becauaes i don't if the compiler would optimize this, so i'll just do it
    cdef DTYPE2_t s_factor3 = (3.0*s)**2
    cdef int xlim = degX.shape[0] # for the python-for-range-loop --> c-for-loop optimization to kick in, range()'s argument must be some size of c-integer
    cdef int ylim = degX.shape[1] # it might also optimize xrange, but im not sure, so i just stick with range.
    cdef int zlim = stimArray.shape[2]
    cdef DTYPE2_t d, gauss1D # d for distance
    
    # go
    
    # now that we are using cython, arithmetic using c-types is fast,
    # so we aren't restricted to using array operations and vectorization for speed
    # then, we don't have to use intermediate lists/arrays,
    # and just apply the result once we find a "good" index (i,j).
    
    cdef np.ndarray[DTYPE2_t,ndim=1,mode='c'] tsStim = np.zeros(zlim,dtype=DTYPE2)
    # mode='c' (or mode='fortran') is another hint to cython, which guarantees that the array is stored contiguously
    # making indexing just a little faster. i know there are some resources about this...
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