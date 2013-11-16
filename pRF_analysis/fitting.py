from __future__ import division
import time
import shutil
import numpy as np
from scipy.special import gamma
from scipy.optimize import fmin_powell
from scipy.stats import linregress
from MakeFastPrediction import MakeFastPrediction

def double_gamma_hrf(delay):
    """
    The double-gamma hemodynamic reponse function (HRF) used to convolve with the stimulus time-series.

    The user specifies only the delay of the peak and under-shoot The delay shifts the peak and
    under-shoot by a variable number of seconds.  The other parameters are hard-coded.  The HRF delay
    is modeled for each voxel independently.  The double-gamme HRF andhard-coded values are based
    on previous work (Glover, 1999).


    Parameters
    ----------
    delay : int
        The delay of the HRF peak and under-shoot.


    Returns
    -------
    hrf : ndarray
        The hemodynamic response function to convolve with the stimulus time-series.


    Reference
    ----------
    Glover, G.H. (1999) Deconvolution of impulse response in event-related BOLD.
    fMRI. NeuroImage 9: 416 429.

    """

    # add delay to the peak and undershoot params (alpha 1 and 2)
    alpha_1 = 5.0+delay
    beta_1 = 1.0
    c = 0.2
    alpha_2 = 15.0+delay
    beta_2 = 1.0

    t = np.arange(0,33)
    scale = 1
    hrf = scale*( ( ( t ** (alpha_1 - 1 ) * beta_1 ** alpha_1 * np.exp( -beta_1 * t )) /gamma( alpha_1 )) - c *
            ( ( t ** (alpha_2 - 1 ) * beta_2 ** alpha_2 * np.exp( -beta_2 * t )) /gamma( alpha_2 )))

    return hrf

def error_function(modelParams,tsActual,degX,degY,stimArray):
    """
    The objective function that yields a minimizeable error between the predicted and actual
    BOLD time-series.

    The objective function takes candidate pRF estimates `modelParams`, including a parameter for
    the screen location (x,y) and dispersion (sigma) of the 2D Gaussian as well as the HRF delay (tau).
    The objective function also takes the ancillary parameters `degX` and `degY`, visual coordinate
    arrays, as well as the digitized stimulus array `stimArray`.  The predicted time-series is generated
    by multiplying the pRF by the stimulus array and convolving the result with the double-gamma HRF.  The
    predicted time-series is mean centered and variance normalized.  The residual sum of squared errors
    is computed between the predicted and actual time-series and returned.

    This function makes use of the Cython optimising static compiler.  MakeFastPrediction is written in Cython
    and computes the pre HRF convolved model time-series.

    Parameters
    ----------
    modelParams : ndarray, dtype=single/double
        A quadruplet model parameters including the pRF estimate (x,y,sigma) and the HRF delay (tau).
    tsActual : ndarray
        A vector of the actual BOLD time-series extacted from a single voxel coordinate.
    degX : ndarray
        An array representing the horizontal extent of the visual display in terms of degrees of
        visual angle.
    degY : ndarray
        An array representing the vertical extent of the visual display in terms of degrees of
        visual angle.
    stimArray : ndarray
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.

    Returns
    -------
    error : float
        The residual sum of squared errors computed between the predicted and actual time-series.


    """

    import numpy as np
    from MakeFastPrediction import MakeFastPrediction

    # if the x or y are off the screen, abort with an inf
    if np.abs(modelParams[0]) > np.max(degY):
        return np.inf
    if np.abs(modelParams[1]) > np.max(degY):
        return np.inf

    # if the sigma is larger than the screen width, abort with an inf
    if np.abs(modelParams[2]) > np.max(degY):
        return np.inf

    # if the sigma is <= 0, abort with an inf
    if modelParams[2] <= 0:
        return np.inf

    # otherwise generate a prediction
    tsStim = MakeFastPrediction(degX,
                                degY,
                                stimArray,
                                modelParams[0],
                                modelParams[1],
                                modelParams[2])

    # compute the double-gamma HRF at the specified delay
    hrf = double_gamma_hrf(modelParams[3])

    # convolve the stimulus time-series with the HRF
    tsModel = np.convolve(tsStim,hrf)
    tsModel = tsModel[0:len(tsActual)]

    # z-score the model time-series
    tsModel -= np.mean(tsModel)
    tsModel /= np.std(tsModel)

    # compute the RSS
    error = np.sum((tsModel-tsActual)**2)

    # catch NaN
    if np.isnan(np.sum(error)):
        return np.inf

    return error

def adaptive_brute_force_grid_search(Bounds,epsilon,rounds,tsActual,degX,degY,stimArray):
    """

    An adaptive brute force grid-search to generate a ball-park pRF estimate for fine tuning
    via a gradient descent error minization.

    The adaptive brute-force grid-search sparsely samples the parameter space and uses a down-sampled
    version of the stimulus and cooridnate matrices.  This is intended to yield an initial, ball-park
    solution that is then fed into the more finely-tuned fmin_powell in the compute_prf_estimate method
    below.

    """

    from scipy.optimize import brute, fmin, fmin_powell

    # set initial pass to 1
    passNum = 1

    # make as many passes as the user specifies in rounds
    while passNum <= rounds:

        # get a fit estimate by sparsely sampling the 4-parameter space
        phat = brute(error_function,
                 args=(tsActual,degX,degY,stimArray),
                 ranges=Bounds,
                 Ns=5,
                 finish=fmin_powell)

        # recompute the grid-search bounds by halving the sampling space
        epsilon /= 2.0
        Bounds = ((phat[0]-epsilon,phat[0]+epsilon),
                  (phat[1]-epsilon,phat[1]+epsilon),
                  (phat[2]-epsilon,phat[2]+epsilon),
                  (phat[3]-epsilon,phat[3]+epsilon))

        # iterate the pass variable
        passNum += 1

    return phat

def compute_prf_estimate(voxels,stimData,funcData,results_q,verbose=True):
    """

    The main pRF estimation method using a single Gaussian pRF model (Dumoulin & Wandell, 2008).

    The function takes voxel coordinates, `voxels`, and uses a combination of the
    adaptive brute force grid-search and a gradient descent error minimization to compute
    the pRF estimate and HRF delay.  An initial guess of the pRF estimate is computed via
    the adaptive brute force grid-search is computed and handed to scipy.opitmize.fmin_powell
    for fine tuning.  Once the error minimization routine converges on a solution, fit statistics
    are computed via scipy.stats.linregress.  The output are stored in the multiprocessing.Queue
    object `results_q` and returned once all the voxels specified in `voxels` have been
    estimated.

    Parameters
    ----------
    voxels : ndarray
        The volumetric coordinates of the voxels to be estimated.
    stimData : dict
        A dictionary containing the stimulus array and other stimulus-related data.  For details, see config.py
    funcData : ndarray
        A 4D numpy array containing the functional data to be used for the pRF estimation. For details, see config.py
    results_q : multiprocessing.Queue object
        A multiprocessing.Queue object into which list of pRF estimates and fit metrics are stacked.
    verbose : bool, optional
        Toggle the progress printing to the terminal.

    Returns
    -------
    results : list
        The hemodynamic response function to convolve with the stimulus time-series.


    Reference
    ----------
    Glover, G.H. (1999) Deconvolution of impulse response in event-related BOLD.
    fMRI. NeuroImage 9: 416 429.

    """

    # bounds for the adaptive brute-force grid-search
    Bounds = ((-10, 10), (-10, 10), (0.25, 5.25),(-1,1))

    # grab voxel indices
    xi,yi,zi = voxels[:]

    # initialize a list in which to store the results
    results = []

    # printing niceties
    numVoxels = len(xi)
    voxelCount = 1
    printLength = len(xi)/10

    # main loop
    for xvoxel,yvoxel,zvoxel in zip(xi,yi,zi):

        # time each voxel's estimation time
        tic = time.clock()

        # z-score the functional data and clip off the intial blank period
        tsActual = funcData[xvoxel,yvoxel,zvoxel,:]
        tsActual /= np.std(tsActual)

        # make sure we're only analyzing valid voxels
        if not np.isnan(np.sum(tsActual)):

            # compute the initial guess with the adaptive brute-force grid-search
            x0 = adaptive_brute_force_grid_search(Bounds,
                                  1,
                                  3,
                                  tsActual,
                                  stimData['degXCoarse'],
                                  stimData['degYCoarse'],
                                  stimData['stimArrayCoarse'])

            # gradient-descent the solution using the x0 from the brute-force grid-search
            pRF_phat = fmin_powell(error_function,
                                   x0,
                                   args=(tsActual,stimData['degXFine'],stimData['degYFine'],stimData['stimArrayFine']),
                                   full_output=True,
                                   disp=False)

            # ensure that the fmin finished OK
            if pRF_phat[-1] == 0 and not np.isnan(pRF_phat[1]) and not np.isinf(pRF_phat[1]):

                # regenerate the best-fit for computing the threshold
                tsStim = MakeFastPrediction(stimData['degXFine'],
                                            stimData['degYFine'],
                                            stimData['stimArrayFine'],
                                            pRF_phat[0][0],
                                            pRF_phat[0][1],
                                            pRF_phat[0][2])

                # convolve with HRF and z-score
                hrf = double_gamma_hrf(pRF_phat[0][3])
                tsModel = np.convolve(tsStim,hrf)
                tsModel = tsModel[0:len(tsActual)]
                tsModel -= np.mean(tsModel)
                tsModel /= np.std(tsModel)

                # compute the p-value to be used for thresholding
                stats = linregress(tsActual,tsModel)

                # close the processing time
                toc = time.clock()

                # assign the fit variables and print
                x = pRF_phat[0][0]
                y = pRF_phat[0][1]
                s = pRF_phat[0][2]
                d = pRF_phat[0][3]
                err = pRF_phat[1]

                if verbose:
                    percentDone = (voxelCount/numVoxels)*100
                    # print the details of the estimation for this voxel
                    print("%.01f%% DONE  VOXEL=(%.03d,%d,%d)  TIME=%.03f  X=%.03f  Y=%.03f  S=%.03f  D=%.03f  ERROR=%d  COV=%.03f"
                          %(percentDone,
                            xvoxel,
                            yvoxel,
                            zvoxel,
                            toc-tic,
                            x,
                            y,
                            s,
                            d,
                            err,
                            stats[2]**2))

                # store the results
                results.append((xvoxel,yvoxel,zvoxel,x,y,s,d,stats))

                # interate variable
                voxelCount += 1

    # add results to the queue
    results_q.put(results)

    return results_q