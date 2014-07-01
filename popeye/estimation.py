from __future__ import division
import time
import shutil

import numpy as np
from scipy.special import gamma
from scipy.optimize import fmin_powell, fmin, brute
from scipy.stats import linregress

from popeye.spinach import MakeFastGaussPrediction
import popeye.utilities as utils

def compute_prf_estimate(deg_x_coarse, deg_y_coarse, deg_x_fine, deg_y_fine,
                         stim_arr_coarse, stim_arr_fine, funcData, 
                         core_voxels, results_q, bounds=(), uncorrected_rval=0, 
                         norm_func=utils.zscore, verbose=True):
    """ 
    The main pRF estimation method using a single Gaussian pRF model (Dumoulin
    & Wandell, 2008). 
    
    The function takes voxel coordinates, `voxels`, and uses a combination of
    the adaptive brute force grid-search and a gradient descent error
    minimization to compute the pRF estimate and HRF delay.  An initial guess
    of the pRF estimate is computed via the adaptive brute force grid-search is
    computed and handed to scipy.opitmize.fmin_powell for fine tuning.  Once
    the error minimization routine converges on a solution, fit statistics are
    computed via scipy.stats.linregress.  The output are stored in the
    multiprocessing.Queue object `results_q` and returned once all the voxels
    specified in `voxels` have been estimated.
    
    Parameters
    ----------
    deg_x_coarse : XXX
    deg_y_coarse : XXX
    deg_x_fine : XXX
    deg_y_fine : XXX
    stim_arr_coarse : XXX
    stim_arr_fine : XXX
    funcData : ndarray
        A 4D numpy array containing the functional data to be used for the pRF
        estimation. For details, see config.py
    bounds : XXX
    core_voxels : XXX
    uncorrected_rval : XXX 
    results_q : multiprocessing.Queue object
        A multiprocessing.Queue object into which list of pRF estimates and fit
        metrics are stacked. 
    norm_func : callable, optional
        The function used to normalize the time-series data. Can be any
        function that takes an array as input and returns a same-shaped array,
    but consider using `utils.percent_change` or `utils.zscore` (the default).
    verbose : bool, optional
        Toggle the progress printing to the terminal.

    Returns
    -------
    results_q : multiprocessing.Queue
        The pRF four parameter estimate for each voxel in addition to
        statistics for evaluating the goodness of fit. 
    
    Reference
    ----------
    Glover, G.H. (1999) Deconvolution of impulse response in event-related BOLD.
    fMRI. NeuroImage 9: 416 429.

    Dumoulin S.O and Wandell B.A. (2008). Population receptive field estimates
    in human visual cortex. Neuroimage 39: 647-660.
    
    """
    # grab voxel indices
    xi,yi,zi = core_voxels
    
    # initialize a list in which to store the results
    results = []
    
    # printing niceties
    numVoxels = len(xi)
    voxelCount = 1
    printLength = len(xi)/10
    
    # main loop
    for xvoxel, yvoxel, zvoxel in zip(xi, yi, zi):
        
        # time each voxel's estimation time
        tic = time.clock()

        # Grab the 1-D timeseries for this voxel
        ts_actual = funcData[xvoxel, yvoxel, zvoxel,:]
        ts_actual = norm_func(ts_actual)
        
        # make sure we're only analyzing valid voxels
        if not np.isnan(np.sum(ts_actual)):
            x, y, s, d, err, stats = voxel_prf(ts_actual,
                                               deg_x_coarse,
                                               deg_y_coarse,
                                               deg_x_fine,
                                               deg_y_fine,
                                               stim_arr_coarse,
                                               stim_arr_fine,
                                               bounds=bounds,
                                               uncorrected_rval=0,
                                               norm_func=norm_func)
    
            # close the processing time
            toc = time.clock()
                                    
            if verbose:
                  percentDone = (voxelCount / numVoxels) * 100
                  # print the details of the estimation for this voxel
                  report_str = "%.01f%% DONE "%percentDone
                  report_str += "VOXEL=(%.03d,%.03d,%.03d) "%(xvoxel,
                                                              yvoxel,
                                                              zvoxel)
                  report_str += "TIME=%.03f "%toc-tic
                  report_str += "ERROR=%.03d "%err
                  report_str += "RVAL=%.02f"%stats[2]
                  print(report_str)
                  # store the results
            results.append((xvoxel, yvoxel, zvoxel, x, y, s, d, stats))
                
            # interate variable
            voxelCount += 1

    # add results to the queue
    results_q.put(results)
    return results_q


def voxel_prf(ts_vox, deg_x_coarse, deg_y_coarse,
              deg_x_fine, deg_y_fine, stim_arr_coarse,
              stim_arr_fine, bounds=(), uncorrected_rval=0,
              norm_func=utils.zscore):
      """
      Compute the pRF parameters for a single voxel.
      
      Start with a brute force grid search, at coarse resolution and follow up
      with a gradient descent at fine resolution.
      
      Parameters
      ----------
      ts_vox : 1D array
         The normalized time-series of a
      deg_x_coarse, deg_y_coarse :
      deg_x_fine, deg_y_fine :
      stim_arr_coarse :
      stim_arr_fine :
      uncorrected_rval : float, optional
      norm_func: callable, optional
      
      Returns
      -------
      The pRF parameters for this voxel
      x : 
      y : 
      sigma : 
      hrf_delay: 
      stats: 
      
      """
      
      # compute the initial guess with the adaptive brute-force grid-search
      x0, y0, s0, hrf0 = adaptive_brute_force_grid_search(bounds,
                                                          1,
                                                          3,
                                                          ts_vox,
                                                          deg_x_coarse,
                                                          deg_y_coarse,
                                                          stim_arr_coarse)
                                                          
      # regenerate the best-fit for computing the threshold
      ts_stim = MakeFastGaussPrediction(deg_x_coarse,
                                   deg_y_coarse,
                                   stim_arr_coarse,
                                   x0,
                                   y0,
                                   s0)
                                   
      # convolve with HRF and z-score
      hrf = double_gamma_hrf(hrf0)
      ts_model = np.convolve(ts_stim, hrf)
      ts_model = ts_model[0:len(ts_vox)]
      ts_model = norm_func(ts_model)
      
      # compute the p-value to be used for thresholding
      stats0 = linregress(ts_vox, ts_model)
      
      # only continue if the brute-force grid-search came close to a
      # solution 
      if stats0[2] > uncorrected_rval:
          # gradient-descent the solution using the x0 from the
          [x, y, sigma, hrf_delay], err,  _, _, _, warnflag =\
              fmin_powell(error_function,
                          (x0, y0, s0, hrf0),
                          args=(ts_vox,
                                deg_x_fine,
                                deg_y_fine,
                                stim_arr_fine),
                                full_output=True,
                                disp=False)
                                
          # ensure that the fmin finished OK:
          if (warnflag == 0 and not np.any(np.isnan([x, y, sigma, hrf_delay]))
             and not np.isinf(err)):
             
              # regenerate the best-fit for computing the threshold
              ts_stim = MakeFastGaussPrediction(deg_x_fine,
                                          deg_y_fine,
                                          stim_arr_fine,
                                          x,
                                          y,
                                          sigma)
                                          
              # convolve with HRF and z-score
              hrf = double_gamma_hrf(hrf_delay)
              ts_model= np.convolve(ts_stim, hrf)
              ts_model= ts_model[0:len(ts_vox)]
              ts_model = norm_func(ts_model)
              
              # compute the final stats:
              stats = linregress(ts_vox, ts_model)
              
              return x, y, sigma, hrf_delay, err, stats, ts_model
          else:
              print("The fmin did not finish properly!")
              return None
      else:
         print("The brute-force did not pass threshold, %.02f < %.02f!" %(stats0[2],uncorrected_rval))
         return None
