import os

import numpy as np
import numpy.testing as npt
import nose.tools as nt

import popeye.utilities as utils
import popeye.estimation as pest

def test_double_gamma_hrf():
    """
    Test voxel-wise prf estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # produce an HRF with 0 delay
    hrf0 = pest.double_gamma_hrf(0)
    hrf0 = np.round(np.sum(hrf0)*100,2)
    
    # produce an HRF with +1 delay
    hrf_pos = pest.double_gamma_hrf(1)
    hrf_pos = np.round(np.sum(hrf_pos)*100,2)
    
    # produce an HRF with -1 delay
    hrf_neg = pest.double_gamma_hrf(-1)
    hrf_neg = np.round(np.sum(hrf_neg)*100,2)
    
    # assert 
    nt.assert_almost_equal(hrf0, 80.02, 2)
    nt.assert_almost_equal(hrf_pos, 80.01, 2)
    nt.assert_almost_equal(hrf_neg, 80.11, 2)
    
def test_error_function():
    """
    Test voxel-wise prf estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # set the path to data
    data_path = os.path.join(os.path.expanduser('~'), '.popeye/popeye')
    
    # load the datasets
    stimulus = np.load('%s/sample_stimulus.npy' %(data_path))
    response = np.load('%s/sample_response.npy' %(data_path))
    estimate = np.load('%s/sample_estimate.npy' %(data_path))
    
    # stimulus and display parameters 
    monitor_width = 25.0 # distance across the width of the image on the
                        # projection screen in cm 
    viewing_distance = 38.0 # viewing distance from the subject's eye to the
                           # projection screen in cm 
    pixels_across = 800 # display resolution across in pixels
    pixels_down = 600 # display resolution down in pixels
    ppd = np.pi*pixels_across/np.arctan(monitor_width/viewing_distance/2.0)/360.0 # degrees of visual angle
    clip_number = 10 # TRs to remove at the beginning
    roll_number = -2 # TRs to rotate the time-series.
    fine_scale = 1.0 # Decimal describing how much to down-sample the
                          # stimulus for increased fitting speed 
    coarse_scale = 0.05 # Decimal describing how much to down-sample the
                             # stimulus for increased fitting speed
    
    
    # the non-resampled stimulus array
    stim_arr = stimulus[:,:,clip_number::]
    stim_arr = np.roll(stim_arr,roll_number,axis=-1)
    stim_arr_fine = utils.resample_stimulus(stim_arr,fine_scale)
    deg_x_fine,deg_y_fine = utils.generate_coordinate_matrices(pixels_across,
                                                               pixels_down,
                                                               ppd,
                                                               fine_scale)
    
    # grab a voxel's time-series and z-score it
    ts_voxel = response[0,clip_number::]
    ts_voxel = utils.zscore(ts_voxel)
    
    # compute the error using the results of a known pRF estimation
    test_results = pest.error_function(estimate[0,0:4],
                                                ts_voxel,
                                                deg_x_fine,
                                                deg_y_fine,
                                                stim_arr_fine)
    
    # get the precomputed error
    gold_standard = estimate[0,4]
    
    # assert equal to 3 decimal places
    npt.assert_almost_equal(gold_standard, test_results)

def test_adapative_brute_force_grid_search():
    """
    Test voxel-wise prf estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # set the path to data
    data_path = os.path.join(os.path.expanduser('~'), '.popeye/popeye')
    
    # load the datasets
    stimulus = np.load('%s/sample_stimulus.npy' %(data_path))
    response = np.load('%s/sample_response.npy' %(data_path))
    
    # stimulus and display parameters 
    monitor_width = 25.0 # distance across the width of the image on the
                        # projection screen in cm 
    viewing_distance = 38.0 # viewing distance from the subject's eye to the
                           # projection screen in cm 
    pixels_across = 800 # display resolution across in pixels
    pixels_down = 600 # display resolution down in pixels
    ppd = np.pi*pixels_across/np.arctan(monitor_width/viewing_distance/2.0)/360.0 # degrees of visual angle
    clip_number = 10 # TRs to remove at the beginning
    roll_number = -2 # TRs to rotate the time-series.
    fine_scale = 1.0 # Decimal describing how much to down-sample the
                          # stimulus for increased fitting speed 
    coarse_scale = 0.05 # Decimal describing how much to down-sample the
                             # stimulus for increased fitting speed
    
    
    # the non-resampled stimulus array
    stim_arr = stimulus[:,:,clip_number::]
    stim_arr = np.roll(stim_arr,roll_number,axis=-1)
    stim_arr_fine = utils.resample_stimulus(stim_arr,fine_scale)
    deg_x_fine,deg_y_fine = utils.generate_coordinate_matrices(pixels_across,
                                                               pixels_down,
                                                               ppd,
                                                               fine_scale)
    
    # the resampled stimulus array
    stim_arr_coarse = utils.resample_stimulus(stim_arr,coarse_scale)
    deg_x_coarse,deg_y_coarse = utils.generate_coordinate_matrices(pixels_across,
                                                                   pixels_down,
                                                                   ppd,
                                                                   coarse_scale)
    
    # bounds for the 4-parameter brute-force grid-search
    bounds = ((-10,10),(-10,10),(0.25,5.25),(-4,4))
    
    # grab a voxel's time-series and z-score it
    ts_voxel = response[0,clip_number::]
    ts_voxel = utils.zscore(ts_voxel)
    
    # compute the initial guess with the adaptive brute-force grid-search
    x0, y0, s0, hrf0 = pest.adaptive_brute_force_grid_search(bounds,
                                                             1,
                                                             3,
                                                             ts_voxel,
                                                             deg_x_coarse,
                                                             deg_y_coarse,
                                                             stim_arr_coarse)
        
    # grab the known pRF estimate for the sample data
    gold_standard = np.array([ 8.446,  2.395,  0.341, -0.777])
    
    # package some of the results for comparison with known results
    test_results = np.round(np.array([x0,y0,s0,hrf0]),3)
        
    # assert
    npt.almost_equal(np.any(gold_standard == test_results))
    
def test_voxel_prf():
    """
    Test voxel-wise prf estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # set the path to data
    data_path = os.path.join(os.path.expanduser('~'), '.popeye/popeye')
    
    # load the datasets
    stimulus = np.load('%s/sample_stimulus.npy' %(data_path))
    response = np.load('%s/sample_response.npy' %(data_path))
    estimate = np.load('%s/sample_estimate.npy' %(data_path))
    
    # stimulus and display parameters 
    monitor_width = 25.0 # distance across the width of the image on the
                        # projection screen in cm 
    viewing_distance = 38.0 # viewing distance from the subject's eye to the
                           # projection screen in cm 
    pixels_across = 800 # display resolution across in pixels
    pixels_down = 600 # display resolution down in pixels
    ppd = np.pi*pixels_across/np.arctan(monitor_width/viewing_distance/2.0)/360.0 # degrees of visual angle
    clip_number = 10 # TRs to remove at the beginning
    roll_number = -2 # TRs to rotate the time-series.
    fine_scale = 1.0 # Decimal describing how much to down-sample the
                          # stimulus for increased fitting speed 
    coarse_scale = 0.05 # Decimal describing how much to down-sample the
                             # stimulus for increased fitting speed
    
    
    # the non-resampled stimulus array
    stim_arr = stimulus[:,:,clip_number::]
    stim_arr = np.roll(stim_arr,roll_number,axis=-1)
    stim_arr_fine = utils.resample_stimulus(stim_arr,fine_scale)
    deg_x_fine,deg_y_fine = utils.generate_coordinate_matrices(pixels_across,
                                                               pixels_down,
                                                               ppd,
                                                               fine_scale)
    
    # the resampled stimulus array
    stim_arr_coarse = utils.resample_stimulus(stim_arr,coarse_scale)
    deg_x_coarse,deg_y_coarse = utils.generate_coordinate_matrices(pixels_across,
                                                                   pixels_down,
                                                                   ppd,
                                                                   coarse_scale)
    
    # bounds for the 4-parameter brute-force grid-search
    bounds = ((-10,10),(-10,10),(0.25,5.25),(-4,4))
    
    # compute the pRF estimate for each time-series and compare to known results
    for voxel in np.arange(0,5):
        
        # grab the voxel's time-series and z-score it
        ts_voxel = response[voxel,clip_number::]
        ts_voxel = utils.zscore(ts_voxel)
        
        x, y, sigma, hrf_delay, err, stats = pest.voxel_prf(ts_voxel,
                                                            deg_x_coarse,
                                                            deg_y_coarse,
                                                            deg_x_fine,
                                                            deg_y_fine,
                                                            stim_arr_coarse,
                                                            stim_arr_fine,
                                                            bounds,
                                                            norm_func=utils.zscore,
                                                            uncorrected_rval=0)
        
        
        # grab the known pRF estimate for the sample data
        gold_standard = np.round(estimate[voxel,0:4],3)
        
        # package some of the results for comparison with known results
        test_results = np.round(np.array([x,y,sigma,hrf_delay]),3)
        
        # assert equivalence
        npt.assert_true(np.any(gold_standard == test_results))
