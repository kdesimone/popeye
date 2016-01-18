import ctypes, multiprocessing
from itertools import repeat

import numpy as np
import nose.tools as nt
import numpy.testing as npt

import nibabel

import popeye.utilities as utils
import popeye.spinach as spin
import popeye.og as og
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus, resample_stimulus

def test_recast_estimation_results():
    
    # create 3 known prf estimates to fit    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_blank_steps = 20
    num_bar_steps = 40
    ecc = 12
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.20
    resample_factor = 0.25
    pixels_across = 800 * resample_factor
    pixels_down = 600 * resample_factor
    dtype = ctypes.c_int16
    Ns = 3
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 0
    
    # insert blanks
    thetas = list(thetas)
    thetas.insert(0,-1)
    thetas.insert(2,-1)
    thetas.insert(5,-1)
    thetas.insert(8,-1)
    thetas.insert(11,-1)
    thetas.append(-1)
    thetas = np.array(thetas)
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
    
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 1.1
    hrf_delay = -0.25
    
    # create data
    data = model.generate_prediction(x, y, sigma, beta, hrf_delay)
    
    # set search grid
    x_grid = (-10,10)
    y_grid = (-10,10)
    s_grid = (1/stimulus.ppd,5.25)
    b_grid = (0.1,1.0)
    h_grid = (-4.0,4.0)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (1/stimulus.ppd,12.0)
    b_bound = (1e-8,1e2)
    h_bound = (-5.0,5.0)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid, b_grid, h_grid)
    bounds = (x_bound, y_bound, s_bound, b_bound, h_bound)
    
    # create 3 voxels of data
    all_data = np.array([data,data,data])
    indices = [(0,0,0),(0,0,1),(0,0,2)]
    bundle = utils.multiprocess_bundle(og.GaussianFit, model, all_data,
                                       grids, bounds, Ns, indices, auto_fit, verbose)
    
    
    # run analysis
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    output = pool.map(utils.parallel_fit, bundle)
    pool.close()
    pool.join()
    
    # create grid parent
    arr = np.zeros((1,1,3))
    grid_parent = nibabel.Nifti1Image(arr,np.eye(4,4))
    
    # recast the estimation results
    nif = utils.recast_estimation_results(output, grid_parent)
    dat = nif.get_data()
    
    # assert equivalence
    npt.assert_almost_equal(np.mean(dat[...,0]), x)
    npt.assert_almost_equal(np.mean(dat[...,1]), y)
    npt.assert_almost_equal(np.mean(dat[...,2]), sigma)
    npt.assert_almost_equal(np.mean(dat[...,3]), beta)
    npt.assert_almost_equal(np.mean(dat[...,4]), hrf_delay)

def test_make_nifti():
    
    # make up a volume
    arr = np.zeros((3,3,2))
    arr[...,0] = 1
    arr[...,1] = 100
    
    # make a nifti
    nif = utils.make_nifti(arr)
    
    npt.assert_equal(arr.shape,nif.shape)
    npt.assert_equal(np.mean(arr[...,0]),1)
    npt.assert_equal(np.mean(arr[...,1]),100)
    npt.assert_equal(nif.get_affine(),np.eye(4,4))
    
    # same except hand it a grid_parent
    grid_parent = nibabel.Nifti1Image(arr,np.eye(4,4))
    
    # make a nifti
    nif = utils.make_nifti(arr, grid_parent)
    
    npt.assert_equal(arr.shape,nif.shape)
    npt.assert_equal(np.mean(arr[...,0]),1)
    npt.assert_equal(np.mean(arr[...,1]),100)
    npt.assert_equal(nif.get_affine(),np.eye(4,4))

def test_normalize():
    
    # create an array
    arr = np.linspace(0,1,100)
    
    # create new bounds
    lo = 0.0
    hi = 200.0
    
    # rescale the array
    arr_new = utils.normalize(arr, lo, hi)
    
    # assert equivalence
    npt.assert_equal(np.min(arr_new), lo)
    npt.assert_equal(np.max(arr_new), hi)

def test_error_function():
    
    # create a parameter to estimate
    params = (10.0,)
    
    # set bounds
    bounds = ((0.0,20.0),)
    
    # set the verbose level 0 is silent, 1 is final estimate, 2 is each iteration
    verbose = 0
    
    # create a simple function to transform the parameters
    func = lambda freq: np.sin( np.linspace(0,1,1000) * 2 * np.pi * freq)
    
    # create a "response"
    response = func(params[0])
    
    # assert 0 error
    npt.assert_equal(utils.error_function(params, bounds, response, func, verbose),0)

def test_gradient_descent_search():
    
    # create a parameter to estimate
    params = (10,10)
    
    # set grids + bounds
    grids = ((0,20),(5,15))
    bounds = ()
    
    # set the verbose level 0 is silent, 1 is final estimate, 2 is each iteration
    verbose = 0
    
    # set the number of search samples
    Ns = 3
    
    # create a simple function to transform the parameters
    func = lambda freq, offset: np.sin( np.linspace(0,1,1000) * 2 * np.pi * freq) + offset
    
    # create a "response"
    response = func(*params)
    
    # get the fine estimate
    phat = utils.gradient_descent_search((8,8), bounds, response, utils.error_function, func, verbose)
    
    # assert that the estimate is equal to the parameter
    npt.assert_almost_equal(params, phat[0])

def test_brute_force_search():
    
    # create a parameter to estimate
    params = (10,10)
    
    # we need to define some search bounds
    grids = ((0,20),(5,15))
    
    # we don't need to specify bounds for the error function
    bounds = ()
    
    # set the number of grid samples for the coarse search
    Ns = 3
    
    # set the verbose level 0 is silent, 1 is final estimate, 2 is each iteration
    verbose = 0
    
    # create a simple function to transform the parameters
    func = lambda freq, offset: np.sin( np.linspace(0,1,1000) * 2 * np.pi * freq) + offset
    
    # create a "response"
    response = func(*params)
    
    # get the ball-park estimate
    p0 = utils.brute_force_search(grids, bounds, Ns, response, utils.error_function, func, verbose)
                                        
    # assert that the estimate is equal to the parameter
    npt.assert_equal(params, p0[0])
    

def test_double_gamma_hrf():
    """
    Test voxel-wise gabor estimation function in popeye.estimation 
    using the stimulus and BOLD time-series data that ship with the 
    popeye installation.
    """
    
    # set the TR length ... this affects the HRF sampling rate ...
    tr_length = 1.0
    
    # compute the difference in area under curve for hrf_delays of -1 and 0
    diff_1 = np.abs(np.sum(utils.double_gamma_hrf(-1, tr_length))-np.sum(utils.double_gamma_hrf(0, tr_length)))
    
    # compute the difference in area under curver for hrf_delays of 0 and 1
    diff_2 = np.abs(np.sum(utils.double_gamma_hrf(1, tr_length))-np.sum(utils.double_gamma_hrf(0, tr_length)))
    
    npt.assert_almost_equal(diff_1, diff_2, 2)

def test_randomize_voxels():
    
    # set the dummy dataset size
    x_dim, y_dim, z_dim = 10, 10, 10
    
    # create a dummy dataset with the 
    dat = np.random.rand(x_dim, y_dim, z_dim)
    
    # mask and grab the indices
    xi,yi,zi = np.nonzero(dat>0.75)
    
    # create a random vector to resort the voxel indices
    rand_vec = np.random.rand(len(xi))
    rand_ind = np.argsort(rand_vec)
    
    # resort the indices
    rand_xi, rand_yi, rand_zi = xi[rand_ind], yi[rand_ind], zi[rand_ind]
    
    # assert that all members of the original and resorted indices are equal
    nt.assert_true(set(xi) == set(rand_xi))
    nt.assert_true(set(yi) == set(rand_yi))
    nt.assert_true(set(zi) == set(rand_zi))
    
def test_zscore():
    
    x = np.array([[1, 1, 3, 3],
                  [4, 4, 6, 6]])
                  
    z = utils.zscore(x)
    npt.assert_equal(x.shape, z.shape)
    
    #Default axis is -1
    npt.assert_equal(utils.zscore(x), np.array([[-1., -1., 1., 1.],
                                                      [-1., -1., 1., 1.]]))
                                                      
    #Test other axis:
    npt.assert_equal(utils.zscore(x, 0), np.array([[-1., -1., -1., -1.],
                                                        [1., 1., 1., 1.]]))
                                                        
                                                        
    # Test the 1D case:
    x = np.array([1, 1, 3, 3])
    npt.assert_equal(utils.zscore(x), [-1, -1, 1, 1])
    
    
def test_percent_change():
    x = np.array([[99, 100, 101], [4, 5, 6]])
    p = utils.percent_change(x)
    
    nt.assert_equal(x.shape, p.shape)
    nt.assert_almost_equal(p[0, 2], 1.0)
    
    ts = np.arange(4 * 5).reshape(4, 5)
    ax = 0
    npt.assert_almost_equal(utils.percent_change(ts, ax), np.array(
        [[-100., -88.23529412, -78.94736842, -71.42857143, -65.2173913],
        [-33.33333333, -29.41176471, -26.31578947, -23.80952381, -21.73913043],
        [33.33333333,   29.41176471,   26.31578947,   23.80952381, 21.73913043],
        [100., 88.23529412, 78.94736842, 71.42857143, 65.2173913]]))
        
    ax = 1
    npt.assert_almost_equal(utils.percent_change(ts, ax), np.array(
        [[-100., -50., 0., 50., 100.],
         [-28.57142857, -14.28571429, 0., 14.28571429, 28.57142857],
          [-16.66666667, -8.33333333, 0., 8.33333333, 16.66666667],
          [-11.76470588, -5.88235294, 0., 5.88235294, 11.76470588]]))


def test_parallel_fit():
    
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_blank_steps = 20
    num_bar_steps = 40
    ecc = 12
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.20
    resample_factor = 0.25
    pixels_across = 800 * resample_factor
    pixels_down = 600 * resample_factor
    dtype = ctypes.c_int16
    Ns = 3
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    
    # insert blanks
    thetas = list(thetas)
    thetas.insert(0,-1)
    thetas.insert(2,-1)
    thetas.insert(5,-1)
    thetas.insert(8,-1)
    thetas.insert(11,-1)
    thetas.append(-1)
    thetas = np.array(thetas)
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance, 
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5
    hrf_delay = -0.25
    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta, hrf_delay)
    
    # make 3 voxels
    data = np.array([data,data,data])
    num_voxels = data.shape[0]
    
    # set search grid
    x_grid = (-10,10)
    y_grid = (-10,10)
    s_grid = (0.25,5.25)
    b_grid = (0.1,1.0)
    h_grid = (-4.0,4.0)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (0.001,12.0)
    b_bound = (1e-8,1e2)
    h_bound = (-5.0,5.0)
    
    # make grids+bounds for all voxels in the sample
    grids = [(x_grid, y_grid, s_grid, b_grid, h_grid),]*num_voxels
    bounds = [(x_bound, y_bound, s_bound, b_bound, h_bound),]*num_voxels
    
    # package the data structure
    dat = zip(repeat(og.GaussianFit,num_voxels),
              repeat(model,num_voxels),
              data,
              grids,
              bounds,
              repeat(3,num_voxels),
              repeat((1,2,3),num_voxels),
              repeat(True,num_voxels),
              repeat(0,num_voxels))
              
    # run analysis
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    output = pool.map(utils.parallel_fit,dat)
    pool.close()
    pool.join()
    
    # assert equivalence
    for fit in output:
        nt.assert_almost_equal(fit.x, x, 2)
        nt.assert_almost_equal(fit.y, y, 2)
        nt.assert_almost_equal(fit.sigma, sigma, 2)
        nt.assert_almost_equal(fit.beta, beta, 2)
        nt.assert_almost_equal(fit.hrf_delay, hrf_delay, 2)

def test_binner():
    
    signal = np.ones(10)
    times = np.linspace(0,1,10)
    bins = np.arange(-0.5,1.5,0.5)
    
    binned_signal = utils.binner(signal, times, bins)
    
    nt.assert_true(len(binned_signal), len(bins)-2)
    nt.assert_true(np.all(binned_signal==[5,5]))
    