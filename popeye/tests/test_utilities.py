from __future__ import division
import ctypes, sharedmem, sys

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

import numpy as np
import nose.tools as nt
import numpy.testing as npt

import nibabel

import popeye.utilities as utils
import popeye.spinach as spin
import popeye.ogb_nohrf as ogb
import popeye.og as og
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus, resample_stimulus, generate_coordinate_matrices

def test_distance_mask():

    x = 0
    y = 0
    sigma = 1
    amplitude=100
    dx,dy = np.meshgrid(np.linspace(-50,50,100),np.linspace(-50,50,100))
    mask = utils.distance_mask(x,y,sigma,dx,dy)
    npt.assert_equal(np.sqrt(np.sum(mask)),2)
    npt.assert_equal(np.max(mask),1)
    mask = utils.distance_mask(x,y,sigma,dx,dy,amplitude)
    npt.assert_equal(np.max(mask),amplitude)


def test_grid_slice():

    # test this case
    from_1 = 0
    to_1 = 20
    from_2 = 0
    to_2 = 2
    Ns=5

    # set a parameter to estimate
    params = (10,1)

    # see if we properly tile the parameter space for Ns=2
    grid_1 = utils.grid_slice(from_1, to_1, Ns)
    grid_2 = utils.grid_slice(from_2, to_2, Ns)
    grids = (grid_1, grid_2)

    # unbounded
    bounds = ()

    # create a simple function to generate a response from the parameter
    func = lambda freq,offset: np.sin( np.linspace(0,1,1000) * 2 * np.pi * freq) + offset

    # create a "response"
    response = func(*params)

    # get the ball-park estimate
    p0 = utils.brute_force_search(response, utils.error_function, func, grids, bounds)

    # make sure we fit right
    npt.assert_equal(params, p0[0])

    # make sure we sliced it right
    npt.assert_equal(p0[2][0].min(),from_1)
    npt.assert_equal(p0[2][0].max(),to_1)
    npt.assert_equal(p0[2][1].min(),from_2)
    npt.assert_equal(p0[2][1].max(),to_2)

    # test this case
    from_1 = 0
    to_1 = 20
    from_2 = 1
    to_2 = 2
    Ns=2

    # set a parameter to estimate
    params = (0,1)

    # see if we properly tile the parameter space for Ns=2
    grid_1 = utils.grid_slice(from_1, to_1, Ns)
    grid_2 = utils.grid_slice(from_2, to_2, Ns)
    grids = (grid_1, grid_2)

    # unbounded
    bounds = ()

    # create a simple function to generate a response from the parameter
    func = lambda freq,offset: np.sin( np.linspace(0,1,1000) * 2 * np.pi * freq) + offset

    # create a "response"
    response = func(*params)

    # get the ball-park estimate
    p0 = utils.brute_force_search(response, utils.error_function, func, grids, bounds)

    # make sure we fit right
    npt.assert_equal(params, p0[0])

    # make sure we sliced it right
    npt.assert_equal(p0[2][0].min(),from_1)
    npt.assert_equal(p0[2][0].max(),to_1)
    npt.assert_equal(p0[2][1].min(),from_2)
    npt.assert_equal(p0[2][1].max(),to_2)



def test_recast_estimation_results():

    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_blank_steps = 0
    num_bar_steps = 30
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.10
    pixels_down = 100
    pixels_across = 100
    dtype = ctypes.c_int16
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance,
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.spm_hrf)
    model.hrf_delay = 0
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5
    baseline = -0.25
    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta, baseline)
    
    # set search grid
    x_grid = utils.grid_slice(-5,4,5)
    y_grid = utils.grid_slice(-5,7,5)
    s_grid = utils.grid_slice(1/stimulus.ppd,5.25,5)
    b_grid = utils.grid_slice(0.1,4.0,5)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (1/stimulus.ppd,12.0)
    b_bound = (1e-8,1e2)
    m_bound = (None,None)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid,)
    bounds = (x_bound, y_bound, s_bound, b_bound, m_bound)
    
    # create 3 voxels of data
    all_data = np.array([data,data,data])
    indices = [(0,0,0),(0,0,1),(0,0,2)]
    
    # bundle the voxels
    bundle = utils.multiprocess_bundle(og.GaussianFit, model, all_data, grids, bounds, indices)
    
    # run analysis
    with sharedmem.Pool(np=3) as pool:
        output = pool.map(utils.parallel_fit, bundle)
        
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
    npt.assert_almost_equal(np.mean(dat[...,4]), baseline)
    
    # recast the estimation results - OVERLOADED
    nif = utils.recast_estimation_results(output, grid_parent, True)
    dat = nif.get_data()
    
    # assert equivalence
    npt.assert_almost_equal(np.mean(dat[...,0]), np.arctan2(y,x),2)
    npt.assert_almost_equal(np.mean(dat[...,1]), np.sqrt(x**2+y**2),2)
    npt.assert_almost_equal(np.mean(dat[...,2]), sigma)
    npt.assert_almost_equal(np.mean(dat[...,3]), beta)
    npt.assert_almost_equal(np.mean(dat[...,4]), baseline)


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

    # 1D
    arr = np.linspace(0,1,100)
    lo = 100.0
    hi = 200.0
    arr_new = utils.normalize(arr, lo, hi)
    npt.assert_equal(np.min(arr_new), lo)
    npt.assert_equal(np.max(arr_new), hi)

    # 2D
    arr = np.tile(np.linspace(0,1,100),(10,1))
    lo = np.repeat(lo,10)
    hi = np.repeat(hi,10)
    arr_new = utils.normalize(arr, lo, hi)
    npt.assert_equal(np.min(arr_new), lo)
    npt.assert_equal(np.max(arr_new), hi)

def test_error_function():

    # create a parameter to estimate
    params = (10.0,)

    # set bounds
    bounds = ((0.0,20.0),)

    # set the verbose level 0 is silent, 1 is final estimate, 2 is each iteration
    verbose = 2

    # create a simple function to transform the parameters
    func = lambda freq: np.sin( np.linspace(0,1,1000) * 2 * np.pi * freq)

    # create a "response"
    response = func(params[0])

    # assert 0 error
    npt.assert_equal(utils.error_function(params, bounds, response, func, verbose),0)

    # assert parameter outside of bounds return inf
    params = (30.0,)
    npt.assert_equal(utils.error_function(params, bounds, response, func, verbose), np.inf)

    # test nan returns inf
    response = func(params)
    response[0] = np.nan
    err = utils.error_function(params, bounds, response, func, verbose)
    npt.assert_equal(err,np.inf)

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
    phat = utils.gradient_descent_search(response, utils.error_function, func, (8,8), bounds, verbose)

    # assert that the estimate is equal to the parameter
    npt.assert_almost_equal(params, phat[0])

def test_brute_force_search_manual_grids():

    # create a parameter to estimate
    params = (10,10)

    # we need to define some search bounds
    grid_1 = utils.grid_slice(0,20,5)
    grid_2 = utils.grid_slice(5,15,5)
    grids = (grid_1,grid_2,)
    bounds = ()

    # set the verbose level 0 is silent, 1 is final estimate, 2 is each iteration
    verbose = 0

    # create a simple function to transform the parameters
    func = lambda freq, offset: np.sin( np.linspace(0,1,1000) * 2 * np.pi * freq) + offset

    # create a "response"
    response = func(*params)

    # get the ball-park estimate
    p0 = utils.brute_force_search(response, utils.error_function, func, grids, bounds)

    # assert that the estimate is equal to the parameter
    npt.assert_equal(params, p0[0])

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
    p0 = utils.brute_force_search(response, utils.error_function, func, grids, bounds, Ns=Ns)

    # assert that the estimate is equal to the parameter
    npt.assert_equal(params, p0[0])


def test_double_gamma_hrf():

    # set the TR length ... this affects the HRF sampling rate ...
    tr_length = 1.0

    hrf_0 = utils.double_gamma_hrf(-1, tr_length)
    hrf_1 = utils.double_gamma_hrf(0, tr_length)
    hrf_2 = utils.double_gamma_hrf(1, tr_length)
    npt.assert_almost_equal(hrf_0.sum(), hrf_1.sum(), 3)
    npt.assert_almost_equal(hrf_0.sum(), hrf_2.sum(), 3)
    npt.assert_almost_equal(hrf_1.sum(), hrf_2.sum(), 3)

    hrf_0 = utils.double_gamma_hrf(-1, tr_length, integrator=None)
    hrf_1 = utils.double_gamma_hrf(0, tr_length, integrator=None)
    hrf_2 = utils.double_gamma_hrf(1, tr_length, integrator=None)
    npt.assert_array_less(hrf_0.sum(),hrf_1.sum())
    npt.assert_array_less(hrf_1.sum(),hrf_2.sum())
    npt.assert_array_less(hrf_0.sum(),hrf_2.sum())


def test_spm_hrf():

    # set the TR length ... this affects the HRF sampling rate ...
    tr_length = 1.0

    # compute the difference in area under curve for hrf_delays of -1 and 0
    diff_1 = np.abs(np.sum(utils.spm_hrf(-1, tr_length))-np.sum(utils.spm_hrf(0, tr_length)))

    # compute the difference in area under curver for hrf_delays of 0 and 1
    diff_2 = np.abs(np.sum(utils.spm_hrf(1, tr_length))-np.sum(utils.spm_hrf(0, tr_length)))

    npt.assert_almost_equal(diff_1, diff_2, 2)

def test_double_gamma_hrf():

    # set the TR length ... this affects the HRF sampling rate ...
    tr_length = 1.0

    # compute the difference in area under curve for hrf_delays of -1 and 0
    diff_1 = np.abs(np.sum(utils.spm_hrf(-1, tr_length))-np.sum(utils.spm_hrf(0, tr_length)))

    # compute the difference in area under curver for hrf_delays of 0 and 1
    diff_2 = np.abs(np.sum(utils.spm_hrf(1, tr_length))-np.sum(utils.spm_hrf(0, tr_length)))

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
    npt.assert_equal(set(xi),set(rand_xi))
    npt.assert_equal(set(yi),set(rand_yi))
    npt.assert_equal(set(zi),set(rand_zi))

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

    npt.assert_equal(x.shape, p.shape)
    npt.assert_almost_equal(p[0, 2], 1.0)

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



def test_parallel_fit_Ns():

    # stimulus features
    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,90)
    num_blank_steps = 0
    num_bar_steps = 30
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.10
    pixels_down = 100
    pixels_across = 100
    dtype = ctypes.c_int16
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    Ns = 3
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance,
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    model.hrf_delay = 0
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5
    baseline = -0.25
    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta, baseline)
    
    # make 3 voxels
    all_data = np.array([data,data,data])
    num_voxels = data.shape[0]
    indices = [(1,2,3)]*3
    
    # set search grid
    x_grid = (-10,10)
    y_grid = slice(-10,10)
    s_grid = (0.25,5.25)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (0.001,12.0)
    b_bound = (1e-8,1e2)
    m_bound = (None, None)
    
    # make grids+bounds for all voxels in the sample
    grids = (x_grid, y_grid, s_grid,)
    bounds = (x_bound, y_bound, s_bound, b_bound, m_bound)
    
    # fitting params
    auto_fit = True
    verbose = 1
    
    # bundle the voxels
    bundle = utils.multiprocess_bundle(og.GaussianFit, model, all_data, grids, bounds, indices, Ns=3)
    
    # run analysis
    with sharedmem.Pool(np=3) as pool:
        output = pool.map(utils.parallel_fit, bundle)
        
    # assert equivalence
    for fit in output:
        npt.assert_almost_equal(fit.x, x, 2)
        npt.assert_almost_equal(fit.y, y, 2)
        npt.assert_almost_equal(fit.sigma, sigma, 2)
        npt.assert_almost_equal(fit.beta, beta, 2)
        npt.assert_almost_equal(fit.baseline, baseline, 2)
        
def test_parallel_fit():

    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_blank_steps = 0
    num_bar_steps = 30
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.10
    pixels_down = 100
    pixels_across = 100
    dtype = ctypes.c_int16
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance,
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    model.hrf_delay = 0
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5
    baseline = -0.25
    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta, baseline)
    
    # set search grid
    x_grid = slice(-5,4,5)
    y_grid = slice(-5,7,5)
    s_grid = slice(1/stimulus.ppd,5.25,5)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (1/stimulus.ppd,12.0)
    b_bound = (1e-8,1e2)
    m_bound = (None, None)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid)
    bounds = (x_bound, y_bound, s_bound, b_bound, m_bound)
    
    # make 3 voxels
    all_data = np.array([data,data,data])
    num_voxels = data.shape[0]
    indices = [(1,2,3)]*3
    
    # bundle the voxels
    bundle = utils.multiprocess_bundle(og.GaussianFit, model, all_data, grids, bounds, indices)
    
    fit = utils.parallel_fit(bundle[0])
    
    # assert equivalence
    npt.assert_almost_equal(fit.x, x, 2)
    npt.assert_almost_equal(fit.y, y, 2)
    npt.assert_almost_equal(fit.sigma, sigma, 2)
    npt.assert_almost_equal(fit.beta, beta, 2)
    npt.assert_almost_equal(fit.baseline, baseline, 2)

def test_parallel_fit_manual_grids():

    # stimulus features
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_blank_steps = 0
    num_bar_steps = 30
    ecc = 10
    tr_length = 1.0
    frames_per_tr = 1.0
    scale_factor = 0.10
    pixels_down = 100
    pixels_across = 100
    dtype = ctypes.c_int16
    voxel_index = (1,2,3)
    auto_fit = True
    verbose = 1
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance,
                                screen_width, thetas, num_bar_steps, num_blank_steps, ecc)
                                
    # create an instance of the Stimulus class
    stimulus = VisualStimulus(bar, viewing_distance, screen_width, scale_factor, tr_length, dtype)
    
    # initialize the gaussian model
    model = og.GaussianModel(stimulus, utils.double_gamma_hrf)
    model.hrf_delay = 0
    
    # generate a random pRF estimate
    x = -5.24
    y = 2.58
    sigma = 1.24
    beta = 2.5
    baseline = -0.25
    
    # create the "data"
    data = model.generate_prediction(x, y, sigma, beta, baseline)
    
    # set search grid
    x_grid = slice(-5,4,5)
    y_grid = slice(-5,7,5)
    s_grid = slice(1/stimulus.ppd,5.25,5)
    b_grid = slice(0.1,4.0,5)
    
    # set search bounds
    x_bound = (-12.0,12.0)
    y_bound = (-12.0,12.0)
    s_bound = (1/stimulus.ppd,12.0)
    b_bound = (1e-8,1e2)
    m_bound = (None, None)
    
    # loop over each voxel and set up a GaussianFit object
    grids = (x_grid, y_grid, s_grid,)
    bounds = (x_bound, y_bound, s_bound, b_bound, m_bound)
    
    # make 3 voxels
    all_data = np.array([data,data,data])
    num_voxels = data.shape[0]
    indices = [(1,2,3)]*3
    
    # bundle the voxels
    bundle = utils.multiprocess_bundle(og.GaussianFit, model, all_data, grids, bounds, indices)
    
    # run analysis
    with sharedmem.Pool(np=sharedmem.cpu_count()-1) as pool:
        output = pool.map(utils.parallel_fit, bundle)
        
    # assert equivalence
    for fit in output:
        npt.assert_almost_equal(fit.x, x, 2)
        npt.assert_almost_equal(fit.y, y, 2)
        npt.assert_almost_equal(fit.sigma, sigma, 2)
        npt.assert_almost_equal(fit.beta, beta, 2)
        npt.assert_almost_equal(fit.baseline, baseline, 2)

def test_gaussian_2D():

    # set some dummy display parameters
    pixels_across = 101
    pixels_down = 101
    ppd = 1.0
    scale_factor = 1.0

    # generate coordinates
    deg_x, deg_y = generate_coordinate_matrices(pixels_across,pixels_down,ppd,scale_factor)

    # generate 2D case
    G = utils.gaussian_2D(deg_x,deg_y,0,0,10,10,0)

    # generate 1D case
    gx = np.exp(-((deg_x[0,:]-0)**2)/(2*10**2))
    gy = np.exp(-((deg_y[:,0]-0)**2)/(2*10**2))

    # assertions
    npt.assert_equal(np.round(G[:,50],5),np.round(gx,5))
    npt.assert_equal(np.round(G[50,:],5),np.round(gy,5))
    
def test_cartes_to_polar():
    cartes = np.array([5,0]).astype('double')
    polar = utils.cartes_to_polar(cartes)
    npt.assert_equal(polar[...,0], 0)
    npt.assert_equal(polar[...,1], 5)

    cartes = np.array([-5,0]).astype('double')
    polar = utils.cartes_to_polar(cartes)
    npt.assert_equal(polar[...,0], np.pi)
    npt.assert_equal(polar[...,1], 5)

    cartes = np.array([0,5]).astype('double')
    polar = utils.cartes_to_polar(cartes)
    npt.assert_equal(polar[...,0], np.pi/2)
    npt.assert_equal(polar[...,1], 5)

    cartes = np.array([0,-5]).astype('double')
    polar = utils.cartes_to_polar(cartes)
    npt.assert_equal(polar[...,0], np.pi*3/2)
    npt.assert_equal(polar[...,1], 5)

def test_binner():

    signal = np.ones(10)
    times = np.linspace(0,1,10)
    bins = np.arange(-0.5,1.5,0.5)
    binned_signal = utils.binner(signal, times, bins)

    npt.assert_equal(len(binned_signal), len(bins)-2)
    npt.assert_equal(binned_signal,[5,5])

def test_find_files():

    f = open('/tmp/test_abc.txt', 'w')
    f.close()

    path = utils.find_files('/tmp/','test*.txt')

    npt.assert_equal(path[0],'/tmp/test_abc.txt')

def test_peakdet():

    ts = np.zeros(100)

    peaks = np.arange(0,100,20)
    troughs = np.arange(10,100,20)

    ts[peaks] = 1
    ts[troughs] = -1

    a,b = utils.peakdet(ts,0.5)

    npt.assert_equal(a[:,0], peaks)
    npt.assert_equal(b[:,0], troughs)
    npt.assert_equal(a[:,1], 1)
    npt.assert_equal(b[:,1], -1)

# def test_OLS():
#
#     o = utils.ols(np.arange(100),np.arange(100))
#
#     npt.assert_equal(o.R2,1.0)
#     npt.assert_almost_equal(np.sum(o.e),0.0)
#     npt.assert_almost_equal(np.sum(o.se),0.0)
#     npt.assert_true(o.F == np.inf)
#     npt.assert_almost_equal(np.sum(o.b),1.0)
#     npt.assert_true(o.df_e == len(np.arange(100-2)))
#     npt.assert_almost_equal(o.p[1],0)
#     npt.assert_true(o.ll() == (2987.7752827161585, -59.715505654323174, -59.663402250603411))
#     npt.assert_true(o.nobs == 100)
#     omni_1 = o.omni()[0]
#     omni_2 = o.omni()[1]
#     npt.assert_almost_equal(omni_1,18.093297390235648)
#     npt.assert_almost_equal(omni_2, 0.00011778511003501986)
#     npt.assert_true(o.JB() == (5.0825725194665461,0.07876502232916649,0.16483617111543283,1.9458968022816807))
#     npt.assert_true(o.dw() == 0.0051450432267976026)
#
