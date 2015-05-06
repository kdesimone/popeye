import os
import popeye.utilities as utils
import numpy as np
import nose.tools as nt
import numpy.testing as npt
from scipy.special import gamma

def test_normalize():
    
    # create an array
    arr = np.arange(0,100)
    
    # create new bounds
    lo = -50
    hi = 50
    
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
    params = (10.0,5)
    
    # set grids + bounds
    grids = ((2.0,15.0),(0.0,10.0))
    bounds = ((0.0,20.0),(-1.0,20.0))
    
    # set the verbose level 0 is silent, 1 is final estimate, 2 is each iteration
    verbose = 0
    
    # set the number of search samples
    Ns = 5
    
    # create a simple function to transform the parameters
    func = lambda freq, offset: np.sin( np.linspace(0,1,1000) * 2 * np.pi * freq) + offset
    
    # create a "response"
    response = func(*params)
    
    # get the ball-park estimate
    p0 = utils.brute_force_search(grids, bounds, Ns, response, utils.error_function, func, verbose)
    
    # get the fine estimate
    phat = utils.gradient_descent_search(p0, bounds, response, utils.error_function, func, verbose)
    
    # assert that the estimate is equal to the parameter
    npt.assert_almost_equal(params, phat)

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
    npt.assert_equal(params, p0)
    

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
