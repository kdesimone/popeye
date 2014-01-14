import os
import popeye.utilities as utils
import numpy as np
import nose.tools as nt
import numpy.testing as npt

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
    

def test_generate_coordinate_matrices():
    
    # set some dummy display parameters
    pixels_across = 100
    pixels_down = 100
    ppd = 1.0
    scale_factor = 1.0
    
    # generate coordinates
    deg_x, deg_y = utils.generate_coordinate_matrices(pixels_across,pixels_down,ppd,scale_factor)
    
    # assert
    nt.assert_true(np.sum(deg_x[0,0:50]) == np.sum(deg_x[0,50::])*-1)
    nt.assert_true(np.sum(deg_y[0:50,0]) == np.sum(deg_y[50::,0])*-1)
    
    # try the same with an odd number of pixels
    pixels_across = 101
    pixels_down = 101
    
    # generate coordinates
    deg_x, deg_y = utils.generate_coordinate_matrices(pixels_across,pixels_down,ppd,scale_factor)
    
    # assert
    nt.assert_true(np.sum(deg_x[0,0:50]) == np.sum(deg_x[0,50::])*-1)
    nt.assert_true(np.sum(deg_y[0:50,0]) == np.sum(deg_y[50::,0])*-1)
    
    # try with another rescaling factor
    scale_factor = 0.5
    
    # get the horizontal and vertical coordinate matrices
    deg_x, deg_y = utils.generate_coordinate_matrices(pixels_across,pixels_down,ppd,scale_factor)
    
    # assert
    nt.assert_true(np.sum(deg_x[0,0:50]) == np.sum(deg_x[0,50::])*-1)
    nt.assert_true(np.sum(deg_y[0:50,0]) == np.sum(deg_y[50::,0])*-1)


def test_resample_stimulus():
    
    # set the downsampling rate
    scale_factor = 0.5
    
    # set the path to data
    data_path = os.path.join(os.path.expanduser('~'), '.popeye/popeye')
    
    # load the datasets
    stimulus = np.load('%s/sample_stimulus.npy' %(data_path))
    
    # downsample the stimulus by 50%
    stimulus_coarse = utils.resample_stimulus(stimulus,scale_factor)
    
    # grab the stimulus dimensions
    stim_dims = np.shape(stimulus)
    stim_coarse_dims = np.shape(stimulus_coarse)
    
    # assert
    nt.assert_true(stim_coarse_dims[0]/stim_dims[0] == scale_factor)
    nt.assert_true(stim_coarse_dims[1]/stim_dims[1] == scale_factor)
    nt.assert_true(stim_coarse_dims[2] == stim_dims[2])
    
    
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
