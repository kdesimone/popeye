import numpy as np
import numpy.testing as npt
import nose.tools as nt

import popeye.utilities as utils
import popeye.spinach as spin
from popeye.visual_stimulus import generate_coordinate_matrices
from popeye.spinach import generate_og_receptive_field, generate_og_timeseries

def test_generate_og_receptive_field():
    xpixels = 500 # simulated screen width
    ypixels = 500 # simulated screen height
    ppd = 1 # simulated visual angle
    scale_factor = 1.0 # simulated stimulus resampling rate
    distance = 5 # standard deviations to compute gauss out to
    xcenter = 0 # x coordinate of the pRF center
    ycenter = 0 # y coordinate of the pRF center
    sigma = 1 # width of the pRF
    
    test_value = 6 # this is the sum of a gaussian given 1 ppd 
                   # and a 1 sigma prf centered on (0,0)
                   
    # generate the visuotopic coordinates
    dx,dy = generate_coordinate_matrices(xpixels,
                                         ypixels,
                                         ppd,
                                         scale_factor)
    
    # generate a pRF at (0,0) and 1 sigma wide
    rf = generate_og_receptive_field(xcenter, ycenter, sigma, dx, dy)
    
    # divide by integral
    rf /= 2 * np.pi * sigma ** 2
    
    # compare the volume of the pRF to a known value
    nt.assert_almost_equal(np.sum(rf),1)


def test_generate_og_timeseries():
    xpixels = 100 # simulated screen width
    ypixels = 100 # simulated screen height
    ppd = 1 # simulated visual angle
    scaleFactor = 1.0 # simulated stimulus resampling rate
    
    xcenter = 0 # x coordinate of the pRF center
    ycenter = 0 # y coordinate of the pRF center
    sigma = 10 # width of the pRF
    
    # generate the visuotopic coordinates
    dx,dy = generate_coordinate_matrices(xpixels,
                                         ypixels,
                                         ppd,
                                         scaleFactor)
    
    timeseries_length = 15 # number of frames to simulate our stimulus array
    
    # initialize the stimulus array
    stim_arr = np.zeros((xpixels, ypixels, timeseries_length)).astype('uint8')
    
    # make a circular mask appear for the first 5 frames
    xi,yi = np.nonzero(np.sqrt((dx-xcenter)**2 + (dy-ycenter)**2)<sigma)
    stim_arr[xi,yi,0:5] = 1
    
    # make an annulus appear for the next 5 frames
    xi,yi = np.nonzero(np.sqrt((dx-xcenter)**2 + (dy-ycenter)**2)>sigma)
    stim_arr[xi,yi,5:10] = 1
    
    # make a circular mask appear for the next 5 frames
    xi,yi = np.nonzero(np.sqrt((dx-xcenter)**2 + (dy-ycenter)**2)<sigma)
    stim_arr[xi,yi,10::] = 1
    
    
    # make the response prediction
    response = generate_og_timeseries(dx,
                                      dy,
                                      stim_arr,
                                      xcenter,
                                      ycenter,
                                      sigma)
                                
    # make sure the RSS is 0
    step = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    rval = np.corrcoef(response, step)[0, 1]
    
    # The voxel responds when the stimulus covers its PRF, so it perfectly
    # correlates with a step function: 
    nt.assert_equal(round(rval, 3), 1)

# def test_binner():
#     
#     signal = np.ones(10)
#     times = np.linspace(0,1,10)
#     bins = np.arange(-0.5,1.5,0.5)
#     
#     binned_signal = spin.binner(signal, times, bins)
#     
#     nt.assert_true(len(binned_signal), len(bins)-2)
#     nt.assert_true(np.all(binned_signal==[5,5]))
#     
