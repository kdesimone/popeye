import numpy as np

import popeye
import popeye.estimation as pest
from popeye.spinach import MakeFastRF, MakeFastPrediction

def test_vox_prf():
    """
    Test the 
    """
    xpixels = 100 # simulated screen width
    ypixels = 100 # simulated screen height
    ppd = 1 # simulated visual angle
    scaleFactor = 1.0 # simulated stimulus resampling rate
    
    xcenter = 0 # x coordinate of the pRF center
    ycenter = 0 # y coordinate of the pRF center
    sigma = 10 # width of the pRF
    
    # generate the visuotopic coordinates
    dx,dy = popeye.utilities.generate_coordinate_matrices(xpixels,
                                                          ypixels,
                                                          ppd,
                                                          scaleFactor)
    
    timeseries_length = 15 # number of frames to simulate our stimulus array
    
    # initialize the stimulus array
    stimulus_array = np.zeros((xpixels, ypixels, timeseries_length)).astype(
                                                                     'short')
    
    # make a circular mask appear for the first 5 frames
    xi,yi = np.nonzero(np.sqrt((dx-xcenter)**2 + (dy-ycenter)**2)<sigma)
    stimulus_array[xi,yi,0:5] = 1
    
    # make an annulus appear for the next 5 frames
    xi,yi = np.nonzero(np.sqrt((dx-xcenter)**2 + (dy-ycenter)**2)>sigma)
    stimulus_array[xi,yi,5:10] = 1
    
    # make a circular mask appear for the next 5 frames
    xi,yi = np.nonzero(np.sqrt((dx-xcenter)**2 + (dy-ycenter)**2)<sigma)
    stimulus_array[xi,yi,10::] = 1
    
    
    # make the response prediction
    response = MakeFastPrediction(dx,
                                  dy,
                                  stimulus_array,
                                  xcenter,
                                  ycenter,
                                  np.short(sigma))


    # We'll reuse the same grids for fine and coarse resolutions:
    pest.voxel_prf(response, dx, dy, dx, dy, stimulus_array,
                stimulus_array, bounds=([-50, 50], [-50, 50], [1,10], [1, 10]))
