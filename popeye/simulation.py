from __future__ import division
import time

import numpy as np
from scipy.ndimage.measurements import standard_deviation
from scipy.optimize import fmin_powell, fmin

from popeye.spinach  import MakeFastRFs
from popeye.spinach import MakeFastRF

def error_function(sigma,old_sigma,xs,ys,degX,degY,voxel_RF):
    if sigma <= 0:
        return np.inf
    if sigma > old_sigma:
        return np.inf
    neural_RF = MakeFastRFs(degX,degY,xs,ys,sigma)
    neural_RF /= np.max(neural_RF)
    error = np.sum((neural_RF-voxel_RF)**2)
    return error

def simulate_neural_sigma(stimData,funcData,metaData,results_q,verbose=True):
    
    # grab voxel indices
    xi,yi,zi = metaData['core_voxels']
    
    # initialize a list in which to store the results
    results = []
    
    # printing niceties
    numVoxels = len(xi)
    voxelCount = 1
    printLength = len(xi)/10
    
    # grab the pRF volume
    pRF = funcData['pRF_cartes']
    pRF_polar = funcData['pRF_polar']
    
    
    # grab the 3D meshgrid for creating spherical mask around a seed voxel
    X,Y,Z = funcData['volume_meshgrid'][:]
    
    # main loop
    for xvoxel,yvoxel,zvoxel in zip(xi,yi,zi):
        
        # get a timestamp
        toc = time.clock()
        
        # grab the pRF estimate for the seed voxel
        x_0 = pRF[xvoxel,yvoxel,zvoxel,0]
        y_0 = pRF[xvoxel,yvoxel,zvoxel,1]
        s_0 = pRF[xvoxel,yvoxel,zvoxel,2]
        d_0 = pRF[xvoxel,yvoxel,zvoxel,3]
        old_sigma = s_0.copy()
        
        # recreate the voxel's pRF
        voxel_RF = MakeFastRF(stimData['degXFine'],stimData['degYFine'],x_0,y_0,s_0)
        voxel_RF /= np.max(voxel_RF)
        voxel_RF[np.isnan(voxel_RF)] = 0
        
        # find all voxels within the neighborhood
        d = np.sqrt((X-xvoxel)**2 + (Y-yvoxel)**2 + (Z-zvoxel)**2)
        mask = np.zeros_like(d)
        mask[d <= 2] = 1
        mask[xvoxel,yvoxel,zvoxel] = 0
        [dx,dy,dz] = np.nonzero((mask==1) & (pRF[:,:,:,6]>0.20))
        
        # compute the mean visuotopic scatter
        meanScatter = np.mean(np.sqrt((pRF[dx,dy,dz,0]-x_0)**2 + (pRF[dx,dy,dz,1]-y_0)**2))/2
        
        # find all the pixels that are within the scatter range of the pRF
        [xpixels,ypixels] = np.nonzero((stimData['degXFine']-x_0)**2+(stimData['degYFine']-y_0)**2< meanScatter**2)
        
        if xpixels.any():
            randPixels = np.random.randint(0,len(xpixels),metaData['neurons'])
            
            # grab the locations from the coordinate matrices
            xs = stimData['degXFine'][xpixels[randPixels],ypixels[randPixels]]
            ys = stimData['degYFine'][xpixels[randPixels],ypixels[randPixels]]
            
            # compute the neural sigma and the difference in size
            sigma_phat = fmin_powell(error_function,s_0,args=(old_sigma,xs,ys,stimData['degXFine'],stimData['degYFine'],voxel_RF),full_output=True,disp=False)
            percentChange = ((sigma_phat[0]-s_0)/s_0)*100
            
            # get a timestamp
            tic = time.clock()
            
            # # print the details of the estimation for this voxel
            if verbose:
                percentDone = (voxelCount/numVoxels)*100
                print("%.02d%%  VOXEL=(%.03d,%.03d,%.03d)  TIME=%.03E  ERROR=%.03E  OLD=%.03f  NEW=%.03f  DIFF=%+.02E%%  SCATTER=%.02E" 
                      %(percentDone,
                        xvoxel,
                        yvoxel,
                        zvoxel,
                        tic-toc,
                        sigma_phat[1],
                        s_0,
                        sigma_phat[0],
                        percentChange,
                        meanScatter))
                    
            # store the results
            results.append((xvoxel,yvoxel,zvoxel,sigma_phat[0],sigma_phat[1],meanScatter,percentChange))
            
        # interate variable
        voxelCount += 1
            
    # add results to the queue
    results_q.put(results)
    
    return results_q
            