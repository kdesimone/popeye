def double_gamma_hrf(delay):
	"""
	The HRF that is convolved with the stimulus time-series. The delay shifts 
	the peak and under-shoot by a variable number of seconds.
	
	"""
	
	from scipy.special import gamma
	import numpy as np
	
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
	The objective function that produces the minimizeable error between
	the predicted and actual BOLD time-series.
	
	The model parameters include the (x,y,sigma) as well as the HRF delay.
	
	"""
	
	import numpy as np
	from MakeFastPrediction import MakeFastPrediction
	
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
	The adaptive brute-force grid-search sparsely samples the parameter space and uses a down-sampled
	version of the stimulus and cooridnate matrices.  This is intended to yield an initial, ball-park 
	solution that is then fed into the more finely-tuned fmin_powell in the compute_prf_estimate method 
	below.
	
	"""
	
	from scipy.optimize import brute, fmin
	
	# set initial pass to 1
	passNum = 1
	
	# make as many passes as the user specifies in rounds
	while passNum <= rounds:
		
		# get a fit estimate by sparsely sampling the 4-parameter space
		phat = brute(error_function,
			     args=(tsActual,degX,degY,stimArray),
			     ranges=Bounds,
			     Ns=5,
			     finish=fmin)
		
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
	The main pRF estimation method.
	
	TODO:  modify this method in conjuction with utilities.multiprocess_prf_estimates
	so that it takes a jobs queue object and processes a single voxel's time-series.
	
	"""
	
	
	import time
	import shutil
	import numpy as np
	from scipy.optimize import fmin_powell
	from scipy.stats import linregress
	from guppy import hpy
	from MakeFastPrediction import MakeFastPrediction
	
	
	# bounds for the adaptive brute-force grid-search
	Bounds = ((-10, 10), (-10, 10), (0.25, 5.25),(-1,1))
	
	# grab voxel indices
	xi,yi,zi = voxels[:]
	
	# counter for reporting progress
	numVoxel = 0
	
	# initialize a list in which to store the results
	results = []
	
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
					# print the details of the estimation for this voxel
					print("VOXEL=(%d,%d,%d,%d/%d),TIME=%.03f,X=%.03f,Y=%.03f,S=%.03f,D=%.03f,ERROR=%d,COV=%.03f" 
					      %(xvoxel,
					        yvoxel,
  						zvoxel,
						numVoxel+1,
						len(xi),
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
				numVoxel += 1
				
	# add results to the queue
	results_q.put(results)
	
	return results_q
