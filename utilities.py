def resample_stimulus(stimArray,scaleFactor):
	"""
	Resamples the stimulus array.  Used for ball-parking 
	the initial guess to the gradient-descent error minimization.
	
	Returns:
	resampledStim	stimulus array resampled according to the user-specified scaleFactor
	
	"""
	
	import numpy as np
	from scipy.misc import imresize
	
	dims = np.shape(stimArray)
	resampledStim = np.zeros((dims[0]*scaleFactor,dims[1]*scaleFactor,dims[2]))
	for tp in range(dims[2]):
		resampledStim[:,:,tp] = imresize(stimArray[:,:,tp],scaleFactor)
	return resampledStim.astype('short')

def generate_coordinate_matrices(pixelsAcross,pixelsDown,pixelsPerDegree,scaleFactor):
	"""
	Creates stimulus-referred coordinate matrices for modeling 
	receptive fields in the visual space.
	
	Returns:
	degX	the coordinate matrix along the horizontal dimension of the display.
	degY	the coordinate matrix along the vertical dimension of the display.
	
	"""
	
	import numpy as np
	
	[X,Y] = np.meshgrid(np.arange(pixelsAcross*scaleFactor),np.arange(pixelsDown*scaleFactor))
	degX = (X-np.shape(X)[1]/2)/(pixelsPerDegree*scaleFactor).astype('double')
	degY = (Y-np.shape(Y)[0]/2)/(pixelsPerDegree*scaleFactor).astype('double')
	
	return degX,degY

def recast_results_queue_output(output,metaData):
	"""
	Recasts the output of the pRF estimation into 9D nifti_gz volumes.  Two files are generated, 
	containing the pRF estimatation results in Cartesian and polar coordinates.
	
	The resulting files contain:
	0 x / polar angle
	1 y / eccentricity
	2 sigma
	3 HRF delay
	4 slope of the model-actual fit
	5 intercept of the model-actual fit
	6 correlation of the model-actual fit
	7 two-tailed p-value of the model-actual fit
	8 standard error of the model-actual fit
	
	Returns:
	cartesFileName	the absolute file path of the pRF estimates in Cartesian coordinates
	polarFileName	the absolute path path of the pRF estimates in polar coordinates
	
	"""
	
	import sys, os, time
	import nibabel
	import numpy as np
	
	# load the gridParent
	gridParent = nibabel.load(metaData['maskPath'])
	dims = list(gridParent.get_shape())
	dims.append(9)
	pRF_polar = np.zeros(dims)
	pRF_cartes = np.zeros(dims)
	
	# extract the pRF model estimates from the results queue output
	for job in output:
		for voxel in job:
			xi,yi,zi = voxel[0:3]
			x,y,s,d = voxel[3:7]
			stats = voxel[7]
			slope,intercept,rval,pval,stderr = stats[:]
			pRF_cartes[xi,yi,zi,:] = x,y,s,d,slope,intercept,rval,pval,stderr
			pRF_polar[xi,yi,zi,:] = np.mod(np.arctan2(x,y),2*np.pi),np.sqrt(x**2+y**2),s,d,slope,intercept,rval,pval,stderr
	
	# get header information from the gridParent and update for the pRF volume
	aff = gridParent.get_affine()
	hdr = gridParent.get_header()
	hdr.set_data_shape(dims)
	voxelDims = list(hdr.get_zooms())
	voxelDims[-1] = 9
	hdr.set_zooms(voxelDims)
	
	# write the files
	now = time.strftime('%Y%m%d_%H%M%S')
	
	nif = nibabel.Nifti1Image(pRF_polar,aff)
	nif.set_data_dtype('float32')
	polarFileName = '%s/%s_polar_%s.nii.gz' %(metaData['outputPath'],metaData['baseFileName'],now)
	nibabel.save(nif,polarFileName)
	
	nif = nibabel.Nifti1Image(pRF_cartes,aff)
	nif.set_data_dtype('float32')
	cartesFileName = '%s/%s_cartes_%s.nii.gz' %(metaData['outputPath'],metaData['baseFileName'],now)
	nibabel.save(nif,cartesFileName)
	
	return polarFileName,cartesFileName

def multiprocess_prf_estimates(stimData,funcData,metaData):
	"""
	Uses multiprocessing.Process and multiprocessing.Queue to submit pRF estimation jobs to the 
	user-specified number of CPUs gleaned from metaData.  The results are stored in a Queue
	that are returned at the conclusion of each CPUs task.
	
	TODO:  Implement a job queue in addition to the results queue.  This would maximize the resources
	of each CPU and process.
	
	"""
	
	from multiprocessing import Process, Queue
	import numpy as np
	from fitting import compute_prf_estimate
	
	# figure out how many voxels are in the mask & the number of jobs we have allocated
	[xi,yi,zi] = metaData['voxels']
	jobs = metaData['jobs']
	
	# Set up the voxel lists for each job
	voxelLists = []
	cutOffs = [int(np.floor(i)) for i in np.linspace(0,len(xi),jobs+1)]
	for i in range(len(cutOffs)-1):
		l = range(cutOffs[i],cutOffs[i+1])
		voxelLists.append(l)
		
	# initialize Queues for managing the outputs of the jobs
	results_q = Queue()
	
	# start the jobs
	procs = []
	for j in range(jobs):
		voxels = [xi[voxelLists[j]],yi[voxelLists[j]],zi[voxelLists[j]]]
		p = Process(target=compute_prf_estimate,args=(voxels,stimData,funcData,results_q))
		procs.append(p)
		p.start()
		
	# gather the outputs from the queue
	output = []
	for i in range(len(procs)):
		output.append(results_q.get())
		
	# close the jobs
	for p in procs:
		p.join()
		
	return output
	