"""
Cross-validation analysis of population receptive field models
"""

from __future__ import division, print_function, absolute_import
from copy import deepcopy
import time
import numpy as np
import nibabel

def recast_xval_results(output, grid_parent, folds):
    
    # set dims for model+data
    dims = list(grid_parent.shape)
    dims.append(5)
    
    # initialize the cod array
    cod = np.zeros(dims)
    
    # extract the prf model estimates from the results queue output
    for o in output:
        xvoxel = o[0].voxel_index[0]
        yvoxel = o[0].voxel_index[1]
        zvoxel = o[0].voxel_index[2]
        cod[xvoxel,yvoxel,zvoxel,0] = np.mean((o[0].cod,o[1].cod))
        cod[xvoxel,yvoxel,zvoxel,1] = np.mean((o[0].rsquared,o[1].rsquared))
        cod[xvoxel,yvoxel,zvoxel,2] = np.mean((o[0].coefficient,o[1].coefficient))
        cod[xvoxel,yvoxel,zvoxel,3] = np.mean((o[0].mse,o[1].mse))
        cod[xvoxel,yvoxel,zvoxel,3] = np.mean((o[0].stderr,o[1].stderr))
        
    # get header information from the gridParent and update for the prf volume
    aff = grid_parent.get_affine()
    hdr = grid_parent.get_header()
    
    # recast as nifti
    nif = nibabel.Nifti1Image(cod,aff,header=hdr)
    nif.set_data_dtype('float32')
    
    return nif

def coeff_of_determination(data, model, axis=-1):
    
    r"""
    Calculate the coefficient of determination for a model prediction, relative
    to data.
    
    Parameters
    ----------
    data : ndarray
        The data
    model : ndarray
        The predictions of a model for this data. Same shape as the data.
    axis: int, optional
        The axis along which different samples are laid out (default: -1).
        
    Returns
    -------
    COD : ndarray
       The coefficient of determination.
    
    """
    
    residuals = data - model
    ss_err = np.sum(residuals ** 2, axis=axis)
    
    demeaned_data = data - np.mean(data, axis=axis)[..., np.newaxis]
    ss_tot = np.sum(demeaned_data **2, axis=axis)
    
    # Don't divide by 0:
    if np.all(ss_tot==0.0):
        return np.nan
        
    return 100 * (1 - (ss_err/ss_tot))

def kfold_xval(models, data, Fit, folds, fit_args, fit_kwargs):
    
    r"""
    Perform k-fold cross-validation to generate out-of-sample predictions for
    each measurement.
    
    Parameters
    ----------
    models : list of instances of Model
        A list containing the Model instances to be handed to Fit.  If the length of `models` is
        1, then it is assumed that `data` is composed of either a single run of data or of multiple
        runs with the same, repeated stimulus presented.  
    
    data : ndarray
        An m x n array representing a single voxel time-series, where m is the number of
        time-points and n is the number of runs
    
    Fit : Fit class object instance 
        The Fit class that will be instantiated with the left-in and left-out datasets.
    
    folds : int
        The number of divisions to apply to the data
        
    fit_args :
        Additional arguments to the model initialization
        
    fit_kwargs :
        Additional key-word arguments to the model initialization
        
    Notes
    -----
    This function assumes that a prediction API is implemented in the Fit
    class from which a prediction is conducted. That is, the Fit object that gets
    generated upon fitting the model needs to have a `prediction` method, which
    receives a functional time-series and a Model class instance as input 
    and produces a predicted signal as output.
    
    References
    ----------
    .. [1] Rokem, A., Chan, K.L. Yeatman, J.D., Pestilli, F., Mezer, A.,
       Wandell, B.A., 2014. Evaluating the accuracy of diffusion models at
       multiple b-values with cross-validation. ISMRM 2014.
    """
    
    
    # fold the data
    div_by_folds =  np.mod(data.shape[0], folds)
    
    # Make sure that an equal* number of samples get left out in each fold:
    if div_by_folds!= 0:
        msg = "The number of folds must divide the diffusion-weighted "
        msg += "data equally, but "
        msg = "np.mod(%s, %s) is %s"%(data.shape[0], folds, div_by_folds)
        raise ValueError(msg)
    
    # number of samples per fold
    n_in_fold = data.shape[-1]/folds
    
    # iniitialize a prediciton. this may not be necessary in popeye
    prediction = np.zeros(data.shape[-1])
    
    # We are going to leave out some randomly chosen samples in each iteration
    order = np.random.permutation(data.shape[0])
    order = np.reshape(order,(folds,len(order)/folds))
    
    # initilize a list of predictions
    fits = []
    
    # Do the thing
    for k in range(folds):
        
        # Select the timepoints for this fold
        fold_mask = np.zeros(data.shape[0], dtype=bool)
        fold_mask[order[k]] = 1
        
        # Grab left-in data
        left_in_data = data[fold_mask,:]
        
        # Grab left-out data
        left_out_data = data[~fold_mask,:]
        
        # If there is only 1 model specified, compute the mean time-series for the voxel
        if len(models) == 1:
            
            left_in_data = np.mean(left_in_data,axis=0)
            left_out_data = np.mean(left_out_data,axis=0)
            model = models[0]
        
        # initialize the left-in fit object
        ensemble = []
        ensemble.append(model)
        ensemble.append(left_in_data)
        ensemble.extend(fit_args)
        ensemble.extend(fit_kwargs.values())
        left_in_fit = Fit(*ensemble)
                
        # initialize the left-out fit object
        ensemble = []
        ensemble.append(model)
        ensemble.append(left_out_data)
        ensemble.extend(fit_args)
        ensemble.extend(fit_kwargs.values())
        left_out_fit = Fit(*ensemble)
        
        # run the left-in Fit
        left_in_fit.estimate;
        
        # assign that estimate to the left out fit
        left_out_fit.estimate = left_in_fit.estimate
        
        # predict the left-out data
        left_out_fit.prediction;
        left_out_fit.dof = fold_mask.sum()-1
        
        
        # store the left-out data
        fits.append(left_out_fit)
    
    return fits


def parallel_xval(args):
    
    r"""
    This is a convenience function for parallelizing the fitting
    procedure.  Each call is handed a tuple or list containing
    all the necessary inputs for instantiaing a `GaussianFit`
    class object and estimating the model parameters.
    
    
    Paramaters
    ----------
    args : list/tuple
        A list or tuple containing all the necessary inputs for fitting
        the kfold_xval method.
    
    Returns
    -------
    
    fits : `PopulationFit` class objects that have been cross-validated.
        
    
    """ 
    
    # start timestamp
    tic = time.clock()
    
    # unpackage the arguments
    models = args[0]
    data = args[1]
    Fit = args[2]
    folds = args[3]
    fit_args = args[4]
    fit_kwargs = args[5]
    voxel_index = args[6]
    
    # cross-validate
    xval_fits =  kfold_xval(models, data, Fit, folds, fit_args, fit_kwargs)
    
    # compute the cod
    for fit in xval_fits:
        fit.cod = coeff_of_determination(fit.data, fit.prediction)
        fit.voxel_index = voxel_index
        fit.mse = np.std(fit.data)/np.sqrt(fit.dof)
    
    # end timestamp
    toc = time.clock()
    
    # print statement
    xvox = voxel_index[0]
    yvox = voxel_index[1]
    zvox = voxel_index[2]
    duration = toc-tic
    mean_cod = (xval_fits[0].cod+xval_fits[1].cod)/2/100
    mean_mse = (xval_fits[0].mse+xval_fits[1].mse)/2
    mean_bco = (xval_fits[0].coefficient+xval_fits[1].coefficient)/2
    mean_stderr = (xval_fits[0].stderr+xval_fits[1].stderr)/2
    print("Finished voxel %.03d,%.03d,%.03d in %.03d seconds  COD=%.02f BETACOEFF=%.02f  MSE=%.02f  STDERR=%.02f" % (xvox,yvox,zvox,duration,mean_cod,mean_bco,mean_mse,mean_stderr))
    return xval_fits
    
    