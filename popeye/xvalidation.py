"""
Cross-validation analysis of diffusion models
"""
from __future__ import division, print_function, absolute_import
from copy import deepcopy

import numpy as np


def coeff_of_determination(data, model, axis=-1):
    """
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
       The coefficient of determination. This has shape `data.shape[:-1]`


    Notes
    -----

    See: http://en.wikipedia.org/wiki/Coefficient_of_determination

    The coefficient of determination is calculated as:

    .. math::

        R^2 = 100 * (1 - \frac{SSE}{SSD})

    where SSE is the sum of the squared error between the model and the data
    (sum of the squared residuals) and SSD is the sum of the squares of the
    deviations of the data from the mean of the data (variance * N).
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
    
    """
    Perform k-fold cross-validation to generate out-of-sample predictions for
    each measurement.
    
    Parameters
    ----------
    models : list of instances of Model
        A list containing the Model instances to be handed to Fit.  If the length of `models` is
        1, then it is assumed that `data` is composed of either a single run of data or of multiple
        runs with the same, repeated stimulus presented.  
    
    data : ndarray
    
    Fit : the Fit class that will be instantiated with the left-in and left-out datasets.
    
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
    div_by_folds =  np.mod(data.shape[-1], folds)
    
    # Make sure that an equal* number of samples get left out in each fold:
    if div_by_folds!= 0:
        msg = "The number of folds must divide the diffusion-weighted "
        msg += "data equally, but "
        msg = "np.mod(%s, %s) is %s"%(data.shape[-1], folds, div_by_folds)
        raise ValueError(msg)
    
    # number of samples per fold
    n_in_fold = data.shape[-1]/folds
    
    # iniitialize a prediciton. this may not be necessary in popeye
    prediction = np.zeros(data.shape[-1])
    
    # We are going to leave out some randomly chosen samples in each iteration
    order = np.random.permutation(data.shape[-1])
    
    # initilize a list of predictions
    predictions = []
    dat = []
    
    # Do the thing
    for k in range(folds):
        
        # Select the timepoints for this fold
        fold_mask = np.ones(data.shape[-1], dtype=bool)
        fold_idx = order[k*n_in_fold:(k+1)*n_in_fold]
        fold_mask[fold_idx] = False
        
        # Grab the left-in data and concatenate the runs
        left_in_data = np.reshape(data[...,fold_mask], data.shape[0]*2, order='F')
        
        # Grab the left-out data and concatenate the runs
        left_out_data = np.reshape(data[...,~fold_mask], data.shape[0]*2, order='F')
        
        # If there is only 1 model specified, repeat it over the concatenated functionals
        if len(models) == 1:
            
            # Grab the stimulus instance from one of the models
            stimulus = deepcopy(models[0].stimulus)
            
            # Tile it according to the number of runs ...
            stimulus.stim_arr = np.tile(stimulus.stim_arr.copy(), folds)
            stimulus.stim_arr_coarse = np.tile(stimulus.stim_arr_coarse.copy(), folds)
            
            # Create a new Model instance
            model = models[0].__class__(stimulus)
        
        # otherwise, concatenate each of the unique stimuli
        elif len(models) == data.shape[-1]:
            
            # Grab the stimulus instance from one of the models
            stimulus = deepcopy(models[fold_idx[0]].stimulus)
            
            # Grab the first model in the fold
            stim_arr_cat = stimulus.stim_arr
            stim_arr_coarse_cat = stimulus.stim_arr_coarse
            
            # Grab the remaining models in the fold
            for f in fold_idx[1::]:
                
                stim_arr_cat = np.concatenate((stim_arr_cat, models[f].stimulus.stim_arr.copy()), axis=-1)
                stim_arr_coarse_cat = np.concatenate((stim_arr_coarse_cat, models[f].stimulus.stim_arr_coarse.copy()), axis=-1)
            
            # Put the concatenated arrays into the Stimulus instance
            stimulus.stim_arr = stim_arr_cat
            stimulus.stim_arr_coarse = stim_arr_coarse_cat
            
            # Create a new Model instance
            model = models[0].__class__(stimulus)
            
        # initialize the left-in fit object
        ensemble = []
        ensemble.append(left_in_data)
        ensemble.append(model)
        ensemble.extend(fit_args)
        ensemble.extend(fit_kwargs.values())
        left_in_fit = Fit(*ensemble)
                
        # initialize the left-out fit object
        ensemble = []
        ensemble.append(left_out_data)
        ensemble.append(model)
        ensemble.extend(fit_args)
        ensemble.extend(fit_kwargs.values())
        left_out_fit = Fit(*ensemble)
        
        # run the left-in Fit
        left_out_fit.estimate = left_in_fit.estimate
        
        # store the prediction
        predictions.append(left_out_fit.prediction)
        
        # store the left-out data
        dat.append(left_out_fit.data)
    
    return dat, predictions
