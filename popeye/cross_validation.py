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


def kfold_xval(model, fit, folds, fit_args):
    
    # extract the data
    data = fit.data
    
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
    # order = np.random.permutation(data.shape[-1])
    order = np.arange(data.shape[-1])
    
    # Grab the full-sized stimulus arrays
    stim_arr = model.stimulus.stim_arr.copy()
    stim_arr_coarse = model.stimulus.stim_arr_coarse.copy()
        
    # Do the thing
    for k in range(folds):
        
        # select the timepoints for this fold
        fold_mask = np.ones(data.shape[-1], dtype=bool)
        fold_idx = order[k*n_in_fold:(k+1)*n_in_fold]
        fold_mask[fold_idx] = False
        
        # grab the left-in data and stimulus
        left_in_data = data[...,fold_mask]
        left_in_stim_arr = stim_arr[...,fold_mask]
        left_in_stim_arr_coarse = stim_arr_coarse[...,fold_mask]
        
        # initialize a left-in Model to estimate the model parameters
        left_in_stimulus = deepcopy(model.stimulus)
        left_in_stimulus.stim_arr = left_in_stim_arr.copy()
        left_in_stimulus.stim_arr_coarse = left_in_stim_arr_coarse.copy()
        left_in_model = model.__class__(left_in_stimulus)
        
        # grab the left-out data and stimulus
        left_out_data = data[...,~fold_mask]
        left_out_stim_arr = stim_arr[...,~fold_mask]
        left_out_stim_arr_coarse = stim_arr_coarse[...,~fold_mask]
        
        # initialize a left-in Model to estimate the model parameters
        left_out_stimulus = deepcopy(model.stimulus)
        left_out_stimulus.stim_arr = left_out_stim_arr.copy()
        left_out_stimulus.stim_arr_coarse = left_out_stim_arr_coarse.copy()
        left_out_model = model.__class__(left_out_stimulus)
        
        # initialize the left-in fit object
        ensemble = []
        ensemble.append(left_in_data)
        ensemble.append(left_in_model)
        ensemble.extend(fit_args)
        left_in_fit = fit.__class__(*ensemble)
        
        # initialize the left-in fit object
        ensemble = []
        ensemble.append(left_out_data)
        ensemble.append(left_out_model)
        ensemble.extend(fit_args)
        left_out_fit = fit.__class__(*ensemble)
        
        # fit the left-in fit object
        left_in_fit.prediction;
        
        # transfer the parameter estimates from the left-in fit to the left-out fit
        left_out_fit.estimate = left_in_fit.estimate
        
        # fill the run-wide prediction values with the left-out predictions ...
        prediction[..., fold_idx] = left_out_fit.prediction
    
    return prediction
