#!/usr/bin/python

""" Classes and functions for fitting SpatioTemporal population receptive field models """

from __future__ import division, print_function, absolute_import
import time
import gc
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.stats import linregress
from scipy.signal import fftconvolve
from scipy.integrate import trapz, simps
import nibabel
import statsmodels.api as sm

from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries

def recast_estimation_results(output, grid_parent, polar=False):
    """
    Recasts the output of the prf estimation into two nifti_gz volumes.
    
    Takes `output`, a list of multiprocessing.Queue objects containing the
    output of the prf estimation for each voxel.  The prf estimates are
    expressed in both polar and Cartesian coordinates.  If the default value
    for the `write` parameter is set to False, then the function returns the
    arrays without writing the nifti files to disk.  Otherwise, if `write` is
    True, then the two nifti files are written to disk.
    
    Each voxel contains the following metrics: 
    
        0 x / polar angle
        1 y / eccentricity
        2 sigma
        3 HRF delay
        4 RSS error of the model fit
        5 correlation of the model fit
        
    Parameters
    ----------
    output : list
        A list of PopulationFit objects.
    grid_parent : nibabel object
        A nibabel object to use as the geometric basis for the statmap.  
        The grid_parent (x,y,z) dim and pixdim will be used.
        
    Returns
    ------ 
    cartes_filename : string
        The absolute path of the recasted prf estimation output in Cartesian
        coordinates. 
    plar_filename : string
        The absolute path of the recasted prf estimation output in polar
        coordinates. 
        
    """
    
    
    # load the gridParent
    dims = list(grid_parent.shape)
    dims = dims[0:3]
    dims.append(7)
    
    # initialize the statmaps
    estimates = np.zeros(dims)
    
    # extract the prf model estimates from the results queue output
    for fit in output:
        
        if polar is True:
            estimates[fit.voxel_index] = (fit.theta,
                                          fit.spatial_sigma,
                                          fit.temporal_sigma,
                                          fit.weight,
                                          fit.rsquared,
                                          fit.coefficient,
                                          fit.stderr)
        else:
            estimates[fit.voxel_index] = (fit.theta, 
                                          fit.spatial_sigma,
                                          fit.temporal_sigma,
                                          fit.weight,
                                          fit.rsquared,
                                          fit.coefficient,
                                          fit.stderr)
                                       
                             
    # get header information from the gridParent and update for the prf volume
    aff = grid_parent.get_affine()
    hdr = grid_parent.get_header()
    hdr.set_data_shape(dims)
    
    # recast as nifti
    nifti_estimates = nibabel.Nifti1Image(estimates,aff,header=hdr)
    
    return nifti_estimates

class SpatioTemporalModel(PopulationModel):
    
    """
    A Gaussian population receptive field model class
    
    """
    
    def __init__(self, stimulus, hrf_model):
        
        """
        A spatiotemporal population receptive field model.
        
        Paramaters
        ----------
        
        stimulus : `VisualStimulus` class object
            A class instantiation of the `VisualStimulus` class
            containing a representation of the visual stimulus.
        
        
        """
        
        PopulationModel.__init__(self, stimulus, hrf_model)
    
    
    def generate_ballpark_prediction(self, theta, spatial_sigma, temporal_sigma, weight):
        
        rho = 4
        
        # convert polar to cartesian
        x = np.cos(theta) * 4
        y = np.sin(theta) * 4
        
        # create Gaussian
        spatial_rf = generate_og_receptive_field(x, y, spatial_sigma,
                                                 self.stimulus.deg_x_coarse,
                                                 self.stimulus.deg_y_coarse)
        
        # normalize it by integral
        spatial_rf /= 2 * np.pi * spatial_sigma ** 2
        
        # create mask for speed
        distance = (self.stimulus.deg_x_coarse - x)**2 + (self.stimulus.deg_y_coarse - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (5*spatial_sigma)**2] = 1
        
        # set the coordinate
        t = np.linspace(0, 1, self.stimulus.projector_hz)
        
        # set the center of the responses
        center = t[len(t)/2]
        
        # create sustain response for 1 TR
        p = np.exp(-((t-center)**2)/(2*temporal_sigma**2))
        p = p * 1/(np.sqrt(2*np.pi)*temporal_sigma)
        
        # create transient response for 1 TR
        m = np.insert(np.diff(p),0,0)
        m = m/(simps(np.abs(m),t))
        
        # bound temporal sigma so we don't have the Gaussian
        # running off the end of the coordinate vector
        # there must be way to figure this bound out ahead of time...
        if np.round(p[0],3) != 0 or np.round(m[0],3) != 0:
            return np.inf
        
        # extract the timeseries
        s_resp = generate_rf_timeseries(self.stimulus.stim_arr_coarse, spatial_rf, mask)
        
        # convolve with transient and sustained RF
        p_resp = np.abs(fftconvolve(s_resp,p,'same'))
        m_resp = np.abs(fftconvolve(s_resp,m,'same'))
        
        # take the mean of each TR
        s_ts = np.array([np.mean(s_resp[tp:tp+self.stimulus.projector_hz*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.projector_hz*self.stimulus.tr_length)])
        p_ts = np.array([np.mean(p_resp[tp:tp+self.stimulus.projector_hz*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.projector_hz*self.stimulus.tr_length)])
        m_ts = np.array([np.mean(m_resp[tp:tp+self.stimulus.projector_hz*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.projector_hz*self.stimulus.tr_length)])
        
        # convolve with hrf 
        hrf = self.hrf_model(0, self.stimulus.tr_length)
        sustained_model = fftconvolve(p_ts, hrf, 'same')
        transient_model = fftconvolve(m_ts, hrf, 'same')
        
        # normalize to a range
        sustained_norm = utils.zscore(sustained_model)
        transient_norm = utils.zscore(transient_model)
        
        # mix it together
        model = sustained_norm * weight + transient_norm * (1-weight)
        
        # # scale by beta
        # model *= beta
        # 
        # # offset
        # model += baseline
        
        return model
        
    def generate_prediction(self, theta, spatial_sigma, temporal_sigma, weight):
        
        rho = 4
        
        # convert polar to cartesian
        x = np.cos(theta) * rho
        y = np.sin(theta) * rho
        
        # create Gaussian
        spatial_rf = generate_og_receptive_field(x, y, spatial_sigma,
                                                 self.stimulus.deg_x,
                                                 self.stimulus.deg_y)
        
        # normalize it by integral
        spatial_rf /= 2 * np.pi * spatial_sigma ** 2
        
        # create mask for speed
        distance = (self.stimulus.deg_x - x)**2 + (self.stimulus.deg_y - y)**2
        mask = np.zeros_like(distance, dtype='uint8')
        mask[distance < (5*spatial_sigma)**2] = 1
        
        # set the coordinate
        t = np.linspace(0, 1, self.stimulus.projector_hz)
        
        # set the center of the responses
        center = t[len(t)/2]
        
        # create sustain response for 1 TR
        p = np.exp(-((t-center)**2)/(2*temporal_sigma**2))
        p = p * 1/(np.sqrt(2*np.pi)*temporal_sigma)
        
        # create transient response for 1 TR
        m = np.insert(np.diff(p),0,0)
        m = m/(simps(np.abs(m),t))
        
        # bound temporal sigma so we don't have the Gaussian
        # running off the end of the coordinate vector
        # there must be way to figure this bound out ahead of time...
        if np.round(p[0],3) != 0 or np.round(m[0],3) != 0:
            return np.inf
        
        # extract the timeseries
        s_resp = generate_rf_timeseries(self.stimulus.stim_arr, spatial_rf, mask)
        
        # convolve with transient and sustained RF
        p_resp = np.abs(fftconvolve(s_resp,p,'same') / len(s_resp))
        m_resp = np.abs(fftconvolve(s_resp,m,'same') / len(s_resp))
        
        # take the mean of each TR
        s_ts = np.array([np.mean(s_resp[tp:tp+self.stimulus.projector_hz*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.projector_hz*self.stimulus.tr_length)])
        p_ts = np.array([np.mean(p_resp[tp:tp+self.stimulus.projector_hz*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.projector_hz*self.stimulus.tr_length)])
        m_ts = np.array([np.mean(m_resp[tp:tp+self.stimulus.projector_hz*self.stimulus.tr_length]) for tp in np.arange(0,len(s_resp),self.stimulus.projector_hz*self.stimulus.tr_length)])
        
        # convolve with hrf 
        hrf = self.hrf_model(0, self.stimulus.tr_length)
        sustained_model = fftconvolve(p_ts, hrf, 'same') / len(p_ts)
        transient_model = fftconvolve(m_ts, hrf, 'same') / len(p_ts)
        
        # normalize to a range
        sustained_norm = utils.zscore(sustained_model)
        transient_norm = utils.zscore(transient_model)
        
        # mix it together
        model = sustained_norm * weight + transient_norm * (1-weight)
        
        # # scale by beta
        # model *= beta
        # 
        # # offset
        # model += baseline
        
        return model
    
class SpatioTemporalFit(PopulationFit):
    
    """
    A spatiotemporal population receptive field fit class
    
    """
    
    def __init__(self, model, data, grids, bounds, Ns,
                 voxel_index=(1,2,3), auto_fit=True, verbose=0):
        
        PopulationFit.__init__(self, model, data, grids, bounds, Ns, 
                               voxel_index, auto_fit, verbose)
    
    @auto_attr
    def theta0(self):
        return np.mod(self.ballpark[0],2*np.pi)
        
    @auto_attr
    def spatial_s0(self):
        return self.ballpark[1]
    
    @auto_attr
    def weight0(self):
        return self.ballpark[2]
    
    # @auto_attr
    # def beta0(self):
    #     return self.ballpark[3]
    # 
    # @auto_attr
    # def baseline0(self):
    #     return self.ballpark[4]
    
    @auto_attr
    def theta(self):
        return np.mod(self.estimate[0],2*np.pi)
    
    @auto_attr
    def spatial_sigma(self):
        return self.estimate[1]
    
    @auto_attr
    def weight(self):
        return self.estimate[2]
    
    # @auto_attr
    # def beta(self):
    #     return self.estimate[3]
    # 
    # @auto_attr
    # def baseline(self):
    #     return self.estimate[4]
    
    @auto_attr
    def rho(self):
        return 4
    
    @auto_attr
    def temporal_sigma(self):
        return 0.005
        
    @auto_attr
    def spatial_rf(self):
        return generate_og_receptive_field(self.model.stimulus.deg_x, 
                                           self.model.stimulus.deg_y, 
                                           self.x, self.y, self.spatial_sigma)
    
    @auto_attr
    def spatial_rf_norm(self):
        return self.spatial_rf / (2 * np.pi * spatial_sigma ** 2)
    
    @auto_attr
    def prediction(self):
        return self.model.generate_prediction(*self.estimate)
    
    @auto_attr
    def tr_timescale(self):
        return np.linspace(0,self.tr_length,self.model.stimulus.projector_hz*self.tr_length)
    
    @auto_attr
    def response_center(self):
        return self.tr_timescale[len(self.tr_timescale)/2]
    
    @auto_attr
    def sustained_response(self):
        return np.exp(-((self.tr_timescale-self.response_center)**2)/(2*self.temporal_sigma**2))
    
    @auto_attr
    def sustained_response_norm(self):
        return self.sustained_response/(self.temporal_sigma*np.sqrt(2*np.pi))
    
    @auto_attr
    def transient_response(self):
        return np.append(np.diff(self.sustained_response),0)
    
    @auto_attr
    def transient_response_norm(self):
        return self.transient_response/(simps(np.abs(self.transient_response),self.tr_timescale)/2)
    
    @auto_attr
    def hemodynamic_response(self):
        return utils.double_gamma_hrf(self.hrf_delay, self.tr_length)
    
    @auto_attr
    def msg(self):
        txt = ("VOXEL=(%.03d,%.03d,%.03d)  TIME=%.03d  RSQ=%.02f  THETA=%.02f  SSIGMA=%.02f  TSIGMA=%.02f  WEIGHT=%.02f"
            %(self.voxel_index[0],
              self.voxel_index[1],
              self.voxel_index[2],
              self.finish-self.start,
              self.rsquared,
              self.theta,
              self.spatial_sigma,
              self.temporal_sigma,
              self.weight))
        return txt
    
