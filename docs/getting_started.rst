Getting started
=================

To get started using :mod:`popeye`, you will need to organize your data in the
following manner:

  - You put the lime in the coconut.
  - Drink it all up.

To run the analysis::

    from popeye.base import PopulationModel, PopulationFit
    from popeye.stimulus import Stimulus

    # stimulus features
    pixels_across = 800 
    pixels_down = 600
    viewing_distance = 38
    screen_width = 25
    thetas = np.arange(0,360,45)
    num_steps = 30
    ecc = 10
    tr_length = 1.0
    
    # create the sweeping bar stimulus in memory
    bar = simulate_bar_stimulus(pixels_across, pixels_down, viewing_distance,
                                                  screen_width, thetas, num_steps, ecc)
    
    # instantiate an instance of the Stimulus class
    stimulus = Stimulus(bar, viewing_distance, screen_width, 0.05, 0, 0)
    
    # set up bounds for the grid search
    bounds = ((-10, 10), (-10, 10), (0.25, 5.25), (-5, 5))
    
    # initialize the gaussian model
    prf_model = prf.GaussianModel(stimulus)
    
   
    # generate the modeled BOLD response
    response = prf.MakeFastPrediction(stimulus.deg_x, stimulus.deg_y,
                                      stimulus.stim_arr, estimate[0], estimate[1], estimate[2])

    hrf = prf.double_gamma_hrf(estimate[3], 1)
    response = utils.zscore(np.convolve(response, hrf)[0:len(response)])
    
    # fit the response
    prf_fit = prf.GaussianFit(prf_model, response, bounds, tr_length)
