# -*- coding: utf-8 -*-

"""
Factory function for simulating a measurement of fluorescence decay.
"""

import numpy as np
from .models import AddConstant, Linearize, Convolve, Exponential

def make_simulation(user_kwargs, time, instrument_response, peak_cnts=None, verbose=True):
    """Simulates measured fluorescence decay distorted by the instrument response.

    Simulates fluorescence decay distorted by the instrument response with Poisson-distributed observed values using Monte Carlo method

    Parameters
    ----------
    user_kwargs : dict
        Dict with user provided info about model.
    time : ndarray
        1D ndarray with times (x-scale of data).
    instrument_response : ndarray
        1D ndarray with instrument_response functions 
        (for convolution with calculated model).
    peak_cnts : int, optional
        Counts in maximum (by default max of `instrument_response`).
    verbose : bool
        Print simulation progress.

    Returns
    -------
    ndarray
    """
    independent_var = dict(
                time=time, 
                instrument_response=instrument_response
                )
    # make model
    ModelClass = Convolve(
        AddConstant(Linearize(Exponential(**user_kwargs))),
        convolution_method='monte_carlo',
        peak_cnts=peak_cnts,
        verbose=verbose
        )
    model = ModelClass.make_model(**independent_var)
    return model.eval(**ModelClass.make_parameters())
