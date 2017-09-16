# -*- coding: utf-8 -*-

"""Provide the function simulating a measurement of fluorescence decay."""

from .models import AddConstant, Linearize, Convolve, Exponential

def make_simulation(user_kwargs, time, instrument_response, peak_cnts=None, verbose=True):
    """Simulate fluorescence decay distorted by the instrument response.

    Fluorescence decay distorted by the instrument response is calulated
    using Monte Carlo method.

    Parameters
    ----------
    user_kwargs : dict
        Dict with user provided info about model.
    time : ndarray
        1D ndarray with times (x-scale of data).
    instrument_response : ndarray
        1D ndarray with instrument_response functions (for convolution with
        calculated model).
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
    model_class = Convolve(
        AddConstant(Linearize(Exponential(**user_kwargs))),
        convolution_method='monte_carlo',
        peak_cnts=peak_cnts,
        verbose=verbose
        )
    model = model_class.make_model(**independent_var)
    return model.eval(**model_class.make_parameters())
