# -*- coding: utf-8 -*-

"""Provide Fitter object and factory functions to make it.

Fitter is a composition of objects need to evaluate a model: Model, Parameters,
independent variables. It also stores other objects need to perform a fit:
dependent variable and Statistic.

"""

import re
import itertools
import numpy as np
from lmfit import Parameters, report_fit
from .statistics import CStatistic, ChiSquareStatistic, ChiSquareStatisticVariableProjection
from .models import GlobalModel, AddConstant, Linearize, Convolve, Exponential
from .utils import process_amplitudes, pretty_print_dictionary

def iterative_least_squares(fitter_class, iterations):
    """Least squares in a loop.

    Perform least squares minimization in iterations,
    with initial parameters values from previous iteration
    and variance approximation according to Pearson
    (based on fitted model).

    Parameters
    ----------
    fitter_class : fluo.Fitter
    iterations : int

    Returns
    -------
    fits : list of lmfit.ModelResult
        List with fit from every iteration.

    """
    print("0-th iteration. Initial fit.")
    ini_fit = fitter_class.fit(report=True)
    i_params = ini_fit.params
    fits = [ini_fit]
    for i in range(iterations):
        print()
        print(
            "{}-th iteration".format(i+1)
            )
        fitter_class.statistic = ChiSquareStatistic(
            variance_approximation='Pearson'
            )
        fitter_class.parameters = i_params
        i_fit = fitter_class.fit(report=True)
        i_params = i_fit.params
        fits.append(i_fit)
    return fits


def make_global_lifetime_fitter(local_user_kwargs, local_times, local_decays,
    local_instrument_responses=None, fit_statistic='c_statistic', shared=None):
    """Make a Fitter for simultaneous (global) fitting to multiple measurements.

    Parameters
    ----------
    local_user_kwargs : list of dict
        List of dict with user provided info about model and fit.
    local_times : list of ndarray
        List of 1D ndarray with times (x-scale of data).
    local_decays : list of ndarray
        List of 1D ndarray with fluorescence decays (y-scale of data).
    local_instrument_responses : list of ndarray, optional
        List of 1D ndarray with instrument_response functions
        (for convolution with calculated model).
    fit_statistic : str, optional
        Statisic used in fitting minimization.
        Accepts the following str: 'c_statistic', 'chi_square_statistic',
        'chi_square_statistic_variable_projection'.
    shared : list of str, optional
        List of parameters names shared between fitted measurements.

    Raises
    ------
    ValueError
        If invalid `fit_statistic` is provided.

    Returns
    -------
    fluo.Fitter

    """
    # making local_instrument_responses good for zipping if None
    if local_instrument_responses is None:
        local_instrument_responses = iter([])
    local_zipped = itertools.zip_longest(
        local_user_kwargs,
        local_times,
        local_decays,
        local_instrument_responses
        )
    # making local Fitter objects
    local_fitter_classes = [
        make_lifetime_fitter(*args, fit_statistic) for args in local_zipped
    ]
    # gluing it together
    global_pre_fitter_cls = GlobalModel(
        fitter_classes=local_fitter_classes,
        shared=shared)
    independent_var = dict(
        independent_var=global_pre_fitter_cls.local_independent_var
        )
    dependent_var = np.concatenate(global_pre_fitter_cls.local_dependent_var)
    statistic_cls = global_pre_fitter_cls.statistic

    return Fitter(
            model_class=global_pre_fitter_cls,
            independent_var=independent_var,
            dependent_var=dependent_var,
            statistic=statistic_cls
    )


def make_lifetime_fitter(user_kwargs, time, decay, instrument_response=None,
    fit_statistic='c_statistic'):
    """Make a Fitter for fitting to a single measurement.

    Parameters
    ----------
    user_kwargs : dict
        Dict with user provided info about model and fit.
    time : ndarray
        1D ndarray with times (x-scale of data).
    decay : ndarray
        1D ndarray with fluorescence decays (y-scale of data).
    instrument_response : ndarray, optional
        1D ndarray with instrument_response functions
        (for convolution with calculated model).
    fit_statistic : str, optional
        Statisic used in fitting minimization.
        Accepts the following str: 'c_statistic', 'chi_square_statistic',
        'chi_square_statistic_variable_projection'.

    Raises
    ------
    ValueError
        If invalid `fit_statistic` is provided.

    Returns
    -------
    fluo.Fitter

    """
    # making allowed statistic class
    allowed_fit_statistics = dict(
        c_statistic=CStatistic(),
        chi_square_statistic=ChiSquareStatistic(),
        chi_square_statistic_variable_projection=\
        ChiSquareStatisticVariableProjection()
        )
    try:
        statistic_cls = allowed_fit_statistics[fit_statistic]
    except KeyError:
        allowed_fit_statistics_names = ", ".join(
            list(
                allowed_fit_statistics.keys()
                )
                )
        raise ValueError(
            "fit_statistic: '{0}' not implemented. Available fit_statistic: {1}".format(
                fit_statistic,
                allowed_fit_statistics_names)
                )
    # pre-process fit range
    user_kwargs = user_kwargs.copy()
    fit_start = user_kwargs.pop('fit_start')
    fit_stop = user_kwargs.pop('fit_stop')
    if fit_start is None:
        fit_start = 0
    if fit_stop is None:
        fit_stop = np.inf
    range_mask = (time >= fit_start) & (time <= fit_stop)
    decay = decay[range_mask].astype(float)
    time = time[range_mask].astype(float)
    independent_var = dict(time=time)
    # making exponential_cls
    exponential_cls = Exponential(**user_kwargs)
    # checking if tail-fit or convolution-fti
    if instrument_response is not None:
        exponential_cls = Convolve(exponential_cls)
        independent_var['instrument_response'] = \
        instrument_response[range_mask].astype(float)
    # checking if VarPro
    if isinstance(
        statistic_cls,
        ChiSquareStatisticVariableProjection):
        return Fitter(
            model_class=exponential_cls,
            independent_var=independent_var,
            dependent_var=decay,
            statistic=statistic_cls
            )

    return Fitter(
        model_class=AddConstant(Linearize(exponential_cls)),
        independent_var=independent_var,
        dependent_var=decay,
        statistic=statistic_cls
    )


class Fitter():
    """Fitter object.

    Fitter is a composition of objects need to evaluate a model. It wraps
    `fluo.Model`, makes `lmfit.Parameters`, containes independent variables.
    It also stores other objects need to perform a fit: dependent variable and
    `fluo.Statistic`.

    Parameters
    ----------
    model_class : fluo.Model
        Model class inheriting from fluo.Model
    independent_var : dict
        Independent variables for a model evaluation. Dict with names of
        independent variables encoded by keys (str) and values as ndarrays.
    dependent_var : ndarray
        1D ndarray with dependent variable for fitting.
    statistic : fluo.Statistic
        Statistic class for fitting.

    Attributes
    ----------
    parameters : lmfit.Parameters
        lmfit.Parameters for model evaluation.
    model : fluo.GenericModel

    Methods
    -------
    fit : lmfit.ModelResult

    """

    def __init__(self, model_class, independent_var, dependent_var, statistic):
        """Initialize Fitter object.

        Parameters
        ----------
        model_class : fluo.Model
            Model class inheriting from fluo.Model
        independent_var : dict
            Independent variables for a model evaluation. Dict with names of
            independent variables encoded by keys (str) and values as ndarrays.
        dependent_var : ndarray
            1D ndarray with dependent variable for fitting.
        statistic : fluo.Statistic
            Statistic class for fitting.

        """
        self.model_class = model_class
        self.independent_var = independent_var
        self.dependent_var = dependent_var
        self.statistic = statistic
        self.name = ''
        self.parameters = model_class.make_parameters()
        self.model = model_class.make_model(**independent_var)

    def fit(self, report=True):
        """Perform a fit.

        Parameters
        ----------
        report : bool, optional
            Report fit (True by default).

        Returns
        -------
        lmfit.ModelResult

        """
        self.name = '{} fitted using {}'.format(
            self.model.name,
            self.statistic.name)
        result = self.model.generic_fit(
            data=self.dependent_var,
            statistic=self.statistic,
            params=self.parameters
            )
        if report:
            print()
            print('Report: {}'.format(self.name))
            report_fit(result)
            print('[[Scaled Variables]]')
            try:
                pattern_file = re.compile('file[0-9]+')
                found_pattern = set(
                    pattern_file.search(key).group() for key in result.best_values
                )
                splited_params = []
                for file in found_pattern:
                    splited_params.append(
                        {
                        key:result.best_values[key] for key in result.best_values if key.endswith(file)
                    }
                    )
            except AttributeError:
                splited_params = [result.best_values]
            for params in splited_params:
                pretty_print_dictionary(process_amplitudes(params))
        return result

    def __getattr__(self, attr):
        """Return attribute from `model_class`."""
        return getattr(self.model_class, attr)


def autocorrelation(residuals):
    """Calculate residuals autocorrelation.

    Residuals autocorrelation is a correlation between residuals in i-th and
    (i+j)-th channels.

    Parameters
    ----------
    residuals : ndarray

    Returns
    -------
    ndarray

    """
    residuals_full = residuals
    residuals = residuals[~np.isnan(residuals)]
    residuals_len = len(residuals)
    inv_n = 1. / residuals_len
    # normalization weight in autocorrelation function
    denominator = inv_n * np.sum(np.square(residuals))
    residuals = list(residuals)
    autocorr_len = residuals_len // 2
    numerator = []
    for j in range(autocorr_len):
        k = residuals_len - j
        numerator_sum = 0.0
        for i in range(k):
            numerator_sum += residuals[i] * residuals[i + j]
        numerator.append(numerator_sum / k)
    numerator = np.array(numerator)
    autocorr = numerator / denominator
    over_range = np.array([np.nan] * len(residuals_full))
    autocorr = np.append(autocorr, over_range)
    return autocorr
