#!/usr/bin/env python3
from .statistics import CStatistic, ChiSquareStatistic, ChiSquareStatisticVariableProjection
from .models import Exponential, ConvolvedExponential, Linear, Linearize, GlobalModel
import numpy as np
np.set_printoptions(threshold=np.nan)
from lmfit import report_fit
import matplotlib
from matplotlib import pyplot as plt
import itertools



def iterative_least_squares(FitterClass, iterations):
    print(
        "0-th iteration. Initial fit."
        )
    ini_fit = FitterClass.fit(report=True)
    i_params = ini_fit.params
    fits = [ini_fit]
    for i in range(iterations):
        print()
        print(
            "{}-th iteration".format(i+1)
            )
        FitterClass.statistic = ChiSquareStatistic(variance_approximation='Pearson')
        FitterClass.parameters = i_params
        i_fit = FitterClass.fit(report=True)
        i_params = i_fit.params
        fits.append(i_fit)
    return fits

def make_global_lifetime_fitter(local_user_kwargs, local_times, local_decays, local_instrument_responses=None, fit_statistic='c_statistic', shared=None):
    if local_instrument_responses is None:
        local_instrument_responses = iter([])
    local_zipped = itertools.zip_longest(local_user_kwargs, local_times, local_decays, local_instrument_responses)
    local_fitter_classes = [
        make_lifetime_fitter(user_kwargs, time, decay, instrument_response, fit_statistic=fit_statistic) for (user_kwargs, time, decay, instrument_response) in local_zipped
    ]

    global_pre_fitter_cls = GlobalModel(
        FitterClasses=local_fitter_classes, 
        shared=shared)
    independent_var = dict(
        independent_var = global_pre_fitter_cls.local_independent_var
        )
    dependent_var = np.concatenate(global_pre_fitter_cls.local_dependent_var)
    statistic_cls = global_pre_fitter_cls.statistic

    return Fitter(
            ModelClass=global_pre_fitter_cls, 
            independent_var=independent_var, 
            dependent_var=dependent_var,
            statistic=statistic_cls
    )


def make_lifetime_fitter(user_kwargs, time, decay, instrument_response=None, fit_statistic='c_statistic', fit_kwargs=None):
    
    allowed_fit_statistics = dict(
    c_statistic = CStatistic(),
    chi_square_statistic = ChiSquareStatistic(),
    chi_square_statistic_variable_projection = ChiSquareStatisticVariableProjection()
    )

    try:
        statistic_cls = allowed_fit_statistics[fit_statistic]
    except KeyError as err:
        allowed_fit_statistics_names = ", ".join(list(allowed_fit_statistics.keys()))
        print(
            "fit_statistic: '{0}' not implemented. Available fit_statistic: {1}".format(
                fit_statistic, 
                allowed_fit_statistics_names)
        )
        raise err

    if instrument_response is None:
        exponential_cls = Exponential(user_kwargs)
        independent_var = dict(
            time=time
        )
    else:
        exponential_cls = ConvolvedExponential(user_kwargs)
        independent_var = dict(
            time=time, 
            instrument_response=instrument_response
            )

    # pre-process fit range
    fit_start, fit_stop = user_kwargs['--start'], user_kwargs['--stop']
    if fit_start is None:
        fit_start = 0
    if fit_stop is None:
        fit_stop = np.inf
    range_mask = (time >= fit_start) & (time <= fit_stop)
    decay = decay[range_mask].astype(float)
    for key, var in independent_var.items():
        independent_var[key] = var[range_mask].astype(float) 

    if isinstance(
        statistic_cls, 
        ChiSquareStatisticVariableProjection):  
        # pre_fitter_cls = Linear(user_kwargs)
        #     independent_var = dict(
        #         independent_var = 
        #     )
        #     statistic_cls = ChiSquareStatistic()      
        return Fitter(
            ModelClass=exponential_cls,
            independent_var=independent_var, 
            dependent_var=decay,
            statistic=statistic_cls
            )
    else:
        return Fitter(
            ModelClass=Linearize(exponential_cls),
            independent_var=independent_var, 
            dependent_var=decay,
            statistic=statistic_cls
        )


class Fitter(): 
    '''

    '''

    def __init__(self, ModelClass, independent_var, dependent_var, statistic):
        self.ModelClass = ModelClass
        self.independent_var = independent_var
        self.dependent_var = dependent_var
        self.statistic = statistic 
        self.parameters = ModelClass.make_parameters()
        self.model = ModelClass.make_model(**independent_var)
                
    def fit(self, report=True):
        self.name = '{} fitted using {}'.format(
            self.model.name,
            self.statistic.name)
        result = self.model.generic_fit(
                    data=self.dependent_var,
                    statistic=self.statistic,
                    params=self.parameters
                    )
        if report:
            print('Report: {}'.format(self.name))
            report_fit(result)
        return result

    @staticmethod
    def autocorrelation(residuals):
        """Calculates correlation between residuals in i-th and (i+j)-th channels.

        Parameters
        ----------
        residuals : ndarray

        Returns
        -------
        ndarray
        """

        residuals_full = residuals
        residuals = residuals[~np.isnan(residuals)]
        n = len(residuals)
        inv_n = 1. / n
        denominator = inv_n * np.sum(np.square(
            residuals))  # normalization weight in autocorrelation function
        residuals = list(residuals)
        m = n // 2
        numerator = []
        for j in range(m):
            k = n - j
            numerator_sum = 0.0
            for i in range(k):
                numerator_sum += residuals[i] * residuals[i + j]
            numerator.append(numerator_sum / k)
        numerator = np.array(numerator)
        autocorr = numerator / denominator
        over_range = np.array([np.nan] * len(residuals_full))
        autocorr = np.append(autocorr, over_range)
        return autocorr
