#!/usr/bin/env python3
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.linalg
import warnings

class Statistic():
    __metaclass__ = ABCMeta

    def __init__(self, name, optimization_method):
        self.name = name
        self.optimization_method = optimization_method
    
    @abstractmethod
    def objective_func(self, *args):
        raise NotImplementedError("`Statistic.objective_func()` not implemented")


class CStatistic(Statistic):

    _allowed_optimization_methods = ['nelder', 'powell']

    def __init__(self, optimization_method='nelder'):
        if optimization_method not in self._allowed_optimization_methods:
            raise ValueError(
                "Only {} are allowed as a `optimization_method`.".format(
                    ', '.join(['`'+meth+'`' for meth in self._allowed_optimization_methods])))
        super().__init__(
            name='c_statistic',
            optimization_method=optimization_method
        )

    def objective_func(self, model, dependent_var):
        model_copy = np.copy(model)
        dependent_var_copy = np.copy(dependent_var)        
        if np.sum(model < 0):
            warnings.warn("Negative values in model. The C statistic do not comprehend negative values in model. The negative values excluded from fit.")
            model_copy[model < 0] = np.abs(model[model < 0])  
        dependent_var_copy[dependent_var == 0] = np.nan
        result = dependent_var_copy * np.log(model_copy / dependent_var_copy)
        res = -2*(result + (dependent_var_copy - model_copy))
        return np.nan_to_num(res)


class ChiSquareStatistic(Statistic):

    _allowed_variance_approximations = [None, 'Neyman', 'Pearson']

    def __init__(self, optimization_method='leastsq',
                 variance_approximation='Neyman'):
        if variance_approximation not in self._allowed_variance_approximations:
            raise ValueError(
                "Only {} are allowed as a `variance_approximation`.".format(
                    ', '.join(['`'+appr+'`' for appr in self._allowed_variance_approximations])))
        self.variance_approximation = variance_approximation
        super().__init__(
            name='chi_square_statistic with {} variance_approximation'.format(
                self.variance_approximation
                ),
            optimization_method=optimization_method)

    def objective_func(self, model, dependent_var):
        
        if self.variance_approximation is not None:
            model, dependent_var = self._weight_input(model, dependent_var)
        return (model - dependent_var)

    def _weight_input(self, model, dependent_var):
        if self.variance_approximation == 'Neyman':
            weights = self.weight(dependent_var)
        elif self.variance_approximation == 'Pearson':
            weights = self.weight(model)
        return (self.apply_weight(model, weights), 
    self.apply_weight(dependent_var, weights))

    @staticmethod
    def weight(arr):
        arr[arr <= 0] = 1.
        return np.reciprocal(np.sqrt(arr))

    @staticmethod
    def apply_weight(arr, weights):

        ncols, *nrows = arr.shape
        arr_weighted = np.copy(arr)
        try:
            arr_weighted *= np.tile(weights, (*nrows, 1)).T
        except ValueError:
            arr_weighted *= weights

        return arr_weighted
        

class ChiSquareStatisticVariableProjection(ChiSquareStatistic):

    def __init__(self, optimization_method='leastsq',
                 variance_approximation=None):
        super().__init__(
            optimization_method=optimization_method,
            variance_approximation=variance_approximation)
        self.name='chi_square_statistic_variable_projection'

    def objective_func(self, model, dependent_var):
        
        if self.variance_approximation is not None:
            model, dependent_var = self._weight_input(model, dependent_var)
        
        pseudoinv_model = scipy.linalg.pinv(model)
        model = model.dot(pseudoinv_model.dot(dependent_var))
        
        return (model - dependent_var)