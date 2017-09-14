# -*- coding: utf-8 -*-

"""
Module with models for fitting. GenericModel overrides lmfit.Model to utilize Statistic object in fit.
"""

from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.interpolate
import lmfit  
import random

class Model(): 
    """Wrapper around fluo.GenericModel.

    Abstract class for Model objects. 

    Parameters
    ----------
    model_components : int
        Number of components in model (i. e. number of exponents).
    model_parameters : dict
        Dict with names of parameters encoded by keys (str)
        and values with dictionary. 
    """    
    __metaclass__ = ABCMeta
    
    def __init__(self, model_components, model_parameters=None):
        self.model_components = model_components
        self.model_parameters = model_parameters

    def make_model(self, **independent_var): 
        """Makes fluo.GenericModel.
        """
        return GenericModel(
    self.model_function(**independent_var), 
    missing='drop', 
    name=self.__class__.__name__
    )

    @abstractmethod
    def model_function(self, **independent_var):
        raise NotImplementedError()

    @abstractmethod
    def make_parameters(self):
        raise NotImplementedError()

class GlobalModel(Model):
    
    def __init__(self, FitterClasses, shared=None):
        self.FitterClasses = FitterClasses
        self.shared = shared
        if self.shared is None:
            self.shared = []   
        self.local_ModelClass = self.make_local_atrribute(FitterClasses, 'ModelClass')
        self.local_independent_var = self.make_local_atrribute(FitterClasses, 'independent_var')
        self.local_dependent_var = self.make_local_atrribute(FitterClasses, 'dependent_var')
        self.local_indexes = self.make_local_indexes(self.local_dependent_var)
        self.statistic = FitterClasses[0].statistic # statistic in every Fitter class should be the same
        self.local_parameters = self.make_local_atrribute(FitterClasses, 'parameters')
        self._parameters, self.parameters_references = self.glue_parameters()

    @staticmethod
    def make_local_atrribute(fitters, atrr):
        return [getattr(fitter, atrr) for fitter in fitters]

    @staticmethod
    def make_local_indexes(arrs):
        return np.cumsum([len(arr) for arr in arrs])[:-1]

    def model_function(self, **independent_var):
        return self.global_eval(**independent_var)

    def make_parameters(self):
        return self._parameters

    def global_eval(self, independent_var):
        def inner_global_eval(**params):
            for name, value in params.items():
                fitter_i, local_name = self.parameters_references[name]
                self.local_parameters[fitter_i][local_name].value = value
            global_eval = []
            for i, local_fitter in enumerate(self.FitterClasses):
                model_i = local_fitter.ModelClass.make_model(**independent_var[i])
                local_eval = model_i.eval(**self.local_parameters[i])
                global_eval.append(local_eval)
            return np.concatenate(global_eval)
        return inner_global_eval

    def glue_parameters(self):
        parameters_references = dict()
        all_params = lmfit.Parameters()
        for i, params_i in enumerate(self.local_parameters):
            for old_name, param in params_i.items():
                new_name = old_name + '_file%d' % (i + 1)
                parameters_references[new_name] = (i, param.name)
                all_params.add(new_name,
                               value=param.value,
                               vary=param.vary,
                               min=param.min,
                               max=param.max,
                               expr=param.expr)
        for param_name, param in all_params.items():
            for constraint in self.shared:
                if param_name.startswith(constraint) and not param_name.endswith('_file1'):
                    param.expr = constraint + '_file1'
        return all_params, parameters_references

class AddConstant():
    
    def __init__(self, ModelClass):
        self.ModelClass = ModelClass

    def make_model(self, **independent_var): 
        return GenericModel(self.model_function(**independent_var), missing='drop', name=self.ModelClass.__class__.__name__)

    def model_function(self, **independent_var):
        return self.add_constant(**independent_var)    

    def make_parameters(self):
        pars = self.ModelClass.make_parameters()
        pars.add(
            'offset', 
            **self.model_parameters.get('offset', {'value': 0, 'vary': True})
            )
        return pars     

    def add_constant(self, **independent_var): 
        func=self.ModelClass.model_function(**independent_var)
        def inner_add_constant(**params):
            offset = params.pop('offset')
            return func(**params) + offset

        return inner_add_constant

    def __getattr__(self, attr):
        return getattr(self.ModelClass, attr)
     
class Linearize():
    
    def __init__(self, ModelClass):
        self.ModelClass = ModelClass

    def make_model(self, **independent_var): 
        return GenericModel(self.model_function(**independent_var), missing='drop', name=self.ModelClass.__class__.__name__)

    def model_function(self, **independent_var):
        return self.composite(**independent_var)
         
    def make_parameters(self):
        nonlinear_params = self.ModelClass.make_parameters()
        linear_params = Linear(self.model_components, self.model_parameters).make_parameters()
        for param_name, param in linear_params.items():
            nonlinear_params.add(
            param_name,
            value=param.value,
            vary=param.vary,
            min=param.min,
            max=param.max,
            expr=param.expr)
        
        return nonlinear_params
    
    def composite(self, **independent_var):
        linear_func=Linear.linear
        nonlinear_func=self.ModelClass.model_function(**independent_var)        
        def inner_composite(**params):          
            nonlinear_params = {
                key: params[key] for key in params.keys() if (
                    key.startswith('tau') or key.startswith('shift')
                    )
                    }
            linear_params = {
                key: params[key] for key in params.keys() if (
                    key.startswith('amplitude')
                    )
                    } 
            return linear_func(nonlinear_func(**nonlinear_params))(**linear_params)

        return inner_composite
 
    def __getattr__(self, attr):
        return getattr(self.ModelClass, attr)

class Convolve():

    def __init__(self, ModelClass, convolution_method='discrete'):
        self.ModelClass = ModelClass
        self.convolution_method = convolution_method
        self._convolve = self._allowed_convolutions[convolution_method]

    def make_model(self, **independent_var): 
        name = '{} convolved wtih instrument response using {}'.format(
            self.ModelClass.__class__.__name__,
            self.convolution_method)
        return GenericModel(self.model_function(**independent_var), missing='drop', name=name)

    def model_function(self, **independent_var):
        return self.convolved_exponential(**independent_var)

    def make_parameters(self):
        nonlinear_pars = self.ModelClass.make_parameters()
        nonlinear_pars.add(
            'shift', 
            **self.model_parameters.get('shift', {'value': 0, 'vary': True})
            )
        return nonlinear_pars    

    def convolved_exponential(self, **independent_var):
        independent_var = independent_var.copy()
        time = independent_var['time']
        instrument_response = independent_var.pop('instrument_response')

        def inner_convolved_exponential(**params):
            params = params.copy()
            shifted_instrument_response = self.shift_decay(
                time, 
                instrument_response, 
                params.pop('shift')
                )
            to_convolve_with = self.ModelClass.model_function(**independent_var)(**params)      
            ncols, *nrows = to_convolve_with.shape
            try:
                convolved = np.zeros(to_convolve_with.shape)       
                for i in range(*nrows):
                    convolved[:, i] = self._convolve(shifted_instrument_response, to_convolve_with[:, i])
            except TypeError:
                convolved = self._convolve(
                    shifted_instrument_response, 
                    to_convolve_with)
            return convolved

        return inner_convolved_exponential

    @property
    def _allowed_convolutions(self):
        return dict(
                discrete=self.convolve,
                monte_carlo=self.monte_carlo_convolve
                )

         

    @staticmethod
    def shift_decay(time, intensity, shift):
        """
        Shift decay in time x-axis.
        """
        s = scipy.interpolate.interp1d(
            time,
            intensity,
            kind='slinear',
            bounds_error=False,
            fill_value=0.0)
        return s(time + shift)

    @staticmethod
    def convolve(left, right):
        return np.convolve(left, right, mode='full')[:len(right)]
    
    @staticmethod
    def monte_carlo_convolve(left, right, peak_cnts=None, verbose=True):
        """
        Compute Monte Carlo convolution.

        Parameters
        ----------
        left : ndarray
            1D array
        right : ndarray
            1D array (should be the same length as `left`).
        peak_cnts : int, optional
            By default max of `left`.
        verbose : bool
            Print simulation progress.

        Returns
        -------
        list
            Convolution using Monte Carlo method.
        """
        left_max = np.max(left)
        P_left = list(left/left_max)  # probability distribution of left scalled to 1
        X_max = len(P_left)-1
        P_right = list(right/np.max(right))  # probability distribution of right scalled to 1
        print()
        print('[[Wait until Monte Carlo simulation is done. It may take some time.]]')
        print()
        MC_convolution = [0] * len(P_left)
        if peak_cnts == None:
            peak_cnts = left_max
        else:
            peak_cnts = int(peak_cnts)
        while (max(MC_convolution) < peak_cnts): # stops when peak_counts is reached
            if verbose:
                print('Peak counts\t: {}'.format(max(MC_convolution)))
            X_left = draw_from_probability_distribution(P_left)
            X_right = draw_from_probability_distribution(P_right)
            X_drawn = X_left + X_right  # draw channel number
            if X_drawn <= X_max:  # channel must be in range
                MC_convolution[X_drawn] += 1  # add count in channel
        return MC_convolution


    def __getattr__(self, attr):
        return getattr(self.ModelClass, attr)

class Exponential(Model):

    def model_function(self, **independent_var):
        return self.exponential(**independent_var)
    
    def make_parameters(self):
        nonlinear_pars = lmfit.Parameters()         
        for i in range(self.model_components):
            nonlinear_pars.add(
                'tau{}'.format(i+1), 
                **self.model_parameters.get('tau{}'.format(i+1), {'value': 1, 'vary': True, 'min': 1E-6})
                )
        return nonlinear_pars

    @staticmethod
    def exponential(time):
        def inner_exponential(**taus):
            # taus = sorted_values(taus)
            taus = np.asarray(list(taus.values())) # may fail if not sorted
            return np.exp(-time[:, None] / taus[None, :])
        return inner_exponential

class Linear(Model):
    
    def model_function(self, independent_var):
        return self.linear(independent_var)

    def make_parameters(self):
        linear_pars = lmfit.Parameters()
        for i in range(self.model_components):
            linear_pars.add(
               'amplitude{}'.format(i+1), 
                **self.model_parameters.get('amplitude{}'.format(i+1), {'value': 1, 'vary': True})
            )               
        return linear_pars
    
    @staticmethod
    def linear(independent_var):
        def inner_linear(**linear_params):
            # amplitudes = sorted_values(linear_params)
            amplitudes = np.asarray(list(linear_params.values())) # may fail if not sorted
            return independent_var.dot(amplitudes)
        return inner_linear


class GenericModel(lmfit.Model):

    def __init__(self, func, independent_vars=None,
                 param_names=None, missing='none', prefix='', name=None, **kws):
        super().__init__(func, independent_vars, param_names,
                 missing, prefix, name, **kws)
        self._statistic = None

    def _residual(self, params, data, weights, **kwargs):
        model = self.eval(params, **kwargs)
        result = self._statistic.objective_func(model, data)
        return np.asarray(result).ravel()

    def generic_fit(self, data, statistic, params,
            iter_cb=None, scale_covar=True, verbose=False, fit_kws=None,
            **kwargs):
        self._statistic = statistic
        method = self._statistic.optimization_method
        weights = None
        return super().fit(data, params, weights, method,
            iter_cb, scale_covar, verbose, fit_kws,
            **kwargs)


def draw_from_probability_distribution(distribution):
    """
    Draw from arbitrary distribution using acceptance-rejection method.

    Parameters
    ----------
    distribution : list
        List with probabalities distribution (scalled to 1).

    Returns
    -------
    int
        Drawn channel's index.
    """
    x_max = len(distribution)-1
    y_min = min(distribution)

    accepted = False
    while (not accepted):
        x_random = random.randint(0, x_max)
        y_random = random.uniform(y_min, 1.)
        if y_random <= distribution[x_random]:
            accepted = True
            return x_random