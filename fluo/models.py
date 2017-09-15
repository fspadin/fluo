# -*- coding: utf-8 -*-

"""
Module with fluo.Model object for fitting. GenericModel overrides lmfit.Model to utilize Statistic object in fit.
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

    Abstract class for fluo.Model objects. 

    Parameters
    ----------
    model_components : int
        Number of components in model (i. e. number of exponents).
    model_parameters : dict
        Dict with names of parameters encoded by keys (str)
        and values with dictionary. 

    Attributes
    ----------
    name : str

    Methods
    -------
    make_model : fluo.GenericModel
    """    
    __metaclass__ = ABCMeta
    
    def __init__(self, model_components, model_parameters=None):
        self.model_components = model_components
        self.model_parameters = model_parameters
        self.name = self.__class__.__name__

    def make_model(self, **independent_var): 
        """Makes a Model for evaluation and fitting.

        Prameteres 
        ----------
        independent_var : dict 
            Keyword arguments for model evaluation.

        Returns
        -------
        fluo.GenericModel
        """
        return GenericModel(
    self._model_function(**independent_var), 
    missing='drop', 
    name=self.name
    )

    @abstractmethod
    def _model_function(self, **independent_var):
        raise NotImplementedError()

    @abstractmethod
    def make_parameters(self):
        raise NotImplementedError()

class GlobalModel(Model):
    """GlobalModel object composed from multiple fluo.Fitter classes.

    Parameters
    ----------
    FitterClasses : list of fluo.Fitter
    shared : list of str, optional
        List of parameters names shared between fitted measurements.

    Attributes
    ----------
    name : str
    local_independent_var : list of dict
        Independent variables for a local model evaluation. Dict with names of independent variables encoded by keys (str)
        and values as ndarrays.
    local_dependent_var : list of ndarray
        List of 1D ndarrays with dependent variable for fitting.
    local_indexes : ndarray
        Indexes separating individual measurements.
    statistic : fluo.Statistic
        Global Statistic class for simultaneous fit.

   
    Methods
    -------
    make_model : fluo.GenericModel
        Global Model for simultaneous fitting.
    make_parameters : lmfit.Parameters
    make_local_atrribute : list
        List of local attribute of FitterClasses.
    """
    def __init__(self, FitterClasses, shared=None):
        self.FitterClasses = FitterClasses
        self.shared = shared
        if self.shared is None:
            self.shared = []  
        self.name = '{}({})'.format(
            self.__class__.__name__,
            FitterClasses[0].ModelClass.name # name in every Fitter class should be the same
            )
        self.local_independent_var = self.make_local_atrribute(FitterClasses, 'independent_var')
        self.local_dependent_var = self.make_local_atrribute(FitterClasses, 'dependent_var')
        self.local_indexes = self._make_local_indexes(self.local_dependent_var)
        self.statistic = FitterClasses[0].statistic # statistic in every Fitter class should be the same
        self._local_parameters = self.make_local_atrribute(FitterClasses, 'parameters')
        self._parameters, self._parameters_references = self._glue_parameters()

    def _model_function(self, **independent_var):
        return self._global_eval(**independent_var)

    def make_parameters(self):
        """Makes parameters for Model evaluation.

        Returns
        -------
        lmfit.Parameters
        """
        return self._parameters

    def _global_eval(self, independent_var):
        def _inner_global_eval(**params):
            for name, value in params.items():
                fitter_i, local_name = self._parameters_references[name]
                self._local_parameters[fitter_i][local_name].value = value
            _global_eval = []
            for i, local_fitter in enumerate(self.FitterClasses):
                model_i = local_fitter.ModelClass.make_model(**independent_var[i])
                local_eval = model_i.eval(**self._local_parameters[i])
                _global_eval.append(local_eval)
            return np.concatenate(_global_eval)
        return _inner_global_eval

    def _glue_parameters(self):
        _parameters_references = dict()
        all_params = lmfit.Parameters()
        for i, params_i in enumerate(self._local_parameters):
            for old_name, param in params_i.items():
                new_name = old_name + '_file%d' % (i + 1)
                _parameters_references[new_name] = (i, param.name)
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
        return all_params, _parameters_references

    @staticmethod
    def make_local_atrribute(fitters, atrr):
        return [getattr(fitter, atrr) for fitter in fitters]

    @staticmethod
    def _make_local_indexes(arrs):
        return np.cumsum([len(arr) for arr in arrs])[:-1]

class AddConstant():
    """Adds a constant to a model.
    
    Wrapper around fluo.Model. 

    Parameters
    ----------
    ModelClass : fluo.Model

    Methods
    -------
    make_model : fluo.GenericModel
    make_parameters : lmfit.Parameters
    """
    def __init__(self, ModelClass):
        self.ModelClass = ModelClass
        self.name = '{}({})'.format(
            self.__class__.__name__,
            self.ModelClass.name
            )

    def make_model(self, **independent_var): 
        """Makes a Model for evaluation and fitting.

        Prameteres 
        ----------
        independent_var : dict 
            Dictionary with independen variable for model evaluation.

        Returns
        -------
        fluo.GenericModel
        """        
        return GenericModel(self._model_function(**independent_var), missing='drop', name=self.name)

    def _model_function(self, **independent_var):
        return self._add_constant(**independent_var)    

    def make_parameters(self):
        """Makes parameters for Model evaluation.

        Returns
        -------
        lmfit.Parameters
        """        
        pars = self.ModelClass.make_parameters()
        pars.add(
            'offset', 
            **self.model_parameters.get('offset', {'value': 0.1, 'vary': True})
            )
        return pars     

    def _add_constant(self, **independent_var): 
        model = self.ModelClass.make_model(**independent_var)
        def _inner_add_constant(**params):
            offset = params.pop('offset')
            return model.eval(**params) + offset

        return _inner_add_constant

    def __getattr__(self, attr):
        return getattr(self.ModelClass, attr)
     
class Linearize():
    """Linear combination of a Model.
    
    Wrapper around fluo.Model. Makes dot product of a Model and linear coeficients. Utilizes Linear class.

    Parameters
    ----------
    ModelClass : fluo.Model

    Methods
    -------
    make_model : fluo.GenericModel
    make_parameters : lmfit.Parameters    
    """
    def __init__(self, ModelClass):
        self.ModelClass = ModelClass
        self.name = '{}({})'.format(
            self.__class__.__name__, 
            self.ModelClass.name
            )

    def make_model(self, **independent_var): 
        """Makes a Model for evaluation and fitting.

        Prameteres 
        ----------
        independent_var : dict 
            Dictionary with independen variable for model evaluation.

        Returns
        -------
        fluo.GenericModel
        """         
        return GenericModel(self._model_function(**independent_var), missing='drop', name=self.name)

    def _model_function(self, **independent_var):
        return self._composite(**independent_var)
         
    def make_parameters(self):
        """Makes parameters for Model evaluation.

        Returns
        -------
        lmfit.Parameters
        """            
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
    
    def _composite(self, **independent_var):
        linear_func = Linear.linear
        nonlinear_model = self.ModelClass.make_model(**independent_var)        
        def _inner_composite(**params):          
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
            return linear_func(nonlinear_model.eval(**nonlinear_params))(**linear_params)

        return _inner_composite
 
    def __getattr__(self, attr):
        return getattr(self.ModelClass, attr)

class Convolve():
    """Convolves model with instrument response.
    
    Wrapper around fluo.Model.

    Parameters
    ----------
    ModelClass : fluo.Model
    convolution_method : str, optional
        'discrete' by default. Accepts the following str: 'discrete', 'monte_carlo' (for Monte Carlo convolution).

    Methods
    -------
    make_model : fluo.GenericModel
    make_parameters : lmfit.Parameters  
    shift_decay : ndarray
    convolve : ndarray
    monte_carlo_convolve : ndarray
    """
    def __init__(self, ModelClass, convolution_method='discrete', **convolution_kwargs):
        self.ModelClass = ModelClass
        self.convolution_method = convolution_method
        self.convolution_kwargs = convolution_kwargs
        self.__convolve = self._allowed_convolutions[convolution_method]
        self.name = '{}({})'.format(
            self.__class__.__name__,
            self.ModelClass.name
            )

    def make_model(self, **independent_var): 
        """Makes a Model for evaluation and fitting.

        Prameteres 
        ----------
        independent_var : dict 
            Keyword arguments for model evaluation.

        Returns
        -------
        fluo.GenericModel
        """      
        return GenericModel(self._model_function(**independent_var), missing='drop', name=self.name)

    def _model_function(self, **independent_var):
        return self._convolve(**independent_var)

    def make_parameters(self):
        """Makes parameters for Model evaluation.

        Returns
        -------
        lmfit.Parameters
        """            
        nonlinear_pars = self.ModelClass.make_parameters()
        nonlinear_pars.add(
            'shift', 
            **self.model_parameters.get('shift', {'value': 1, 'vary': True})
            )
        return nonlinear_pars    

    def _convolve(self, **independent_var):
        independent_var = independent_var.copy()
        time = independent_var['time']
        instrument_response = independent_var.pop('instrument_response')

        def _inner_convolve(**params):
            params = params.copy()
            shifted_instrument_response = self.shift_decay(
                time, 
                instrument_response, 
                params.pop('shift')
                )
            to_convolve_with = self.ModelClass.make_model(**independent_var).eval(**params)      
            ncols, *nrows = to_convolve_with.shape
            try:
                convolved = np.zeros(to_convolve_with.shape)       
                for i in range(*nrows):
                    convolved[:, i] = self.__convolve(shifted_instrument_response, to_convolve_with[:, i], **self.convolution_kwargs)
            except TypeError:
                convolved = self.__convolve(
                    shifted_instrument_response, 
                    to_convolve_with, **self.convolution_kwargs)
            return convolved

        return _inner_convolve

    @property
    def _allowed_convolutions(self):
        return dict(
                discrete=self.convolve,
                monte_carlo=self.monte_carlo_convolve
                )

    @staticmethod
    def shift_decay(x_var, y_var, shift):
        """
        Shift y-axis variable on x-axis.

        Parameters
        ----------
        x_var : ndarray
        y_var : ndarray
        shift : float

        Returns
        -------
        ndarray
        """
        y_var_interpolated = scipy.interpolate.interp1d(
            x_var,
            y_var,
            kind='slinear',
            bounds_error=False,
            fill_value=0.0)
        return y_var_interpolated(x_var + shift)

    @staticmethod
    def convolve(left, right):
        """Discrete convolution of `left` with `right`.

        Parameters
        ----------
        left : ndarray
            Left 1D ndarray
        right : ndarray
            Right 1D ndarray      
            
        Returns
        -------
        ndarray         
        """
        return np.convolve(left, right, mode='full')[:len(right)]
    
    @staticmethod
    def monte_carlo_convolve(left, right, peak_cnts=None, verbose=True):
        """Monte Carlo convolution of `left` with `right`. 
        
        Simulates distorted fluorescence decay distorted by instrument response with Poisson distributed observed values.

        Parameters
        ----------
        left : ndarray
            1D array
        right : ndarray
            1D array (should be the same length as `left`).
        peak_cnts : int, optional
            Counts in maximum (max of `left` by default).
        verbose : bool
            Print simulation progress.

        Returns
        -------
        ndarray
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
        while (max(MC_convolution) <= peak_cnts): # stops when peak_counts is reached
            if verbose:
                print('Peak counts\t: {}'.format(max(MC_convolution)))
            X_left = draw_from_probability_distribution(P_left)
            X_right = draw_from_probability_distribution(P_right)
            X_drawn = X_left + X_right  # draw channel number
            if X_drawn <= X_max:  # channel must be in range
                MC_convolution[X_drawn] += 1  # add count in channel
        return np.asarray(MC_convolution)

    def __getattr__(self, attr):
        return getattr(self.ModelClass, attr)

class Exponential(Model):
    """Exponential Model.

    Parameters
    ----------
    model_components : int
        Number of components in model (i. e. number of exponents).
    model_parameters : dict
        Dict with names of parameters encoded by keys (str)
        and values with dictionary. 

    Attributes
    ----------
    name : str

    Methods
    -------
    make_model : fluo.GenericModel     
    make_parameters : lmfit.Parameters  
    exponential : ndarray
    """
    def _model_function(self, **independent_var):
        return self.exponential(**independent_var)
    
    def make_parameters(self):
        """Makes parameters for Model evaluation.

        Returns
        -------
        lmfit.Parameters
        """             
        nonlinear_pars = lmfit.Parameters()         
        for i in range(self.model_components):
            nonlinear_pars.add(
                'tau{}'.format(i+1), 
                **self.model_parameters.get('tau{}'.format(i+1), {'value': 1, 'vary': True, 'min': 1E-6})
                )
        return nonlinear_pars

    @staticmethod
    def exponential(time):
        """Exponential decay."""
        def inner_exponential(**taus):
            # taus = sorted_values(taus)
            taus = np.asarray(list(taus.values())) # may fail if not sorted
            return np.exp(-time[:, None] / taus[None, :])
        return inner_exponential

class Linear(Model):
    """Linear Model.

    Makes dot product of an independent variable and linear coeficients.

    Parameters
    ----------
    model_components : int
        Number of components in model (i. e. number of exponents).
    model_parameters : dict
        Dict with names of parameters encoded by keys (str)
        and values with dictionary. 

    Attributes
    ----------
    name : str

    Methods
    -------
    make_model : fluo.GenericModel     
    make_parameters : lmfit.Parameters  
    exponential : ndarray
    """
    def _model_function(self, independent_var):
        return self.linear(independent_var)

    def make_parameters(self):
        """Makes parameters for Model evaluation.

        Returns
        -------
        lmfit.Parameters
        """             
        linear_pars = lmfit.Parameters()
        for i in range(self.model_components):
            linear_pars.add(
               'amplitude{}'.format(i+1), 
                **self.model_parameters.get('amplitude{}'.format(i+1), {'value': 0.5, 'vary': True})
            )               
        return linear_pars
    
    @staticmethod
    def linear(independent_var):
        """Linear combination.

        Parameters
        ----------
        independent_var : ndarray
            2D ndarray for a dot product.

        Returns
        -------
        ndarray
            1D ndarray with dot product of an independent variable and linear coeficients.
        """
        def inner_linear(**amplitudes):
            # amplitudes = sorted_values(amplitudes)
            amplitudes = np.asarray(list(amplitudes.values())) # may fail if not sorted
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
    Draws from an arbitrary distribution using acceptance-rejection method.

    Parameters
    ----------
    distribution : ndarray
        1D ndarray with probabalities distribution (scalled to 1).

    Returns
    -------
    int
        Drawn index.
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


def sorted_values(parameters):
    """Values of sorted parameters.
    
    Parameters
    ----------
    parameters : lmfit.Parameters

    Returns
    -------
    ndarray
    """
    parameters = OrderedDict(sorted(parameters.items()))
    sorted_vals = np.asarray(list(parameters.values()))
    return sorted_vals        