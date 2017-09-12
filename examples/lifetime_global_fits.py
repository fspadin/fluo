#!/usr/bin/env python3

"""
Module with example of simultaneously fitting lifetimes to multiple measurements of fluorescence decay.
"""

from fluo.fitter import make_global_lifetime_fitter
import numpy as np
np.set_printoptions(threshold=np.nan)

def main():  
    fit_kwargs = {
        'conf_i': False,
        '-C': False,
        'no-output': False,
        'plot': True,
        '-p': False,
    }
    model_kwargs_e1_tail = {
        'fit_components': 1,
        'initial_parameters': {
            'amplitude1': {'value': 7000, 'vary': True},
            'offset': {'value': 0.1, 'vary': True},
            'tau1': {'value': 5, 'vary': True},
        },
        'start': 12,
        'stop': None
    }
    model_kwargs_e1 = {
        'fit_components': 1,
        'initial_parameters': {
            'amplitude1': {'value': 0.06, 'vary': True},
            'offset': {'value': 0.1, 'vary': True},
            'tau1': {'value': 5, 'vary': True},
        },
        'start': 2,
        'stop': None
    }    
    model_kwargs_e2_tail = {
        'fit_components': 2,
        'initial_parameters': {
            'offset': {'value': 0.1, 'vary': True},
            'tau1': {'value': 5, 'vary': True},
            'tau2': {'value': 4, 'vary': True}
        },
        'start': 12,
        'stop': None
    }    
    model_kwargs_e2 = {
        'fit_components': 2,
        'initial_parameters': {
            'offset': {'value': 0.1, 'vary': True},
            'tau1': {'value': 5, 'vary': True},
            'tau2': {'value': 4, 'vary': True}
        },
        'start': 2,
        'stop': None
    } 
    file1 = np.loadtxt('./1exp_4.9ns.txt', skiprows=1)
    file2 = np.loadtxt('./1exp_5ns_ble.txt', skiprows=1)


    local_time = [file1[:, 0], file2[:, 0]]
    local_data = [file1[:, 2], file2[:, 2]]
    local_irf = [file1[:, 1], file2[:, 1]]

    local_model_kwargs_e1_tail = [model_kwargs_e1_tail, model_kwargs_e1_tail.copy()]
    local_model_kwargs_e1 = [model_kwargs_e1, model_kwargs_e1.copy()]
    
    print('GlobalTailFitter Chi2 statistic')
    global_tail = \
    make_global_lifetime_fitter(local_model_kwargs_e1_tail, local_time, local_data, fit_statistic='chi_square_statistic', shared=['tau1'])
    global_tail.fit()
    print()
        
    # print('GlobalTailFitter Chi2 Var Pro statistic')
    # global_tail = \
    # make_global_lifetime_fitter(local_model_kwargs_e1_tail, local_time, local_data, fit_statistic='chi_square_statistic_variable_projection', shared=['tau1'])
    # global_tail.fit()
    # print()

    # print('GlobalTailFitter C statistic')
    # global_tail = \
    # make_global_lifetime_fitter(local_model_kwargs_e1_tail, local_time, local_data, fit_statistic='c_statistic', shared=['tau1'])
    # global_tail.fit()
    # print()

    # print('GlobalConvolutionFitter Chi2 statistic')
    # global_conv = \
    # make_global_lifetime_fitter(local_model_kwargs_e1, local_time, local_data, local_irf, fit_statistic='chi_square_statistic', shared=['tau1']).fit()
    # print()

    # print('GlobalConvolutionFitter Chi2 Var Pro statistic')
    # global_conv = \
    # make_global_lifetime_fitter(local_model_kwargs_e1, local_time, local_data, local_irf, fit_statistic='chi_square_statistic_variable_projection', shared=['tau1'])
    # local_indexes = global_conv.ModelClass.local_indexes
    # global_conv_fit = global_conv.fit()
    # print(np.split(global_conv_fit.best_fit, local_indexes))
    # print()

    # print('GlobalConvolutionFitter C statistic')
    # global_conv = \
    # make_global_lifetime_fitter(local_model_kwargs_e1, local_time, local_data, local_irf, shared=['tau1']).fit()
    # print()

if __name__ == "__main__":
    main()