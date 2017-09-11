#!/usr/bin/env python3
from fluo import make_global_lifetime_fitter
import numpy as np

def main():  
    fit_kwargs = {
        '--conf_i': False,
        '-C': False,
        '--no-output': False,
        '--plot': True,
        '-p': False,
    }
    model_kwargs_e1_tail = {
        '<lt-fit_components>': '1',
        '--amplitude': ['~7000'],
        '--gamma': [],
        '--noiseless_IRF': False,
        '--offset': '~0.1',
        '--positive-amplitudes': False,
        '--quench': [],
        '--shift': None,
        '--start': 12,
        '--stop': None,
        '--tau': ['~5'],
        '--width': []
    }
    model_kwargs_e1 = {
        '<lt-fit_components>': '1',
        '--amplitude': ['~0.06'],
        '--gamma': [],
        '--noiseless_IRF': False,
        '--offset': '~0.1',
        '--positive-amplitudes': False,
        '--quench': [],
        '--shift': None,
        '--start': 2,
        '--stop': None,
        '--tau': ['~5'],
        '--width': []
    }
    model_kwargs_e2_tail = {
        '--amplitude': [None, None],
        '--gamma': [],
        '--noiseless_IRF': False,
        '--offset': None,
        '--positive-amplitudes': False,
        '--quench': [],
        '--shift': None,
        '--start': 12,
        '--stop': None,
        '--tau': ['~5', '~4'],
        '--width': [],
        '<lt-fit_components>': '2'
    }
    model_kwargs_e2 = {
        '--amplitude': [None, None],
        '--gamma': [],
        '--noiseless_IRF': False,
        '--offset': None,
        '--positive-amplitudes': False,
        '--quench': [],
        '--shift': None,
        '--start': 2,
        '--stop': None,
        '--tau': ['~5', '~4'],
        '--width': [],
        '<lt-fit_components>': '2'
    }
    file1 = np.loadtxt('./1exp_4.9ns.txt', skiprows=1)
    file2 = np.loadtxt('./1exp_5ns_ble.txt', skiprows=1)


    local_time = [file1[:, 0], file2[:, 0]]
    local_data = [file1[:, 2], file2[:, 2]]
    local_irf = [file1[:, 1], file2[:, 1]]

    local_model_kwargs_e1_tail = [model_kwargs_e1_tail]*2
    local_model_kwargs_e1 = [model_kwargs_e1]*2
    
    # print('GlobalTailFitter Chi2 statistic')
    # global_tail = \
    # make_global_lifetime_fitter(local_model_kwargs_e1_tail, local_time, local_data, fit_statistic='chi_square_statistic', shared=['tau1'])
    # global_tail.fit()
    # print()
        
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