#!/usr/bin/env python3

"""
Module with example of fitting lifetimes to single measurements of fluorescence decay.
"""

from fluo.fitter import make_lifetime_fitter
import numpy as np
np.set_printoptions(threshold=np.nan)

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
    file = np.loadtxt('./1exp_4.9ns.txt', skiprows=1)
    time, irf, decay = file[:, 0], file[:, 1], file[:, 2]

    # print('TailFitter Chi2 Statistic')
    # make_lifetime_fitter(model_kwargs_e1_tail, time, decay, fit_statistic='chi_square_statistic').fit(report=True)
    # print()

    # print('TailFitter Chi2 Var Pro Statistic')
    # make_lifetime_fitter(model_kwargs_e1_tail, time, decay, fit_statistic='chi_square_statistic_variable_projection').fit(report=True)
    # print()

    # print('TailFitter C statistic')
    # # tailfit_cstat = \
    # make_lifetime_fitter(model_kwargs_e1_tail, time, decay,fit_statistic='c_statistic').fit()
    # print()
    # plt.plot(tailfit_cstat.residual)  
    # plt.plot(tailfit_cstat.data, 'bo')
    # plt.plot(tailfit_cstat.best_fit, 'r-')
    # plt.yscale('log')    
    # plt.show()

    # print('ConvolutionFitter C statistic')
    # fit_Cstat = \
    # make_lifetime_fitter(model_kwargs_e1, time, decay, instrument_response=irf, fit_statistic='c_statistic').fit()
    # print()    

    print('ConvolutionFitter Chi2 Statistic')
    fitter = \
    make_lifetime_fitter(model_kwargs_e1, time, decay, instrument_response=irf, fit_statistic='chi_square_statistic')
    fitter.fit()
    print()

    # n_iter = 15
    # fits = iterative_least_squares(fitter, n_iter)
    # cmap = matplotlib.cm.cool
    
    # plt.plot(
    #     fit_Cstat.best_fit,
    #     label='C stat',
    #     color='r'
    #     )
    # for ith, fit in enumerate(fits):
    #     plt.plot(
    #         fit.best_fit, 
    #         label='{}-ith iter'.format(ith), 
    #         color=cmap(ith / n_iter)
    #         )
    # plt.legend(loc='best')
    # plt.yscale('log')
    # plt.show()

    # plt.plot(
    #     fit_Cstat.residual,
    #     label='C stat',
    #     color=  'r'
    #     )
    # for ith, fit in enumerate(fits):
    #     plt.plot(
    #         fit.residual, 
    #         label='{}-ith iter'.format(ith), 
    #         color=cmap(ith / n_iter)
    #         )
    # plt.legend(loc='best')
    # plt.show()

    # print('ConvolutionFitter Chi2 Var Pro Statistic')
    # fit_Chi2_varpro = \
    # make_lifetime_fitter(model_kwargs_e1, time, decay, instrument_response=irf, fit_statistic='chi_square_statistic_variable_projection').fit()
    # print()

if __name__ == "__main__":
    main()