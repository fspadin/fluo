#!/usr/bin/env python3

"""
Module with example of simulating fluorescence decay with Monte Carlo method.
"""

from models import ConvolvedExponential, Linearize
import numpy as np
import random

def simulate_measurement_MC(time, IRF, params, arguments):
    shift = params['shift']
    IRF_shifted = shift_intensity(time, IRF, shift)  # shift IRF
    amplitudes = params['amplitude']
    amplitudes = transpose(amplitudes)
    # calculate model
    exponents = create_model_exp(time, params, arguments['--type'])
    model = np.nansum(amplitudes * exponents, axis=0)
    measurement = monte_carlo_convolution(IRF_shifted, model, arguments['--peak'])
    return model, measurement


def draw_from_probability_distribution(distribution, x_max, y_min):
    """
    Draw from arbitrary distribution using acceptance-rejection method.

    Parameters
    ----------
    distribution : list
        List with probabalities distribution (scalled to 1).
    x_max : int
        Maximal index to draw from.
    y_min : int

    Returns
    -------
    int
        Drawn channel's index.
    """
    accepted = False
    while (not accepted):
        x_random = random.randint(0, x_max)
        y_random = random.uniform(y_min, 1.)
        if y_random <= distribution[x_random]:
            accepted = True
            return x_random

def monte_carlo_convolution(IRF, model, peak_cnts=None):
    """
    Compute Monte Carlo convolution.

    Parameters
    ----------
    IRF : ndarray
        1D array with instrument response function.
    model : ndarray
        1D array with model values (should be the same length as IRF)
    peak_cnts : int, optional
        By default max of IRF.

    Returns
    -------
    list
        List with counts.
    """
    IRF_peak = np.max(IRF)
    P_IRF = list(IRF/IRF_peak)  # probability distribution of IRF scalled to 1
    IRF_max = len(P_IRF)-1
    IRF_min = min(P_IRF)
    P_I = list(model/np.max(model))  # probability distribution of model scalled to 1
    I_min = min(P_I)

    print()
    print('[[Wait until Monte Carlo simulation is done. May take some time.]]')
    print()
    I_model_MCconv = [0] * len(P_IRF)
    if peak_cnts == None:
        peak_cnts = IRF_peak
    else:
        peak_cnts = int(peak_cnts)
    while (max(I_model_MCconv) < peak_cnts): # stops when peak_counts is reached
        X_IRF = draw_from_probability_distribution(P_IRF, IRF_max, IRF_min)
        X_I = draw_from_probability_distribution(P_I, IRF_max, I_min)
        X_F = X_IRF + X_I  # draw channel number
        if X_F <= IRF_max:  # channel must be in range
            I_model_MCconv[X_F] += 1  # add count in channel
    return I_model_MCconv


def main():
    pass

if __name__ == "__main__":
    main()