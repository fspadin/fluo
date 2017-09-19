
# -*- coding: utf-8 -*-

"""Provide utility functions."""

def pretty_print_dictionary(dictionary):
    """Print dictionary with keys-values pairs in separate lines.

    Keys are separated from values with colons. The values are formatted with
    engineering notation.

    """
    buff = []
    keylen = max(len(n) for n in dictionary)
    for key in dictionary:
        value = dictionary[key]
        space = ' '*(keylen+1-len(key))
        formatted_key = '{}:{}'.format(key, space)
        try:
            formatted_value = '{0:.6g}'.format(value)
        except (TypeError, ValueError):
            formatted_value = 'Non Numeric Value?'
        buff.append("    {} {}".format(formatted_key, formatted_value))
    print('\n'.join(buff))


def process_amplitudes(parameters):
    """Calculate scaled amplitudes and fractions.

    Scale amplitudes (pre-exponential factors) so they sum up to one. Calculate
    fractional contributions of lifetimes to the steady-state intensity.

    Parameters
    ----------
    parameters : dict
        Dict-like object with amplitudes and taus.

    Returns
    -------
    dict

    """
    keys = [
        key for key in parameters if (
            key.startswith('amplitude')
        )
    ]
    amplitudes = [
        parameters[key] for key in parameters if (
            key.startswith('amplitude')
        )
    ]
    taus = [
        parameters[key] for key in parameters if (
            key.startswith('tau')
        )
    ]
    amplitudes_mul_taus = [
        amp*tau for amp, tau in zip(amplitudes, taus)
    ]
    fractions = [
        ith / sum(amplitudes_mul_taus)  for ith in amplitudes_mul_taus
    ]
    scaled_amplitudes = [
        ith / sum(amplitudes) for ith in amplitudes
    ]
    result = {}
    for key, amp in zip(keys, scaled_amplitudes):
        result['scaled_{}'.format(key)] = amp
    for key, frac in zip(keys, fractions):
        result['fractional_{}'.format(key)] = frac
    return result
