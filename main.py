__author__ = 'zweiss'

import numpy as np

import matplotlib.pyplot as plt


from RSA import *

from Fit_Model import *



from JSD import jsd_non_iterative
from Hellinger import hellinger_non_iterative_probs

from main_complex import get_complex_likelihood_funcs

from main_simple import get_simple_likelihood_funcs


if __name__ == '__main__':

    ###################### KLD ###########################

    complex_s_likelihood_func, complex_l_likelihood_func = get_complex_likelihood_funcs(log)

    simple_s_likelihood_func, simple_l_likelihood_func = get_simple_likelihood_funcs(log)

    ##################### SPEAKER ########################

    combined_func = multiply_funcs(complex_s_likelihood_func, simple_s_likelihood_func)

    ls = [_ for _ in np.arange(0, 10, 0.01)]

    simple_vals = [simple_s_likelihood_func(l) for l in ls]

    complex_vals = [complex_s_likelihood_func(l) for l in ls]

    combined_vals = [combined_func(l) for l in ls]

    plt.plot(ls, simple_vals, label='Speaker')

    plt.plot(ls, complex_vals, label='Listener')

    plt.plot(ls, combined_vals, label='Combined')

    plt.legend()

    plt.show()

    #################### LISTENER #######################

    combined_func = multiply_funcs(complex_l_likelihood_func, simple_l_likelihood_func)

    ls = [_ for _ in np.arange(0, 10, 0.01)]

    simple_vals = [simple_l_likelihood_func(l) for l in ls]

    complex_vals = [complex_l_likelihood_func(l) for l in ls]

    combined_vals = [combined_func(l) for l in ls]

    plt.plot(ls, simple_vals, label='Speaker')

    plt.plot(ls, complex_vals, label='Listener')

    plt.plot(ls, combined_vals, label='Combined')

    plt.legend()

    plt.show()


    ###################### JSD ###########################

    complex_s_likelihood_func, complex_l_likelihood_func = get_complex_likelihood_funcs(jsd_non_iterative)

    simple_s_likelihood_func, simple_l_likelihood_func = get_simple_likelihood_funcs(jsd_non_iterative)

    ##################### SPEAKER ########################

    combined_func = multiply_funcs(complex_s_likelihood_func, simple_s_likelihood_func)

    ls = [_ for _ in np.arange(0, 10, 0.01)]

    simple_vals = [simple_s_likelihood_func(l) for l in ls]

    complex_vals = [complex_s_likelihood_func(l) for l in ls]

    combined_vals = [combined_func(l) for l in ls]

    plt.plot(ls, simple_vals, label='Speaker')

    plt.plot(ls, complex_vals, label='Listener')

    plt.plot(ls, combined_vals, label='Combined')

    plt.legend()

    plt.show()

    #################### LISTENER #######################

    combined_func = multiply_funcs(complex_l_likelihood_func, simple_l_likelihood_func)

    ls = [_ for _ in np.arange(0, 10, 0.01)]

    simple_vals = [simple_l_likelihood_func(l) for l in ls]

    complex_vals = [complex_l_likelihood_func(l) for l in ls]

    combined_vals = [combined_func(l) for l in ls]

    plt.plot(ls, simple_vals, label='Speaker')

    plt.plot(ls, complex_vals, label='Listener')

    plt.plot(ls, combined_vals, label='Combined')

    plt.legend()

    plt.show()


    ###################### HELLINGER ###########################

    complex_s_likelihood_func, complex_l_likelihood_func = get_complex_likelihood_funcs(hellinger_non_iterative_probs)

    simple_s_likelihood_func, simple_l_likelihood_func = get_simple_likelihood_funcs(hellinger_non_iterative_probs)

    ##################### SPEAKER ########################

    combined_func = multiply_funcs(complex_s_likelihood_func, simple_s_likelihood_func)

    ls = [_ for _ in np.arange(0, 10, 0.01)]

    simple_vals = [simple_s_likelihood_func(l) for l in ls]

    complex_vals = [complex_s_likelihood_func(l) for l in ls]

    combined_vals = [combined_func(l) for l in ls]

    plt.plot(ls, simple_vals, label='Speaker')

    plt.plot(ls, complex_vals, label='Listener')

    plt.plot(ls, combined_vals, label='Combined')

    plt.legend()

    plt.show()

    #################### LISTENER #######################

    combined_func = multiply_funcs(complex_l_likelihood_func, simple_l_likelihood_func)

    ls = [_ for _ in np.arange(0, 10, 0.01)]

    simple_vals = [simple_l_likelihood_func(l) for l in ls]

    complex_vals = [complex_l_likelihood_func(l) for l in ls]

    combined_vals = [combined_func(l) for l in ls]

    plt.plot(ls, simple_vals, label='Speaker')

    plt.plot(ls, complex_vals, label='Listener')

    plt.plot(ls, combined_vals, label='Combined')

    plt.legend()

    plt.show()
