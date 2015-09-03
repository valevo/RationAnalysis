__author__ = 'Valentin'

import numpy as np

import matplotlib.pyplot as plt


from RSA import *

from Fit_Model import *



from JSD import jsd_non_iterative
from Hellinger import hellinger_non_iterative

if __name__ == '__main__':

    ################### EXPERIMENT 1 (COMPLEX) ################

    complex_game = np.matrix("""
                           [1 1 0;
                            1 0 1;
                            0 1 0]
                            """)

    complex_game = np.transpose(complex_game)

    complex_salience = [0.206, 0.387, 0.408]

    ################### EXPERIMENT 1 (COMPLEX) - PRODUCTION #####################

    # trigger referent is second row, target message is first in row
    trigger_prod = (0, 0)

    complex_prod_trials = 163

    complex_prod_successes = 84

    ################### EXPERIMENT 1 (SIMPLE) - COMPREHENSION #####################

    # see trigger_prod
    trigger_comp = trigger_prod

    complex_comp_trials = 168

    complex_comp_successes = 90



    ################## FITTING ######################################


    ############# UTILITY FUNCTION = KLD ###################

    s_prob_func, l_prob_func = prob_funcs_given_lambda(complex_game, complex_salience, (trigger_prod, trigger_comp), util_func=log)

    s_likelihood_func = likelhoods_given_lambda(complex_prod_trials, complex_prod_successes, s_prob_func)

    l_likelihood_func = likelhoods_given_lambda(complex_comp_trials, complex_comp_successes, l_prob_func)

    combined_neg_func = neg_func(multiply_funcs(s_likelihood_func, l_likelihood_func))


    s_min_result = minimize(neg_func(s_likelihood_func), 0., method='Nelder-Mead')

    l_min_result = minimize(neg_func(l_likelihood_func), 0., method='Nelder-Mead')

    print("Speaker result:\n", s_min_result, "\n")

    print("Listener result:\n", l_min_result, "\n")


    #### PLOTS ####

    ls = [_ for _ in np.arange(0, 10, 0.01)]

    s_vals = [s_likelihood_func(l) for l in ls]

    print("S_VALS: ", s_vals, "\n")

    l_vals = [l_likelihood_func(l) for l in ls]

    combined_vals = [combined_neg_func(l) for l in ls]

    plt.plot(ls, s_vals, label='S1')

    plt.plot(ls, l_vals, label='R2')

    plt.plot(ls, combined_vals, label='COMB')

    plt.legend()

    plt.show()

    ######################




    ################### UTILITY FUNCTION = JSD ########################

    s_prob_func, l_prob_func = prob_funcs_given_lambda(complex_game, complex_salience, (trigger_prod, trigger_comp), util_func=jsd_non_iterative)

    s_likelihood_func = likelhoods_given_lambda(complex_prod_trials, complex_prod_successes, s_prob_func)

    l_likelihood_func = likelhoods_given_lambda(complex_comp_trials, complex_comp_successes, l_prob_func)

    combined_neg_func = neg_func(multiply_funcs(s_likelihood_func, l_likelihood_func))


    s_min_result = minimize(neg_func(s_likelihood_func), 0., method='Nelder-Mead')

    l_min_result = minimize(neg_func(l_likelihood_func), 0., method='Nelder-Mead')

    print("Speaker result:\n", s_min_result, "\n")

    print("Listener result:\n", l_min_result, "\n")


    #### PLOTS ####

    ls = [_ for _ in np.arange(0, 20, 0.01)]

    s_vals = [s_likelihood_func(l) for l in ls]

    l_vals = [l_likelihood_func(l) for l in ls]

    combined_vals = [combined_neg_func(l) for l in ls]

    # plt.subplot(111)

    plt.plot(ls, s_vals, label='S1')

    plt.plot(ls, l_vals, label='R2')

    plt.plot(ls, combined_vals, label='COMB')

    plt.legend()

    plt.show()

    ######################
    
    ################### UTILITY FUNCTION = Hellinger ########################

    s_prob_func, l_prob_func = prob_funcs_given_lambda(complex_game, complex_salience, (trigger_prod, trigger_comp), util_func=hellinger_non_iterative)

    s_likelihood_func = likelhoods_given_lambda(complex_prod_trials, complex_prod_successes, s_prob_func)

    l_likelihood_func = likelhoods_given_lambda(complex_comp_trials, complex_comp_successes, l_prob_func)

    combined_neg_func = neg_func(multiply_funcs(s_likelihood_func, l_likelihood_func))


    s_min_result = minimize(neg_func(s_likelihood_func), 0., method='Nelder-Mead')

    l_min_result = minimize(neg_func(l_likelihood_func), 0., method='Nelder-Mead')

    print("Speaker result:\n", s_min_result, "\n")

    print("Listener result:\n", l_min_result, "\n")


    #### PLOTS ####

    ls = [_ for _ in np.arange(0, 20, 0.01)]

    s_vals = [s_likelihood_func(l) for l in ls]

    l_vals = [l_likelihood_func(l) for l in ls]

    combined_vals = [combined_neg_func(l) for l in ls]

    # plt.subplot(111)

    plt.plot(ls, s_vals, label='S1')

    plt.plot(ls, l_vals, label='R2')

    plt.plot(ls, combined_vals, label='COMB')

    plt.legend()

    plt.show()

    ######################

    
    

