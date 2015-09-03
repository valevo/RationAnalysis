__author__ = 'Valentin'

import numpy as np

import matplotlib.pyplot as plt


from RSA import *

from Fit_Model import *

# from Utils import *



from JSD import jsd_non_iterative
from Hellinger import hellinger_non_iterative_probs

if __name__ == '__main__':

    ################### EXPERIMENT 1 (SIMPLE) ################

    simple_game = np.matrix("""
                            [0 1 0 0;
                            1 1 0 0;
                            0 0 1 1]
                            """)

    simple_game = np.transpose(simple_game)

    simple_salience = [0.301, 0.259, 0.440]

    ################### EXPERIMENT 1 (SIMPLE) - PRODUCTION #####################

    # trigger referent is second row, target message is first in row
    trigger_prod = (1, 0)

    simple_prod_trials = 251

    simple_prod_successes = 240

    ################### EXPERIMENT 1 (SIMPLE) - COMPREHENSION #####################

    # see trigger_prod
    trigger_comp = trigger_prod

    simple_comp_trials = 252

    simple_comp_successes = 193



    ################## FITTING ######################################


    ############# UTILITY FUNCTION = KLD ###################

    s_prob_func, l_prob_func = prob_funcs_given_lambda(simple_game, simple_salience, (trigger_prod, trigger_comp), util_func=log)

    s_likelihood_func = likelhoods_given_lambda(simple_prod_trials, simple_prod_successes, s_prob_func)

    l_likelihood_func = likelhoods_given_lambda(simple_comp_trials, simple_comp_successes, l_prob_func)

    combined_neg_func = neg_func(multiply_funcs(s_likelihood_func, l_likelihood_func))


    s_min_result = minimize(neg_func(s_likelihood_func), 0., method='Nelder-Mead')

    l_min_result = minimize(neg_func(l_likelihood_func), 0., method='Nelder-Mead')

    print("Speaker result:\n", s_min_result, "\n")

    print("Listener result:\n", l_min_result, "\n")


    #### PLOTS ####

    ls = [_ for _ in np.arange(0, 10, 0.01)]

    s_vals = [s_likelihood_func(l) for l in ls]

    l_vals = [l_likelihood_func(l) for l in ls]

    combined_vals = [combined_neg_func(l) for l in ls]

    plt.plot(ls, s_vals)

    plt.plot(ls, l_vals)

    plt.plot(ls, combined_vals)

    plt.show()

    ######################

    ################### UTILITY FUNCTION = JSD ########################

    s_prob_func, l_prob_func = prob_funcs_given_lambda(simple_game, simple_salience, (trigger_prod, trigger_comp), util_func=jsd_non_iterative)

    s_likelihood_func = likelhoods_given_lambda(simple_prod_trials, simple_prod_successes, s_prob_func)

    l_likelihood_func = likelhoods_given_lambda(simple_comp_trials, simple_comp_successes, l_prob_func)

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

    plt.plot(ls, s_vals)

    plt.plot(ls, l_vals)

    plt.plot(ls, combined_vals)

    plt.show()

    ######################

    ################### UTILITY FUNCTION = Hellinger ########################

    s_prob_func, l_prob_func = prob_funcs_given_lambda(simple_game, simple_salience, (trigger_prod, trigger_comp), util_func=hellinger_non_iterative_probs)

    s_likelihood_func = likelhoods_given_lambda(simple_prod_trials, simple_prod_successes, s_prob_func)

    l_likelihood_func = likelhoods_given_lambda(simple_comp_trials, simple_comp_successes, l_prob_func)

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

    plt.plot(ls, s_vals)

    plt.plot(ls, l_vals)

    plt.plot(ls, combined_vals)

    plt.show()

    ######################



