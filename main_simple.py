__author__ = 'Valentin'

import matplotlib.pyplot as plt

from Fit_Model import *

from JSD import jsd_non_iterative
from Hellinger import hellinger_non_iterative_probs

def get_simple_likelihood_funcs(utility_func):
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

    s_prob_func, l_prob_func = prob_funcs_given_lambda(simple_game, simple_salience, (trigger_prod, trigger_comp), util_func=utility_func)

    simple_s_likelihood_func = likelhoods_given_lambda(simple_prod_trials, simple_prod_successes, s_prob_func)

    simple_l_likelihood_func = likelhoods_given_lambda(simple_comp_trials, simple_comp_successes, l_prob_func)

    return simple_s_likelihood_func, simple_l_likelihood_func



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

    simple_s_likelihood_func = likelhoods_given_lambda(simple_prod_trials, simple_prod_successes, s_prob_func)

    simple_l_likelihood_func = likelhoods_given_lambda(simple_comp_trials, simple_comp_successes, l_prob_func)

    combined_neg_func = neg_func(multiply_funcs(simple_s_likelihood_func, simple_l_likelihood_func))


    simple_s_min_result = minimize(neg_func(simple_s_likelihood_func), 0., method='Nelder-Mead')

    simple_l_min_result = minimize(neg_func(simple_l_likelihood_func), 0., method='Nelder-Mead')

    print("Speaker result:\n", simple_s_min_result, "\n")

    print("Listener result:\n", simple_l_min_result, "\n")


    #### PLOTS ####

    ls = [_ for _ in np.arange(0, 10, 0.01)]

    s_vals = [simple_s_likelihood_func(l) for l in ls]

    l_vals = [simple_l_likelihood_func(l) for l in ls]

    combined_vals = [combined_neg_func(l) for l in ls]

    plt.plot(ls, s_vals, label='Speaker')

    plt.plot(ls, l_vals, label='Listener')

    # plt.plot(ls, combined_vals)

    plt.ylabel('$P(Data|\lambda)$')

    plt.xlabel('$\lambda$')

    plt.title('MLE of Lambda in the simple condition\n $U_S = -D_{KL}$')

    plt.legend()

    plt.show()

    ######################

    ################### UTILITY FUNCTION = JSD ########################

    s_prob_func, l_prob_func = prob_funcs_given_lambda(simple_game, simple_salience, (trigger_prod, trigger_comp), util_func=jsd_non_iterative)

    simple_s_likelihood_func = likelhoods_given_lambda(simple_prod_trials, simple_prod_successes, s_prob_func)

    simple_l_likelihood_func = likelhoods_given_lambda(simple_comp_trials, simple_comp_successes, l_prob_func)

    combined_neg_func = neg_func(multiply_funcs(simple_s_likelihood_func, simple_l_likelihood_func))


    simple_s_min_result = minimize(neg_func(simple_s_likelihood_func), 0., method='Nelder-Mead')

    simple_l_min_result = minimize(neg_func(simple_l_likelihood_func), 0., method='Nelder-Mead')

    print("Speaker result:\n", simple_s_min_result, "\n")

    print("Listener result:\n", simple_l_min_result, "\n")


    #### PLOTS ####

    ls = [_ for _ in np.arange(0, 20, 0.01)]

    s_vals = [simple_s_likelihood_func(l) for l in ls]

    l_vals = [simple_l_likelihood_func(l) for l in ls]

    combined_vals = [combined_neg_func(l) for l in ls]

    plt.plot(ls, s_vals, label='Speaker')

    plt.plot(ls, l_vals, label='Listener')

    # plt.plot(ls, combined_vals)

    plt.ylabel('$P(Data|\lambda)$')

    plt.xlabel('$\lambda$')

    plt.title('MLE of Lambda in the simple condition\n $U_S = -D_{JS}$')

    plt.legend()

    plt.show()

    ######################

    ################### UTILITY FUNCTION = Hellinger ########################

    s_prob_func, l_prob_func = prob_funcs_given_lambda(simple_game, simple_salience, (trigger_prod, trigger_comp), util_func=hellinger_non_iterative_probs)

    simple_s_likelihood_func = likelhoods_given_lambda(simple_prod_trials, simple_prod_successes, s_prob_func)

    simple_l_likelihood_func = likelhoods_given_lambda(simple_comp_trials, simple_comp_successes, l_prob_func)

    combined_neg_func = neg_func(multiply_funcs(simple_s_likelihood_func, simple_l_likelihood_func))


    simple_s_min_result = minimize(neg_func(simple_s_likelihood_func), 0., method='Nelder-Mead')

    simple_l_min_result = minimize(neg_func(simple_l_likelihood_func), 0., method='Nelder-Mead')

    print("Speaker result:\n", simple_s_min_result, "\n")

    print("Listener result:\n", simple_l_min_result, "\n")


    #### PLOTS ####

    ls = [_ for _ in np.arange(0, 20, 0.01)]

    s_vals = [simple_s_likelihood_func(l) for l in ls]

    l_vals = [simple_l_likelihood_func(l) for l in ls]

    combined_vals = [combined_neg_func(l) for l in ls]

    plt.plot(ls, s_vals, label='Speaker')

    plt.plot(ls, l_vals, label='Listener')

    plt.ylabel('$P(Data|\lambda)$')

    plt.xlabel('$\lambda$')

    plt.title('MLE of Lambda in the simple condition\n $U_S = -H$')

    plt.legend()

    plt.show()

    ######################


