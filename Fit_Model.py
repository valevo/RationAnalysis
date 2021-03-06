__author__ = 'Valentin'

from RSA import *

# required in other files
from scipy.optimize import minimize

from scipy.misc import comb

# implementation of the binomial distribution function;
# returns a function which takes the number of successes as an input
# and returns the probability of this number of successes given
# the number of trials and the probability of a success
def binomial(num_trials, success_prob):
    return lambda num_successes: comb(num_trials, num_successes)*(success_prob**num_successes)*((1-success_prob)**(num_trials-num_successes))


# gets the likelihood for the probability predicted by the RSA
# given the number of trials and the number of successes;
# returns a function over lambda (which is then input for the function
# which returns the corresponding RSA model
def likelhoods_given_lambda(num_trials, num_successes, prob_given_lambda):
    return lambda l: binomial(num_trials, prob_given_lambda(l))(num_successes)


# negates the function by multiplying its
# f(x) with -1
def neg_func(func):
    return lambda x: -1.*func(x)


# adds the values returned by functions
# and returns a function over that sum
# in the fashion: f1(x) + f2(x) + ...
def add_funcs(*funcs):
    func_sum = lambda x: sum(func(x) for func in funcs)
    return func_sum


# multiplies the values returned by functions
# and return a function over that product
# in the fashion: f1(x) * f2(x) * ...
def multiply_funcs(*funcs):
    def func_product(x):
        prod = 1
        for func in funcs:
            prod *= func(x)
        return prod
    return func_product


# constructs a function over RSA objects given a lambda
def rsa_given_lambda(semantics_mat, saliences, util_func=log):

    rsa_obj = RSA(semantics_mat)

    rsa_obj.r0()

    # returns a an RSA object given a specific lambda
    def func_over_lambda(lamb):
        rsa_obj.l = lamb

        rsa_obj.s1(util_func)

        return rsa_obj.r2(saliences)

    # return function over RSA objects
    return func_over_lambda


# extracts the matrix for the speaker's choice probabilities
# from a function over RSA objects;
# returns a function over the matrices
def speaker_choice_given_lambda(rsa_func):
    return lambda l: rsa_func(l).speaker_choice


# extracts the matrix for the listener's inferences
# from a function over RSA objects;
# returns a function over the matrices
def listener_reason_given_lambda(rsa_func):
    return lambda l: rsa_func(l).listener_reason


# extracts the probability index by trigger from
# a binomial matrix of probabilities (the value
# is taken as the probability of a success in a binomial distribution)
# returns a function over the probability
def prob_from_mat_func(mat_func, trigger):
    return lambda l: mat_func(l)[trigger[0], trigger[1]].item()


# combines some of the functions by first getting a function over
# RSA object and then extracting the success probabilities (taken to be the probability
# of choosing the target) of both speaker and listener predicted by the model;
# returns a tuple: the function over the speaker's success probabilities and the
# function over the listener's success probabilities
def prob_funcs_given_lambda(semantics_mat, saliences, triggers, util_func=log):
    rsa_obj_func = rsa_given_lambda(semantics_mat, saliences, util_func)

    speaker_func = speaker_choice_given_lambda(rsa_obj_func)

    listener_func = listener_reason_given_lambda(rsa_obj_func)

    speaker_prob_func = prob_from_mat_func(speaker_func, triggers[0])

    listener_prob_func = prob_from_mat_func(listener_func, triggers[1])

    return speaker_prob_func, listener_prob_func



