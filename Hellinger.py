__author__ = 'zweiss'


# Hellinger divergence of two distributions
# dist_a and dist_b [given as vectors of frequencies]
def hellinger(dist_a, dist_b):
    return 1 / 2**.5 * (sum(euclidean(dist_a, dist_b)))**.5


# Euclidean distance of twi distributions
# dist_a and dist_b [given as vectors of frequencies]
def euclidean(dist_a, dist_b):
    if len(dist_a) == len(dist_b):
        for i in range(len(dist_a)):
            yield (dist_a[i] - dist_b[i])**2


# Hellinger divergence implemented non-iteratively assuming
# a degenerated probability distribution P with probability
# 1 on a single event p_i and a uniformly distributed
# # probability distribution Q with each event q = 1/n
# num_messages = the number of messages true of a referent
# -> sum(row of correspondence matrix) or
# 1/uniform distribution
def hellinger_non_iterative_nums(num_messages):
    # dist_a = Pl(X=x) = {1/n, ...}, with len(dist_a) = n
    # dist_b = Ps(X=x) = {1, 0, 0, ...}
    print("hellinger_non_iterative_nums(", num_messages, "): ", (1 / 2**.5) * (((1/num_messages)-1)**2 + ((num_messages-1)*(1/num_messages)**2))**.5)
    return (1 / 2**.5) * (((1/num_messages)-1)**2 + ((num_messages-1)*(1/num_messages)**2))**.5



# Hellinger divergence implemented non-iteratively assuming
# a degenerated probability distribution P with probability
# 1 on a single event p_i and a uniformly distributed
# # probability distribution Q with each event q = 1/n
# num_messages = the number of messages true of a referent
# -> sum(row of correspondence matrix) or
# 1/uniform distribution
def hellinger_non_iterative_probs(prob_messages):
    print("hellinger_non_iterative_probs(", prob_messages, "): ", (1 / 2**.5) * ((prob_messages-1)**2 + (((1/prob_messages)-1) * prob_messages**2))**.5)
    return (1 / 2**.5) * ((prob_messages-1)**2 + (((1/prob_messages)-1) * prob_messages**2))**.5


if __name__ == '__main__':

    # plot where x is the number of messages, and the plotted functions are JSD(x)
    # and log(x) (i.e. KLD), dots on the lines are functions values of common x in the
    # RSA (1, 0.5, 1/3 (-> from the uniform distribution))
    # this is to show the asymptotic behaviour of the two functions
    # and the fact that they are just monotonic transformations of each other
    # and that they are both concave functions and monotonically
    # increasing functions (all of that holds only for 0 < x < 1)

    import matplotlib.pyplot as plt

    num_m = [x for x in range(1, 10000)]

    import numpy as np
    from JSD import jsd_non_iterative, log

    vals = list(np.arange(0.1, 0.999, 0.001))

    jsd_vals = [jsd_non_iterative(x) for x in vals]

    print(log(2))

    kld_vals = [log(x) for x in vals]

    hel_vals = [hellinger_non_iterative_probs(x) for x in vals]

    plt.plot(vals, jsd_vals, label='JSD')

    plt.plot(vals, kld_vals, label='KLD')

    plt.plot(vals, hel_vals, label='HEL')

    # ratio_vals = [jsd_non_iterative(x)/log(x) for x in vals]
    #
    # plt.plot(ratio_vals, label='ratio')


    plt.plot(0.5, jsd_non_iterative(0.5), 'o')

    plt.plot(0.5, log(0.5), 'o')

    plt.plot(1./3, jsd_non_iterative(1./3), 'o')

    plt.plot(1./3, log(1./3), 'o')

    plt.plot(1., jsd_non_iterative(1.), 'o')

    plt.plot(1., jsd_non_iterative(1.), 'o')

    plt.legend()

    plt.show()