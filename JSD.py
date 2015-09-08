__author__ = 'Valentin'

from math import e

from matplotlib import pyplot as plt

from math import log




# Jensen-Shannon divergence of two distributions
# dist_a and dist_b [given as vectors of frequencies]
def jsd(dist_a, dist_b):
    mid = list(midpoints(dist_a, dist_b))
    return (kld(dist_a, mid)+kld(dist_b, mid))*.5


# Midpoints between to distributions dist_a and dist_b
# [given as vectors of frequencies]
# used in jsd
def midpoints(dist_a, dist_b):
    if len(dist_a) == len(dist_b):
        for i in range(len(dist_a)):
            yield (dist_a[i]+dist_b[i])*.5


# Kullback-Leibler divergence of two distributions dist_a
#  and dist_b [given as vectors of frequencies]
def kld(dist_a, dist_b):
    return sum(point_div(dist_a, dist_b))


# Pointwise divergences of dist_a and dist_b
# [given as vectors of frequencies]
# Used in kld
def point_div(dist_a, dist_b):
    if len(dist_a) == len(dist_b):
        for i in range(len(dist_a)):
            if dist_a[i] == 0:
                yield 0
            else:

                yield dist_a[i]*log(dist_a[i]/dist_b[i])


# num_messages = the number of messages true of a referent
# -> sum(row of correspondence matrix) or
# 1/uniform distribution
def jsd_non_iterative(num_messages):

    # might have been necessary but
    # seems to be equivalent to -jsd_non_iterative
    # -> gives same results for MLE, just negated
    num_messages = 1/num_messages

    rval = (log((4*num_messages)/(num_messages+1))-(log(num_messages+1)/num_messages))/2

    # n = num_messages
    #
    # # H[L, M]
    # rval = log((2*n)/(n+1))/n
    # rval += (1-(1/n))*log(2*n)
    #
    # # H[S,M]
    # rval += log((2*n)/(n+1))
    #
    # #H(L)
    # rval += log(n)

    return -rval


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

    vals = list(np.arange(0.001, 1, 0.001))

    jsd_vals = [jsd_non_iterative(x) for x in vals]

    print(log(2))

    kld_vals = [log(x) for x in vals]

    plt.plot(vals, jsd_vals, label='JSD')

    plt.plot(vals, kld_vals, label='KLD')

    from Hellinger import hellinger_non_iterative_probs

    hl_vals = [hellinger_non_iterative_probs(x) for x in vals]

    plt.plot(vals, hl_vals, label='H')

    plt.legend()

    plt.show()

    plt.plot(vals, [jsd_non_iterative(x)/log(x) for x in vals], label='RATIO JSD/KLD')

    plt.plot(vals, [hellinger_non_iterative_probs(x)/log(x) for x in vals], label='RATIO H/KLD')

    plt.plot(vals, [jsd_non_iterative(x)/hellinger_non_iterative_probs(x) for x in vals], label='RATIO JSD/H')

    plt.legend()

    plt.show()



    # v1 = [_ for _ in range(2, 30)]
    #
    # jsd_vals = [jsd_non_iterative(x) for x in v1]
    #
    # plt.plot(v1, jsd_vals)
    #
    # v2 = [1./x for x in v1]
    #
    # kld_vals = [log(x) for x in v2]

    # plt.plot(v2, kld_vals)
    #
    # plt.show()