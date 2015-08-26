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

    # might be necessary because RSA gives 1/num_messages
    # does not seem to matter -> normalise(x)==normalise(1/x)
    #num_messages = 1/num_messages

    return (log((4*num_messages)/(num_messages+1))-(log(num_messages+1)/num_messages))/2


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

    vals = list(np.arange(0.1, 0.999, 0.001))

    jsd_vals = [jsd_non_iterative(x) for x in vals]

    print(log(2))

    kld_vals = [log(x) for x in vals]

    plt.plot(vals, jsd_vals, label='JSD')

    plt.plot(vals, kld_vals, label='KLD')

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