__author__ = 'Valentin'

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


# derived version of the JSD using the knowledge about
# R_0 and S_1
def jsd_non_iterative(speaker_prob):
    num_messages = 1/speaker_prob

    rval = (log((4*num_messages)/(num_messages+1))-(log(num_messages+1)/num_messages))/2

    rval *= -1


    ## entropy-based representation of JSD
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

    return rval


if __name__ == '__main__':

    # first plot used in the paper

    import matplotlib.pyplot as plt

    from Hellinger import hellinger_non_iterative_probs


    num_r = [x for x in range(1, 10000)]

    import numpy as np

    vals = list(np.arange(0.001, 1, 0.001))

    jsd_vals = [jsd_non_iterative(x) for x in vals]

    print(log(2))

    kld_vals = [log(x) for x in vals]

    plt.plot(vals, jsd_vals, label='JSD')

    plt.plot(vals, kld_vals, label='KLD')

    hl_vals = [hellinger_non_iterative_probs(x) for x in vals]

    plt.plot(vals, hl_vals, label='H')


    points = [1./x for x in range(1, 10)]

    for p in points:

        plt.plot(p, log(p), 'o', color='k')
        plt.plot(p, jsd_non_iterative(p), 'o', color='k')
        plt.plot(p, hellinger_non_iterative_probs(p), 'o', color='k')

    plt.plot(0, 0, color='k', label='Uniform distributions')

    plt.title('Plots of $D_{KL}$, $D_{JS}$, $H$\nin the range of probabilities')

    plt.xlabel('$p$')

    plt.ylabel('$f(p)$')

    plt.legend()

    plt.show()

    plt.plot(vals, [jsd_non_iterative(x)/log(x) for x in vals], label='JSD/KLD')

    plt.plot(vals, [hellinger_non_iterative_probs(x)/log(x) for x in vals], label='H/KLD')

    plt.plot(vals, [jsd_non_iterative(x)/hellinger_non_iterative_probs(x) for x in vals], label='JSD/H')

    plt.title('Ratios of $D_{KL}$, $D_{JS}$, $H$')

    plt.xlabel('$x$')

    plt.ylabel('$\\frac{f(x)}{g(x)}$')

    plt.legend()

    plt.show()