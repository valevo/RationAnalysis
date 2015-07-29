__author__ = 'Valentin'

from math import pi
from math import e

from numpy import arange

from matplotlib import pyplot as plt

from scipy import log

def jsd(dist_a, dist_b):
    mid = list(midpoints(dist_a, dist_b))
    return (kld(dist_a, mid)+kld(dist_b, mid))*.5

def midpoints(dist_a, dist_b):
    if len(dist_a) == len(dist_b):
        for i in range(len(dist_a)):
            print(dist_a[i] > dist_a[i]+dist_b[i]*.5, dist_b[i] > dist_a[i]+dist_b[i]*.5)
            yield (dist_a[i]+dist_b[i])*.5

def kld(dist_a, dist_b):
    return sum(point_div(dist_a, dist_b))

def point_div(dist_a, dist_b):
    if len(dist_a) == len(dist_b):
        for i in range(len(dist_a)):
            yield dist_a[i]*log(dist_a[i]/dist_b[i])


def gauss_dist(mean, stddev):
    norm_term = stddev*(2*pi)**.5
    norm_term **= -1
    denom_term = 2*(stddev**2)
    return lambda x: norm_term*(e**(float(-(x-mean)**2)/denom_term))


def uniform_dist(lower, upper):
    prob = (upper - lower)**-1
    element_lower_upper = lambda x: True if ((x>=lower) and (x<=upper)) else False
    # change!!!! back to 0
    return lambda x: prob if element_lower_upper(x) else 0.000000001


if __name__ == '__main__':
    g1 = gauss_dist(5, 2)
    g2 = gauss_dist(8, 2)

    xs = list(_ for _ in arange(-2, 13, 0.01))

    vals_g1 = list(g1(x) for x in xs)

    vals_g2 = list(g2(x) for x in xs)

    g3 = list(midpoints(vals_g1, vals_g2))

    print(jsd(vals_g1, vals_g2))

    plt.plot(xs, vals_g1)
    plt.plot(xs, vals_g2)
    plt.plot(xs, g3)

    plt.show()

    u1 = uniform_dist(4, 7)
    u2 = uniform_dist(4, 9)

    vals_u1 = list(u1(x) for x in xs)
    vals_u2 = list(u2(x) for x in xs)

    u3 = list(midpoints(vals_u1, vals_u2))

    print(jsd(vals_u1, vals_u2))

    plt.plot(xs, vals_u1)
    plt.plot(xs, vals_u2)
    plt.plot(xs, u3)

    plt.show()


