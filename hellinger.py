__author__ = 'zweiss'

from matplotlib import pyplot as plt


def hellinger(distr1, distr2):
    return 1 / 2**.5 * (sum(euclidean(distr1, distr2)))**.5


def euclidean(distr1, distr2):
    return []


def gauss_distr(mean, stddev):
    norm_term = stddev*(2*pi)**.5
    norm_term **= -1
    denom_term = 2*(stddev**2)
    return lambda x: norm_term*(e**(float(-(x-mean)**2)/denom_term))


def uniform_distr(lower, upper):
    prob = (upper - lower)**-1
    element_lower_upper = lambda x: True if ((x>=lower) and (x<=upper)) else False
    # change!!!! back to 0
    return lambda x: prob if element_lower_upper(x) else 0.000000001


if __name__ == '__main__':
    g1 = gauss_distr(5, 2)
    g2 = gauss_distr(8, 2)

    xs = list(_ for _ in arange(-2, 13, 0.01))

    vals_g1 = list(g1(x) for x in xs)

    vals_g2 = list(g2(x) for x in xs)

    print(hellinger(vals_g1, vals_g2))

    plt.plot(xs, vals_g1)
    plt.plot(xs, vals_g2)
    plt.plot(xs, g3)

    plt.show()

    u1 = uniform_distr(4, 7)
    u2 = uniform_distr(4, 9)

    vals_u1 = list(u1(x) for x in xs)
    vals_u2 = list(u2(x) for x in xs)

    print(hellinger(vals_u1, vals_u2))

    plt.plot(xs, vals_u1)
    plt.plot(xs, vals_u2)
    plt.plot(xs, u3)

    plt.show()
