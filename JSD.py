__author__ = 'Valentin'

from math import e

from matplotlib import pyplot as plt

from math import log


from Utils import uniform_dist_discrete


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

    lo, up = 1, 20

    listener = uniform_dist_discrete(lo, up)

    speaker = uniform_dist_discrete(2,2)

    speaker_dist = list(speaker(x) for x in range(lo, up+1))

    listener_dist = list(listener(x) for x in range(lo, up+1))

    print(len(listener_dist))

    print(jsd(speaker_dist, listener_dist))

    m = list(midpoints(speaker_dist, listener_dist))

    # print('speaker distribution:', speaker_dist)
    #
    # print('listener distribution:', listener_dist)
    #
    # print('Midpoints:', m)
    #
    # print('pointwise divergence(s || m):', list(point_div(speaker_dist, m)))
    #
    # print('pointwise divergence(l || m):', list(point_div(listener_dist, m)))

    print(jsd_non_iterative(up-lo+1))

    import matplotlib.pyplot as plt

    num_m = [x for x in range(1, 1000)]

    jsds = [jsd_non_iterative(x)**.5 for x in num_m]

    # plt.plot(num_m, jsds)
    #
    # plt.show()

    result = jsd_non_iterative(0.5)

    print('test', e**result)

    result = jsd_non_iterative(2)

    print('test', e**result)

    # print(jsd_non_iterative(100000)/jsd_non_iterative(100000000000))

    # xs = [_ for _ in range(50,100000)]
    #
    # u1 = [uniform_dist_discrete(1, x) for x in xs]
    #
    # u2 = uniform_dist_discrete(1, 50)
    #
    # u2_vals = [u2(x) for x in range(1,100000)]
    #
    # print('#####')
    #
    # for u in u1:
    #     cur_u_vals = [u(x) for x in range(1,100000)]
    #
    #     print(kld(u2_vals, cur_u_vals))
    #
    #     print(jsd(u2_vals, cur_u_vals))
    #
    #     print('---')

