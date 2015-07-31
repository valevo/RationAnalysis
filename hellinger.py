__author__ = 'zweiss'

from matplotlib import pyplot as plt


def hellinger(distr1, distr2):
    return 1 / 2**.5 * (sum(euclidean(distr1, distr2)))**.5


def euclidean(distr1, distr2):
    return []
