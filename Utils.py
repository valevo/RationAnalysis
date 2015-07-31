__author__ = 'Valentin'

from numpy import pi

from math import e

def gauss_dist(mean, stddev):
    norm_term = stddev*(2*pi)**.5
    norm_term **= -1
    denom_term = 2*(stddev**2)
    return lambda x: norm_term*(e**(float(-(x-mean)**2)/denom_term))


def uniform_dist_cont(lower, upper):
    prob = (upper - lower)**-1
    element_lower_upper = lambda x: True if ((x>=lower) and (x<=upper)) else False
    # change!!!! back to 0
    return lambda x: prob if element_lower_upper(x) else 0.000000001


def uniform_dist_discrete(lower, upper):
    prob = (upper-lower+1)**-1
    element_lower_upper = lambda x: True if ((x>=lower) and (x<=upper)) else False
    return lambda x: prob if element_lower_upper(x) else 0




import numpy as np

import codecs
from os import path


def read_mat(file_str):
    file_str = path.normpath(file_str)
    with codecs.open(file_str, encoding='UTF-8') as handle:
        lines = (line for line in handle)
        mat_str = str.join('', lines)
        return np.matrix(mat_str)
