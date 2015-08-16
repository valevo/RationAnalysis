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


def read_matrix_file(file_str):
    file_str = path.normpath(file_str)
    with codecs.open(file_str, encoding='UTF-8') as handle:
        lines = (line for line in handle)
        mat_str = str.join('', lines)
        return np.matrix(mat_str)


def get_matrix(mat):
    num_cols = len(mat[0,:])-1
    new_mat = mat[:, :num_cols-1]
    ls = mat[:, num_cols-1]
    return new_mat, ls


def get_ratio_matrix(mat):
    matrix, total_nums = get_matrix(mat)
    ratios = [1./cur_total.item() for cur_total in total_nums]
    ratios = np.diag(ratios)
    return ratios*matrix


if __name__ == '__main__':
    m = read_matrix_file('exp1_production.txt')


    new_m, _ = get_matrix(m)

    m = np.matrix("""
                    [135 9 0 0 144]""")

    print(get_ratio_matrix(m))
