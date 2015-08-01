__author__ = 'Valentin'

from Utils import read_mat, uniform_dist_discrete
from math import e
from numpy import log
from JSD import jsd

x = 0

def rsa(x_prime):
    global x
    x = x_prime


def listener0(message_index):
    global x
    pass


def speaker1(referent, rationality=1):
    return e**(rationality*log(listener0(referent)))



def listener2(message, salience, rationality=1):
    pass


if __name__ == '__main__':

    m = read_mat('exp1_production.txt')
