__author__ = 'Valentin'

from Utils import read_mat, uniform_dist_discrete
import numpy as np

from math import e
from math import log


def r0(binary_mat):
    # obtain the uniform probability for each row in the binary matrix
    ps = list(1./np.sum(bin_mat[x, :]) for x in range(len(bin_mat[:, 0])))

    #  transform the probabilities into a matrix with non-zero
    # values on the diagonal; used  later for matrix multiplication
    uniform_mat = np.diag(ps)

    # multiply the binary with the diagonal matrix
    # to obtain the actual uniform distribution
    # of referents give messages
    return uniform_mat*binary_mat


def s1(uniform_mat, l=1, c=0):
    # transpose the matrix for later multiplication
    # also need to reverse the conditional probabilities
    s1_mat = np.transpose(uniform_mat)

    # function which calculates the stochastic choice of the messages
    #  given the referents (if their probability if they have positive
    # probability)
    s1_func = lambda t: e**(l*log(t)) if not (t <= 0) else 0#

    # needed to do efficient vectorised calculations,
    # i.e to apply the function in every cell of the matrix
    s1_func = np.vectorize(s1_func)

    # return the matrix which is the result of applying the
    # vectorised function to the matrix
    return s1_func(s1_mat)


def r2(message_mat, salience_ls, l=1, c=0):
    # transpose the matrix for later multiplication
    # also need to reverse the conditional probabilities
    referent_mat = np.transpose(message_mat)

    # create a matrix with salience values on the diagonal
    # (all other values are zero)
    salience_mat = np.diag(salience_ls)

    # multiply the salience priors with the reversed conditional
    #  probabilities (i.e. apply Bayes rule)
    return np.dot(referent_mat, salience_mat)




# utility function; exchanges 0 values in
# a matrix by non-0 ones
# not used
def not_zero(mat):
    f = lambda x: 0.1 if x == 0 else x

    f = np.vectorize(f)

    return f(mat)


if __name__ == '__main__':

    m = read_mat('exp1_production.txt')

    # binary matrix which represents
    # which messages are true of which referent
    # rows => messages, columns => referent
    bin_mat = np.matrix("""
                        [1 0 0;
                        1 1 0;
                        0 1 1;
                        0 0 1]""")

    # impose a uniform distribution
    # on the binary matrix
    r0_mat = r0(bin_mat)

    # use the utilities of the speaker's
    # inferences about the listener's beliefs
    s1_mat = s1(r0_mat)

    # the salience priors
    sal_ls = [.3, .12, .58]

    # the listener's inferences about the
    # the speaker's intentions, also using the
    # salience of referents as the prior in
    # Bayes rules
    r2_mat = r2(s1_mat, sal_ls)

    print(r2_mat)

