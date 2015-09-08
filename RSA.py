__author__ = 'Valentin'

import numpy as np

from math import exp
from math import log

from JSD import jsd_non_iterative
from Hellinger import hellinger_non_iterative_probs

from scipy.optimize import minimize


class RSA:

    # Initialises an RSA object using a semantics matrix
    # (indicates which messages correspond to which referent);
    # allows to set model parameters lambda and costs:
    def __init__(self, correspondence_mat, l=1., c=()):
        self.correspondence_mat = correspondence_mat
        self.listener_belief = correspondence_mat
        self.speaker_choice = np.matrix("")
        self.listener_reason = np.matrix("")
        self.l = l
        self.c = c

    # Returns a uniform distribution over the semantic correspondence
    # which represents R0's belief
    def r0(self):
        ps = list(1./np.sum(self.correspondence_mat[x, :]) if self.correspondence_mat[x, :].any() else 0.
                  for x in range(len(self.correspondence_mat[:, 0])))
        uniform_mat = np.diag(ps)
        self.listener_belief = uniform_mat*self.correspondence_mat
        return self

    # Returns the speaker's choice probabilities for the
    # messages (using the stochastic choice rule and
    # softmax-normalisation;
    # the utility function can be set
    def s1(self, utility_func=log, normalise=True):
        speaker_choice = np.transpose(self.listener_belief)
        speaker_func = lambda t: exp(self.l*utility_func(t)) if (t > 0) else 0.
        speaker_func = np.vectorize(speaker_func)
        speaker_choice = speaker_func(speaker_choice)
        # print("SPEAKER CHOICE:\n", speaker_choice, " END")
        # normalisation needed
        if normalise:
            self.speaker_choice = RSA.softmax_normalise(speaker_choice)
        else:
            self.speaker_choice = speaker_choice
        return self

    # The listener's reasoning about the speaker's choices;
    # uses Bayes' Rule (the salience matrix is the prior);
    # normalisation is done by dividing by the likelihood
    # of the referent
    def r2(self, salience_list, normalise=True):
        referent_mat = np.transpose(self.speaker_choice)

        if normalise:
            salience_arr = np.asarray(salience_list)
            norm = [1./np.dot(row, salience_arr).item() if row.any() else 0.
                    for row in referent_mat]
            norm = np.diag(norm)
            referent_mat = norm*referent_mat

        salience_mat = np.diag(salience_list)
        self.listener_reason = referent_mat * salience_mat

        return self

    # alternative version for r2 (does the calculations in an iterative way
    # instead of using matrix multiplications)
    def r22(self, salience_list, normalise=True):
        referent_mat = np.transpose(self.speaker_choice)

        num_rows = range(np.shape(referent_mat)[0])

        num_cols = range(len(referent_mat)-1)

        for row in num_rows:
            for col in num_cols:
                referent_mat[row, col] *= salience_list[col]

        for row in num_rows:
            norm_sum = 0
            for col in num_cols:
                norm_sum += referent_mat[row, col]
            for col in num_cols:
                referent_mat[row, col] /= norm_sum

        self.listener_reason = referent_mat
        return self


    # unused
    def softmax(vector, l=1):
        exp_func = lambda x: exp(l*x)
        exp_vector = [exp_func(x) for x in vector]
        norm_term = sum(exp_vector)
        norm_func = lambda x: x/norm_term
        return [norm_func(x) for x in exp_vector]

    # Normalises a matrix by rows using the
    # soft-max choice rule
    def softmax_normalise(matrix, l=1):
        # exp_func = lambda x: exp(l*x) if x > 0 else 0.
        # exp_func = np.vectorize(exp_func)
        # softmax_mat = exp_func(matrix)
        norm_term_ls = [1./np.sum(row) for row in matrix]
        norm_mat = np.diag(norm_term_ls)
        return norm_mat*matrix

if __name__ == '__main__':


    # example of the RSA model on the complex game (for the simple game
    # take the other matrix) -> produces all the relevant matrices
    # to show what RSA does

    complex_game = np.matrix("""
                           [1 1 0;
                            1 0 1;
                            0 1 0]
                            """)
    #
    simple_game = np.matrix("""
                           [0 1 0;
                            1 1 0;
                            0 0 1;
                            0 0 1]
                            """)

    # simple_game = np.transpose(simple_game)

    # create RSA object
    rsa1 = RSA(simple_game)

    # get the dummy listener's belief
    l_belief = rsa1.r0()

    print('L_BELIEF:\n', rsa1.listener_belief, '\nEND')


    ### KLD ###

    # get the speaker's choice probabilities
    # (using KLD as the utility function)
    s_choice = rsa1.s1()

    print('S_CHOICE (KLD):\n', rsa1.speaker_choice, '\nEND')

    l_reason = rsa1.r2([.3, .12, .58])

    print('L_REASON:\n', rsa1.listener_reason, '\nEND')


    ### JSD ###

    # get the speaker's choice probabilities
    # (using the JSD as the utility function)
    s_choice = rsa1.s1(jsd_non_iterative)

    print('S_CHOICE (JSD):\n', rsa1.speaker_choice, '\nEND')

    l_reason = rsa1.r2([.3, .12, .58])

    print('L_REASON:\n', rsa1.listener_reason, '\nEND')


    ### Hellinger ###

    # get the speaker's choice probabilities
    # (using the Hellinger divergence as the utility function)
    s_choice = rsa1.s1(hellinger_non_iterative_probs)

    print('S_CHOICE (Hellinger):\n', rsa1.speaker_choice, '\nEND')

    l_reason = rsa1.r2([.3, .12, .58])

    print('L_REASON:\n', rsa1.listener_reason, '\nEND')

